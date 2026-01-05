import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from event_sam3d.config import IS_CLUSTER, MVSEC_SCENES, PROJ_DIR, RELATED_DIR
from event_sam3d.datasets.ie_dataset import IEDataset
from event_sam3d.datasets.mvsec_ds import MVSECDataset
from event_sam3d.datasets.rgbe_ds import RGBEDataset
from event_sam3d.img2event.model import TeacherStudent
from event_sam3d.img2event.model_utils import get_condition_embedder
from event_sam3d.img2event.utils import (
    EarlyStopping,
    cleanup_distributed,
    compute_embed_loss,
    init_distributed,
    is_main_process,
    load_st_models,
    reduce_dict,
    reduce_tensor,
)
from event_sam3d.utils.common_utils import (
    adjust_depth_for_plt,
    adjust_img_for_plt,
    detach_and_cpu,
)
from event_sam3d.utils.misc_utils import print_args, set_seed


def make_vis_loaders(train_dataset, val_dataset, cfg):
    # fixed small subsets
    train_indices = np.linspace(0, len(train_dataset) - 1, 4).astype(int).tolist()
    val_indices = np.linspace(0, len(val_dataset) - 1, 4).astype(int).tolist()

    train_vis_ds = Subset(train_dataset, train_indices)
    val_vis_ds = Subset(val_dataset, val_indices)

    train_vis_loader = DataLoader(
        train_vis_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    val_vis_loader = DataLoader(
        val_vis_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    return train_vis_loader, val_vis_loader


def build_model(
    cfg, rank=0, device="cuda", block_idxs=None, use_only_encoder=True, is_train=True
):
    """
    Return a regular (non-DDP) model.
    """
    if block_idxs is None:
        block_idxs = [2] if cfg.do_debug else [2, 5, 9, 14, 19, 22]
    s_model, t_model = load_st_models(
        cfg,
        block_idxs=block_idxs,
        rank=rank,
        device=device,
        is_train=is_train,
        use_only_encoder=use_only_encoder,
    )
    forward_args = None
    ts_model = TeacherStudent(
        s=s_model,
        t=t_model,
        block_idxs=block_idxs,
    )
    return {"model": ts_model, "forward_args": forward_args}


def build_datasets(cfg):
    """
    Return train_dataset, val_dataset.
    """
    val_ds_names = cfg.val_ds_names
    train_ds_names = [s for s in MVSEC_SCENES if s not in val_ds_names]
    obj_names = cfg.obj_names
    if cfg.do_overfit:
        train_ds_names = train_ds_names[:1]
        val_ds_names = train_ds_names[:1]
        obj_names = obj_names[:1]
        len_limit = cfg.overfit_n_samples
    else:
        len_limit = None
    train_datasets = {}
    val_datasets = {}
    ds_cls = MVSECDataset if cfg.ds_name == "mvsec" else RGBEDataset
    for obj_name in obj_names:
        common_kwargs = dict(
            obj_name=obj_name,
            use_masks=True,
            use_vg_event_repr=True,
            len_limit=len_limit,
            include_only_if_enough_events=cfg.include_only_if_enough_events,
            min_num_events=cfg.min_num_events,
        )
        for filename in train_ds_names:
            if cfg.ds_name == "mvsec":
                other_kwargs = dict(
                    seq_name=filename,
                )
            else:
                other_kwargs = dict(split='train')
            dataset = ds_cls(
                transform_names=cfg.transform_names,
                **other_kwargs,
                **common_kwargs,
            )
            train_datasets[f"{filename}_{obj_name}"] = dataset
        for filename in val_ds_names:
            if cfg.ds_name == "mvsec":
                other_kwargs = dict(
                    seq_name=filename,
                )
            else:
                other_kwargs = dict(split='test-normal', test_subsplit=filename)
            dataset = ds_cls(
                transform_names=cfg.transform_names,
                **other_kwargs,
                **common_kwargs,
            )
            val_datasets[f"{filename}_{obj_name}"] = dataset
    train_ds = IEDataset(datasets=train_datasets)
    val_ds = IEDataset(datasets=val_datasets)
    if cfg.do_debug:
        train_ds = torch.utils.data.Subset(train_ds, range(cfg.batch_size * 2))
        val_ds = torch.utils.data.Subset(val_ds, range(cfg.batch_size * 2))
    return train_ds, val_ds


def build_optimizer(model, cfg, use_scheduler=False):
    """
    Return optimizer (and optionally scheduler).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
    else:
        scheduler = None
    return optimizer, scheduler


class Trainer:

    def __init__(self, model, cfg, do_hist=False):
        self.do_hist = do_hist
        self.cfg = cfg
        self.model = model
        self.model_wo_ddp = model.module if isinstance(model, DDP) else model
        self.hist = defaultdict(list)

    def ts_forward(
        self,
        batch,
        forward_kwargs=None,
    ):

        s_kwargs = dict(
            image=batch["rgb"],
            mask=batch["mask"],
            seed=42,
            event_image=batch["events"],
        )
        t_kwargs = dict(
            image=batch.get("rgb_clean", batch["rgb"]),
            mask=batch["mask"],
            seed=42,
            event_image=None,
        )
        results_dict = self.model(s_kwargs=s_kwargs, t_kwargs=t_kwargs)
        results_dict["meta"] = {}

        return results_dict

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        scaler,
        epoch,
        device,
        rank,
        world_size,
        cfg,
        forward_args,
    ):
        self.model.train()

        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(
            tqdm(
                train_loader,
                disable=not is_main_process(rank),
                desc="Train",
                leave=True,
            )
        ):
            # If using DistributedSampler, set epoch for proper shuffling
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                outputs = self.ts_forward(
                    batch=batch,
                    forward_kwargs=forward_args,
                )
                losses = self.calc_losses(outputs)
                loss = losses["loss"]
            if self.do_hist:
                for k, v in losses.items():
                    self.hist[k].append(detach_and_cpu(v))

            scaler.scale(loss).backward()
            # Optional: gradient clipping
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Log / stats
            running_loss += loss.detach()
            num_batches += 1

            if (step + 1) % cfg.log_step_freq == 0 and is_main_process(rank):
                avg_loss = running_loss / num_batches
                # current_lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train/loss_step": avg_loss.item(),
                        "step": epoch * len(train_loader) + step,
                    },
                )

        self.model_wo_ddp.s_embeds.clear()
        self.model_wo_ddp.t_embeds.clear()

        # Average loss across processes
        avg_loss = running_loss / max(len(train_loader), 1)
        avg_loss = reduce_tensor(avg_loss, world_size)

        if is_main_process(rank):
            wandb.log(
                {
                    "train/loss_epoch": avg_loss.item(),
                    "epoch": epoch,
                },
            )

        return avg_loss.item()

    def calc_losses(self, outputs):
        embed_losses = compute_embed_loss(
            outputs["s_feats"],
            outputs["t_feats"],
            use_attn="lattn" in self.cfg.exp_name,
        )
        losses = {"embed_losses": embed_losses["losses"]}
        total_loss = embed_losses["total_loss"]
        if "loss-rec" in self.cfg.exp_name:
            rec_loss = F.l1_loss(
                outputs["s_pred"]["flow_preds"][-1], outputs["t_pred"]["flow_preds"][-1]
            )
            total_loss += rec_loss
            losses["rec_loss"] = rec_loss.item()
        losses["loss"] = total_loss
        return losses

    @torch.no_grad()
    def validate(self, val_loader, cfg, forward_args, epoch=0, device="cuda", rank=0):
        self.model.eval()
        running_losses = defaultdict(float)

        for batch in tqdm(
            val_loader, disable=not is_main_process(rank), desc="Val", leave=True
        ):
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                outputs = self.ts_forward(
                    batch=batch,
                    forward_kwargs=forward_args,
                )
                losses = self.calc_losses(outputs)

            for k, v in losses.items():
                if k in ["loss", "rec_loss"]:
                    running_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

        avg_running_losses = reduce_dict(
            {k: v / len(val_loader) for k, v in running_losses.items()}
        )

        if is_main_process(rank):
            wandb.log(
                {
                    **{f"val/{k}_epoch": v for k, v in avg_running_losses.items()},
                    "epoch": epoch,
                }
            )

        return avg_running_losses

    @torch.no_grad()
    def inference_step(self, batch, cfg, forward_args, device="cuda"):
        self.model.eval()

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            outputs = self.ts_forward(
                batch=batch,
                forward_kwargs=forward_args,
            )
            losses = self.calc_losses(outputs)
        return {
            "s_pred": outputs["s_pred"],
            "t_pred": outputs["t_pred"],
            "losses": losses,
        }

    @torch.no_grad()
    def bench_visuals(self, vis_loaders, epoch, device, cfg, forward_args):
        self.model.eval()

        train_loader, val_loader = vis_loaders
        assert len(train_loader) == len(val_loader) == 1, (
            len(train_loader),
            len(val_loader),
        )
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        res = {}
        for alias, batch in zip(["train", "val"], [train_batch, val_batch]):

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                outputs = self.ts_forward(
                    batch=batch,
                    forward_kwargs=forward_args,
                )
                losses = self.calc_losses(outputs)
                loss = losses["loss"]

            wandb.log(
                {
                    f"bench_{alias}/loss_epoch": loss.item(),
                    f"epoch": epoch,
                }
            )
            res[f"{alias}_loss"] = loss.item()

        return res


def main(cfg):
    set_seed(seed=42)
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        print(f"{world_size=}, {rank=}, {local_rank=}\n")

    ckpt_dir = Path(f"{cfg.ckpt_dir}/{cfg.exp_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process(rank):
        run = wandb.init(
            project="img2event_sam3d",
            name=cfg.exp_name,
            config=cfg,
            mode=None if cfg.use_wandb else "disabled",
            save_code=True,
        )
        wandb.define_metric("step")
        wandb.define_metric("epoch")
        wandb.run.log_code(".")
        wandb.run.log_code(
            f"{PROJ_DIR}/event_sam3d/img2event",
            include_fn=lambda path: path.endswith(".py"),
        )
        wandb.run.log_code(
            f"{RELATED_DIR}/rec/sam-3d-objects/sam3d_objects/pipeline",
            include_fn=lambda path: path.endswith(".py"),
        )
        cli_args = " ".join(sys.argv[1:])
        cfg_vars = vars(cfg)
        meta = {
            "cli_args": cli_args,
            "wandb_run_id": wandb.run.id,
        }
        print(cli_args)
        print_args(cfg, logger=None)
        cfg_vars["meta"] = meta
        with open(os.path.join(ckpt_dir, "config.yml"), "w") as file:
            yaml.dump(cfg_vars, file)
        wandb.run.save(os.path.join(ckpt_dir, "config.yml"))
        if IS_CLUSTER:
            wandb.log(
                {"jobid": os.environ.get("SLURM_JOB_ID", "jupyter"), "cluster": 1}
            )

    # Build datasets (non-distributed)
    train_dataset, val_dataset = build_datasets(cfg)

    # Distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and not cfg.do_overfit),
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    vis_loaders = make_vis_loaders(train_dataset, val_dataset, cfg)

    if is_main_process(rank):
        print(f"# CLI command:\npython {' '.join(sys.argv)}")
        print(f"# Experiment created at {wandb.run.url}")
        print(f"# {ckpt_dir=}")
        print(f"# TRAIN DS:\n{train_dataset}\n{len(train_dataset)=}")
        print(f"# VAL DS:\n{val_dataset}\n{len(val_dataset)=}")
        if "SLURM_JOB_ID" in os.environ:
            print(f"# SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}")

    # Model & optimizer
    build_model_res = build_model(cfg, rank=rank)
    model, forward_args = (
        build_model_res["model"].to(device),
        build_model_res["forward_args"],
    )
    model_noddp = model.module if isinstance(model, DDP) else model
    optimizer, scheduler = build_optimizer(model, cfg, use_scheduler=cfg.use_scheduler)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    # Optionally resume_path
    start_epoch = 0
    end_epoch = cfg.epochs
    best_val_loss = float("inf")

    if cfg.resume_dir is not None and os.path.exists(f"{cfg.resume_dir}/state.pt"):
        state = torch.load(f"{cfg.resume_dir}/state.pt", map_location="cpu")
        start_epoch = state["epoch"] + 1
        end_epoch = start_epoch + cfg.epochs
        best_val_loss = state["best_val_loss"]

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    trainer = Trainer(model, cfg)
    early_stopping = EarlyStopping(
        patience=max(1, cfg.es_patience_epochs // cfg.val_epoch_freq),
        delta=cfg.es_delta,
        verbose=True,
    )
    pbar = tqdm(
        range(start_epoch, end_epoch), disable=not is_main_process(rank), desc="Epochs"
    )
    for epoch in pbar:
        train_loss = trainer.train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            device=device,
            rank=rank,
            world_size=world_size,
            cfg=cfg,
            forward_args=forward_args,
        )

        if scheduler is not None:
            scheduler.step()

        desc = f"{epoch=}. Train Loss: {train_loss:.4f}"
        if not cfg.do_overfit and (
            epoch % cfg.val_epoch_freq == 0 or (epoch == end_epoch - 1)
        ):
            val_losses = trainer.validate(
                val_loader=val_loader,
                cfg=cfg,
                forward_args=forward_args,
                epoch=epoch,
                device=device,
                rank=rank,
            )
            val_loss = val_losses["loss"]
            desc += f" | Val Loss: {val_loss:.4f}"

            if is_main_process(rank):
                bench_res = trainer.bench_visuals(
                    epoch=epoch,
                    vis_loaders=vis_loaders,
                    device=device,
                    cfg=cfg,
                    forward_args=forward_args,
                )
                desc += " | (Bench) " + " | ".join(
                    [f"{k}: {v:.4f}" for k, v in bench_res.items()]
                )
                early_stopping(loss=val_loss)

            if is_main_process(rank) and cfg.do_save_ckpt:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                if is_best:
                    state = {
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    }

                    save_ckpt(ckpt_dir, model_noddp, state=state)

        if world_size > 1:
            dist.barrier()

        pbar.set_description(desc)

        if early_stopping.do_stop:
            print(f"WARN: Early stopping on epoch {epoch}")
            break

    if cfg.do_overfit and cfg.do_save_ckpt:
        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        save_ckpt(ckpt_dir=ckpt_dir, model_noddp=model_noddp, state=state)

    if is_main_process(rank):
        wandb.finish()

    cleanup_distributed()


def save_ckpt(ckpt_dir, model_noddp, state=None):
    ss_generator_cond_embedder_ckpt = {
        "state_dict": get_condition_embedder(model_noddp.s).state_dict()
    }
    rgbe_fuser_ckpt = {
        "state_dict": model_noddp.s.condition_embedders[
            "ss_condition_embedder"
        ].rgbe_fuser.state_dict()
    }
    best_ss_generator_cond_embedder_path = (
        ckpt_dir / "best_ss_generator_cond_embedder.pt"
    )
    best_rgbe_fuser_path = ckpt_dir / "best_rgbe_fuser.pt"
    torch.save(
        ss_generator_cond_embedder_ckpt,
        best_ss_generator_cond_embedder_path,
    )
    torch.save(rgbe_fuser_ckpt, best_rgbe_fuser_path)
    if state is not None:
        state_path = ckpt_dir / "state.pt"
        torch.save(state, state_path)


def get_arg_parser():
    p = argparse.ArgumentParser()
    train_args = p.add_argument_group("training")
    train_args.add_argument("--epochs", type=int, default=50)
    train_args.add_argument("--batch_size", type=int, default=4)
    train_args.add_argument("--num_workers", type=int, default=4)
    train_args.add_argument("--lr", type=float, default=5e-5)
    train_args.add_argument("--weight_decay", type=float, default=0.0)
    train_args.add_argument("--use_amp", action="store_true")
    train_args.add_argument("--use_scheduler", action="store_true")
    train_args.add_argument("--grad_clip", type=float, default=1.0)
    train_args.add_argument("--es_patience_epochs", type=int, default=20)
    train_args.add_argument("--es_delta", type=float, default=0.0)

    model_args = p.add_argument_group("model")
    model_args.add_argument(
        "--rgbe_fusion_type", default="gated", choices=["gated", "attn", "cattn"]
    )

    data_args = p.add_argument_group("data")
    data_args.add_argument("--val_ds_names", nargs="+", default=["indoor_flying4_data"])
    data_args.add_argument("--event_window_ms", type=int, default=50)
    data_args.add_argument("--obj_names", nargs="+", default=["barrel"])
    data_args.add_argument("--transform_names", nargs="*")
    data_args.add_argument("--include_only_if_enough_events", action="store_true")
    data_args.add_argument("--min_num_events", type=int, default=500)
    data_args.add_argument("--ds_name", default="mvsec", choices=["mvsec", "rgbe"])

    pipe_args = p.add_argument_group("pipeline")
    pipe_args.add_argument("--log_step_freq", type=int, default=20)
    pipe_args.add_argument("--val_epoch_freq", type=int, default=1)
    pipe_args.add_argument("--ckpt_dir", default=f"{PROJ_DIR}/checkpoints")
    pipe_args.add_argument("--resume_dir", help="Dir of a previous experiment")
    pipe_args.add_argument("--exp_name", default="test")
    pipe_args.add_argument("--use_wandb", action="store_true")
    pipe_args.add_argument("--do_save_ckpt", action="store_true")
    pipe_args.add_argument("--do_debug", action="store_true")
    pipe_args.add_argument("--do_overfit", action="store_true")
    pipe_args.add_argument("--overfit_n_samples", type=int, default=100)

    return p


if __name__ == "__main__":
    sys.path.append(f"{RELATED_DIR}/slam/egsslam")
    p = get_arg_parser()
    cfg = p.parse_args()
    if cfg.use_amp:
        cfg.exp_name += "_amp"
    if cfg.do_overfit:
        cfg.exp_name += "_overfit"
    if cfg.do_debug:
        cfg.exp_name += "_debug"
    if cfg.resume_dir:
        cfg.exp_name += "_resume"
    if cfg.event_window_ms != p.get_default("event_window_ms"):
        cfg.exp_name += f"_windowms-{cfg.event_window_ms}"
    if cfg.include_only_if_enough_events:
        cfg.exp_name += f"_min-events-{cfg.min_num_events}"
    cfg.exp_name += f"_fusion-{cfg.rgbe_fusion_type}"
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = cfg.exp_name + f"_{current_datetime}"
    if exp_name.startswith("_"):
        exp_name = exp_name[1:]
    cfg.exp_name = exp_name.replace("__", "_")
    main(cfg)
