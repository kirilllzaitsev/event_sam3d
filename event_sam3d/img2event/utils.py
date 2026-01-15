import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import is_tensor

from event_sam3d.config import IS_CLUSTER, RELATED_DIR
from event_sam3d.utils.common_utils import cast_to_numpy
from event_sam3d.utils.misc_utils import is_empty


def compute_sparse_sam3d_loss(out_s, out_t):
    res = {}

    for k in ["6drotation_normalized", "scale", "translation"]:
        pred = out_s[k]
        gt = out_t[k]
        res[f"loss_{k}"] = F.mse_loss(pred, gt)

    pred_vg = out_s["ss"]
    gt_vg = out_t["ss"]
    bce = F.binary_cross_entropy_with_logits(
        pred_vg, (gt_vg > 0).float(), reduction="none"
    )
    pt = torch.exp(-bce)
    alpha = 0.25
    gamma = 2.0
    focal_loss = (alpha * (1 - pt) ** gamma * bce).mean()
    res["loss_ss"] = focal_loss
    return res


def compute_embed_loss(s_embeds, t_embeds, use_attn=False):
    # embeds=scales x layers
    losses = defaultdict(lambda: defaultdict(list))
    total_loss = 0.0
    count = 0
    for lname in s_embeds.keys():
        if "_attn" in lname:
            pass
        else:
            for input_cond_idx in range(len(s_embeds[lname])):
                if "rgbe_fuser" in lname:
                    t_lname = "t_final_rgb_tokens"
                elif "t_final_rgb_tokens" in lname and "t_input_rgb_tokens" in t_embeds:
                    t_lname = "t_input_rgb_tokens"
                else:
                    t_lname = lname
                s_lout = s_embeds[lname][input_cond_idx]
                t_lout = t_embeds[t_lname][input_cond_idx]
                s_feat = s_lout if is_tensor(s_lout) else s_lout["output"]
                t_feat = t_lout if is_tensor(t_lout) else t_lout["output"]
                if use_attn:
                    if isinstance(t_lout, dict):
                        attn = t_lout["cross_attn"].flatten(-2)
                    else:
                        attn = t_embeds[f"{lname}_attn"][input_cond_idx]
                    s_feat = s_feat * attn.squeeze().unsqueeze(-1)
                    t_feat = t_feat * attn.squeeze().unsqueeze(-1)
                loss = F.l1_loss(s_feat, t_feat)
                losses[input_cond_idx][lname].append(loss.item())
                total_loss += loss
                count += 1
    total_loss /= count
    return {"total_loss": total_loss, "losses": losses, "count": count}


def load_st_models(
    args, block_idxs, device="cpu", rank=0, is_train=True, use_only_encoder=True
):
    cur_plt_backend = plt.get_backend()
    sys.path.append(f"{RELATED_DIR}/rec/sam-3d-objects/notebook")

    from inference import Inference

    tag = "hf"
    config_base_path = f"{RELATED_DIR}/rec/sam-3d-objects/checkpoints/{tag}"
    if use_only_encoder and not args.use_sam3d:
        config_path = f"{config_base_path}/pipeline_encoder.yaml"
    else:
        config_path = f"{config_base_path}/pipeline.yaml"
    if args.use_sam3d:
        t_model = torch.nn.Identity()
    else:
        t_model = Inference(
            config_path, compile=False, device=device, use_ckpt=not args.do_debug
        )._pipeline
    ckpt_params = {}
    if args.resume_dir is not None:
        ckpt_params = dict(
            ss_generator_cond_embedder_ckpt_path=f"{args.resume_dir}/best_ss_generator_cond_embedder.pt",
            rgbe_fuser_ckpt_path=f"{args.resume_dir}/best_rgbe_fuser.pt",
        )
    s_model = Inference(
        config_path,
        compile=False,
        use_event=True,
        device=device,
        use_ckpt=not args.do_debug,
        rgbe_fusion_type=args.rgbe_fusion_type,
        use_only_sparse=args.use_sam3d,
        **ckpt_params,
    )._pipeline

    for pset in [t_model.parameters(), s_model.parameters()]:
        for p in pset:
            p.requires_grad = False
    t_model.eval()
    if not is_train:
        s_model.eval()

    # event module is the last in ss_generator.yaml
    event_module_idx = (
        len(s_model.condition_embedders.ss_condition_embedder.module_list) - 1
    )
    assert event_module_idx == 3, event_module_idx

    if args.use_sam3d:
        trainable_param_names = []
        for n, p in s_model.named_parameters():
            if is_train and (
                f"module_list.{event_module_idx}" in n
                and (
                    "patch_embed" in n
                    or (
                        any(f"blocks.{i}." in n for i in block_idxs)
                        and any(x in n for x in [".mlp"])
                    )
                )
                or "rgbe_fuser" in n
            ):
                p.requires_grad = True
                trainable_param_names.append(n)
    else:
        for p in [t_model, s_model]:
            p.condition_embedders.ss_condition_embedder.embedder_list = [
                x
                for i, x in enumerate(
                    p.condition_embedders.ss_condition_embedder.embedder_list
                )
                if i in [0, 3]
            ]
            p.condition_embedders.ss_condition_embedder.module_list = (
                torch.nn.ModuleList(
                    [
                        x
                        for i, x in enumerate(
                            p.condition_embedders.ss_condition_embedder.module_list
                        )
                        if i in [0, 3]
                    ]
                )
            )
            p.condition_embedders.ss_condition_embedder.projection_nets = (
                torch.nn.ModuleList(
                    [
                        x
                        for i, x in enumerate(
                            p.condition_embedders.ss_condition_embedder.projection_nets
                        )
                        if i in [0, 3]
                    ]
                )
            )
        trainable_param_names = []
        for n, p in s_model.named_parameters():
            if is_train and (
                f"module_list.{event_module_idx}" in n
                and (
                    "patch_embed" in n
                    or (
                        any(f"blocks.{i}." in n for i in block_idxs)
                        and any(x in n for x in [".mlp"])
                    )
                )
                or "rgbe_fuser" in n
            ):
                p.requires_grad = True
                trainable_param_names.append(n)
    if rank == 0:
        print(f"\n{trainable_param_names=}\n")
        print(
            f"s_model.parameters={sum(p.numel() for p in s_model.parameters() if p.requires_grad)}"
        )
        print(
            f"t_model.parameters={sum(p.numel() for p in t_model.parameters() if p.requires_grad)}"
        )

    # revert plt backend
    plt.switch_backend(cur_plt_backend)

    return s_model, t_model


def init_distributed():
    """
    Initializes torch.distributed using environment variables set by torchrun.
    Returns (rank, world_size, local_rank).
    """
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=local_rank,
        )
        dist.barrier()

    return rank, world_size, local_rank


def is_main_process(rank):
    return rank == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size, op=dist.ReduceOp.SUM):
    """
    Reduce a single scalar tensor over all processes and return the average.
    """
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor, op=op)
        tensor /= world_size
    return tensor


def reduce_dict(input_dict, average=True, device=None):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2 or is_empty(input_dict):
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            v = input_dict[k]
            if not is_tensor(v):
                v = torch.tensor(v, device=device)
            values.append(v)
        values = torch.stack(values, dim=0)
        if values.device != torch.device("cpu"):
            dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def fix_outdated_args(args):
    from event_sam3d.img2event.train import get_arg_parser

    parser = get_arg_parser()

    def noattr(x):
        return not hasattr(args, x)

    def is_none(x):
        return getattr(args, x, None) is None

    if hasattr(args, "t_mlp_num_layers"):
        args.rt_mlps_num_layers = args.t_mlp_num_layers

    # for all args present in parser but not in args, set them to their default values
    for group in parser._action_groups:
        for action in group._group_actions:
            arg_name = action.dest
            if noattr(arg_name):
                setattr(args, arg_name, action.default)

    return args


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.do_stop = False

    def __call__(self, metric=None, loss=None):

        assert (
            metric is not None or loss is not None
        ), "Either metric or loss should be provided"
        current = metric if metric is not None else -loss

        if self.best is None:
            self.best = current
        elif current < self.best + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: {current=:0.4f} and {self.best=:0.4f}. Patience: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.do_stop = True
        else:
            self.best = current
            self.counter = 0
