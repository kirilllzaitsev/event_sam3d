import argparse

import torch
import yaml

from event_sam3d.config import PROJ_DIR


def load_esam3d(ckpt_name, device="cuda", ckpt_type="best"):
    from event_sam3d.img2event.train import Trainer, build_model
    # best/latest
    ckpt_dir = f"{PROJ_DIR}/checkpoints/{ckpt_name}"
    cfg = argparse.Namespace(
        **yaml.load(open(f"{ckpt_dir}/config.yml"), Loader=yaml.UnsafeLoader)
    )
    build_model_res = build_model(cfg, rank=0)
    model, forward_args = (
        build_model_res["model"].to(device),
        build_model_res["forward_args"],
    )
    ckpt = torch.load(f"{ckpt_dir}/{ckpt_type}.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    trainer = Trainer(model, cfg)
    return {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
        "forward_args": forward_args,
    }


def get_condition_embedder(pipe, use_event=True, ss_condition_embedder=None):
    ss_condition_embedder = (
        pipe.condition_embedders["ss_condition_embedder"]
        if ss_condition_embedder is None
        else ss_condition_embedder
    )
    condition_embedder = [
        x
        for x in ss_condition_embedder.embedder_list
        if all(("event" if use_event else "image") in xx[0] for xx in x[1])
    ]
    assert len(condition_embedder) == 1 and len(condition_embedder[0]) == 2, len(
        condition_embedder
    )
    # same encoder for full/cropped imgs
    condition_embedder = condition_embedder[0][0]
    return condition_embedder
