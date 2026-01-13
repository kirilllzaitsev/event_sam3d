import os
from pathlib import Path

import numpy as np
import torch

from event_sam3d.utils.common_utils import cast_to_numpy


def load_sam3_res(sam3_res_path):
    if not sam3_res_path.endswith(".pt"):
        sam3_res_path = Path(sam3_res_path).with_suffix(".pt")
    assert os.path.exists(
        sam3_res_path
    ), f"sam3 res path does not exist: {sam3_res_path}"
    sam3_res = torch.load(sam3_res_path, map_location="cpu")
    masks = cast_to_numpy(sam3_res["masks"].squeeze(1))
    largest_mask_idx = np.argmax([(np.sum(m)) for m in masks])
    mask = masks[largest_mask_idx]
    return mask


def load_sam3d_res(sam3d_res_path):
    if not sam3d_res_path.endswith(".pt"):
        sam3d_res_path = Path(sam3d_res_path).with_suffix(".pt")
    assert os.path.exists(
        sam3d_res_path
    ), f"sam3d res path does not exist: {sam3d_res_path}"
    sam3d_res = torch.load(sam3d_res_path, map_location="cpu")
    res = {k: v.squeeze() for k, v in sam3d_res.items()}
    return res


def get_sam3_path_from_rgb(rgb_path, obj_name):
    frame_name = Path(rgb_path).stem
    return f'{Path(rgb_path).parents[1]/f"sam3/{obj_name}_{frame_name}.pt"}'


def get_sam3d_path_from_rgb(rgb_path, obj_name):
    frame_name = Path(rgb_path).stem
    return f'{Path(rgb_path).parents[1]/f"sam3d_sparse/{obj_name}_{frame_name}.pt"}'
