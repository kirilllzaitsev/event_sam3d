import os
from pathlib import Path
import numpy as np
import torch

from event_sam3d.utils.common_utils import cast_to_numpy


def load_sam3_res(sam3_res_path):
    if not sam3_res_path.endswith(".pt"):
        sam3_res_path = Path(sam3_res_path).with_suffix(".pt")
    assert os.path.exists(sam3_res_path), f"sam3 res path does not exist: {sam3_res_path}"
    sam3_res = torch.load(sam3_res_path, map_location="cpu")
    masks = cast_to_numpy(sam3_res["masks"].squeeze(1))
    largest_mask_idx = np.argmax([(np.sum(m)) for m in masks])
    mask = masks[largest_mask_idx]
    return mask
