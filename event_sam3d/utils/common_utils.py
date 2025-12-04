import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def get_transpose_func(x, dim0=-1, dim1=-2):
    if istensor(x):
        t_func = functools.partial(torch.transpose, dim0=dim0, dim1=dim1)
    else:
        t_func = functools.partial(np.swapaxes, axis1=dim0, axis2=dim1)
    return t_func


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def istensor(x):
    return isinstance(x, torch.Tensor)


def pick_library(x):
    if isinstance(x, torch.Tensor):
        lib = torch
    else:
        lib = np
    return lib


def create_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def adjust_img_for_plt(img):
    img = cast_to_numpy(img)
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            raise RuntimeError(f"Expected 1 image, got {img.shape[0]}")
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if np.max(img) <= 1:
        img = img * 255
    img = img.astype(np.uint8)
    return img


def from_numpy(x):
    if isinstance(x, list):
        return torch.stack([from_numpy(xx) for xx in x])
    elif isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x).float()


def adjust_img_for_torch(rgb):
    if isinstance(rgb, np.ndarray):
        rgb = from_numpy(rgb)
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(2, 0, 1)
    rgb = rgb.float()
    if rgb.max() > 1:
        rgb /= 255.0
    return rgb


def adjust_depth_for_plt(img):
    img = cast_to_numpy(img)
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            raise RuntimeError(f"Expected 1 image, got {img.shape[0]}")
    if img.shape[0] == 1:
        img = img.transpose(1, 2, 0)
    return img


def cast_to_numpy(x, dtype=None) -> np.ndarray:
    if x is None or isinstance(x, str):
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array([cast_to_numpy(xx) for xx in x])
    elif isinstance(x, dict):
        return {k: cast_to_numpy(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        if dtype is not None:
            x = x.astype(dtype)
        return x
    elif isinstance(x, (int, float, complex, np.float32)):
        return x
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def cast_to_torch(x, device=None, include_top_list=False):
    if x is None or isinstance(x, str):
        return x
    elif type(x) in [list, tuple]:
        res = [cast_to_torch(xx, device=device) for xx in x]
        if include_top_list and not isinstance(res[0], str) and res[0] is not None:
            if isinstance(res[0], list):
                return torch.tensor(res)
            return torch.stack(res)
        return res
    elif isinstance(x, dict):
        return {k: cast_to_torch(v, device=device) for k, v in x.items()}
    elif isinstance(x, np.ndarray) or isinstance(
        x, (int, float, complex, bool, np.bool_)
    ):
        return torch.tensor(x, device=device)
    return x.to(device)


def detach_and_cpu(x):
    if type(x) in [list, tuple]:
        return [detach_and_cpu(xx) for xx in x]
    elif isinstance(x, dict):
        return {k: detach_and_cpu(v) for k, v in x.items()}
    elif (
        isinstance(x, np.ndarray)
        or isinstance(x, (int, float, complex))
        or np.isscalar(x)
    ):
        return x
    elif x is None or not hasattr(x, "detach"):
        return x
    return x.detach().cpu()


def convert_arr_to_tensor(v):
    if isinstance(v[0], np.ndarray):
        v = [torch.from_numpy(x) for x in v]
    v_tensor = torch.stack(v)
    return v_tensor


def hw_from_rgb(rgb0):
    return rgb0.shape[:2] if rgb0.shape[-1] == 3 else rgb0.shape[1:]


def interpolate_img(img, size):
    assert isinstance(size, list) or isinstance(size, tuple)
    ori_size = img.shape[-2:]

    # resize before inference
    if size[0] != ori_size[0] or size[1] != ori_size[1]:
        assert img.ndim == 4, img.ndim
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=True)
    return img, ori_size
