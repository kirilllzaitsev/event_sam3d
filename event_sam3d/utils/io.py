import json
import os
import re
import sys

import cv2
import imageio.v3 as iio
import numpy as np
import png
import yaml

from .common_utils import cast_to_numpy


def load_json(path):
    with open(path, "r") as f:
        info = yaml.load(f, Loader=yaml.CLoader)
    return info


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def cast_formats_for_json(data):
    # casting for every keys in dict to list so that it can be saved as json
    for key in data.keys():
        if (
            isinstance(data[key][0], np.ndarray)
            or isinstance(data[key][0], np.float32)
            or isinstance(data[key][0], np.float64)
            or isinstance(data[key][0], np.int32)
            or isinstance(data[key][0], np.int64)
        ):
            data[key] = np.array(data[key]).tolist()
    return data


def load_depth_(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)


def load_mask_(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_rgb_(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def ensure_path_exists(path):
    assert os.path.exists(path), f"File not found: {path}"


def load_depth(path, wh=None, zfar=np.inf, do_convert_to_m=True):
    depth = load_depth_(path)
    if do_convert_to_m:
        depth = depth.astype(np.float32) / 1e3
    if wh is not None:
        depth = resize_img(depth, wh=wh)
    depth[(depth < 0.001) | (depth >= zfar)] = 0
    return depth


def resize_img(depth, wh):
    return cv2.resize(depth, (wh[0], wh[1]), interpolation=cv2.INTER_LINEAR)


def load_color(path, wh=None, use_grayscale=False):
    color = load_rgb_(path)
    if use_grayscale:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    if wh is not None:
        color = resize_img(color, wh=wh)
    return color


def load_mask(path, wh=None):
    mask = load_mask_(path)
    if len(mask.shape) == 3:
        for c in range(3):
            if mask[..., c].sum() > 0:
                mask = mask[..., c]
                break
    if wh is not None:
        mask = resize_img(mask, wh=wh)
    mask = mask.astype(bool)
    return mask


def load_semantic_mask(path, wh=None, excluded_colors=None):
    mask = load_mask_(path)[..., ::-1]

    excluded_colors = [] if excluded_colors is None else excluded_colors
    for color in excluded_colors:
        mask[(mask == color).all(axis=-1)] = 0
    if wh is not None:
        mask = resize_img(mask, wh=wh)
    return mask


def convert_semantic_mask_to_bin(mask, included_colors):
    joint_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for color in included_colors:
        joint_mask[(mask == color).all(axis=-1)] = 1
    return joint_mask.astype(bool)


def save_depth(path, im, is_m=True):
    if is_m:
        im_mm = im * 1e3
    else:
        im_mm = im
    save_depth_16bit(path, im_mm)


def save_depth_16bit(path, im):
    """Saves a depth image (16-bit) to a PNG file.

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if path.split(".")[-1].lower() != "png":
        raise ValueError("Only PNG format is currently supported.")

    im = cast_to_numpy(im)
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def readpfm(file):
    # https://github.com/XiandaGuo/OpenStereo/blob/v2/stereo/datasets/dataset_utils/readpfm.py
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == "3":
        header = header.decode("utf-8")
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    if (sys.version[0]) == "3":
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    else:
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    if (sys.version[0]) == "3":
        scale = float(file.readline().rstrip().decode("utf-8"))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def load_disparity(path):
    disp, scale = readpfm(path)
    disp = disp.astype(np.float32)
    disp[disp == np.inf] = 0
    return disp, scale


def parse_calib_file(path):
    sample = {}
    with open(path, "r") as f:
        lines = f.readlines()
        k0_line = lines[0].strip()
        k1_line = lines[1].strip()
        p = re.compile(r"\[([0-9. ]+); ([0-9. ]+); ([0-9. ]+)\]")
        k0 = np.array(
            [np.fromstring(x, sep=" ") for x in p.match(k0_line.split("=")[1]).groups()]
        ).reshape(3, 3)
        k1 = np.array(
            [np.fromstring(x, sep=" ") for x in p.match(k1_line.split("=")[1]).groups()]
        ).reshape(3, 3)
        sample["k0"] = k0
        sample["k1"] = k1
        sample["doffs"] = float(lines[2].split("=")[1].strip())
        sample["baseline"] = float(lines[3].split("=")[1].strip())

    return sample


def load_video(path):
    return list(iio.imiter(path))
