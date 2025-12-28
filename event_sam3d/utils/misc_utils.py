import argparse
import gc
import glob
import os
import random
import re

import numpy as np
import torch
import yaml

from .common_utils import istensor


def print_cls(cls, exclude_private=True, excluded_attrs=None, extra_str=None):
    msg = "\n" + "-" * 30 + "\n"
    msg += f"self: {type(cls)}\n"
    excluded_attrs = excluded_attrs or []
    attrs = cls.__dict__.items()
    attrs = sorted(attrs, key=lambda x: x[0])
    for k, v in attrs:
        if exclude_private and k.startswith("_"):
            continue
        if k in excluded_attrs:
            continue
        msg += f"{k}: {v}\n"
    if extra_str:
        msg += f"Extras:\n{extra_str}"
    if len(excluded_attrs) > 0:
        msg += f"\nAlso contains: {excluded_attrs}"
    msg += "\n" + "-" * 30
    return msg


def free_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()


def is_empty(v):
    # returns true if no values are in the dict/list. recursive.
    if isinstance(v, dict):
        return all(is_empty(x) for x in v.values())
    elif isinstance(v, list) and len(v) > 0:
        return all(is_empty(x) for x in v)
    elif istensor(v):
        return v.ndim > 0 and len(v) == 0
    elif hasattr(v, "__len__"):
        return len(v) == 0
    elif v is None:
        return True
    else:
        return False


def print_args(args, logger=None):
    from tabulate import tabulate

    msg = tabulate(sorted(vars(args).items()), tablefmt="grid")
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_ordered_paths(pattern, sort_fn=None, exts=None):
    sort_fn = sort_fn or (
        lambda x: [int(xx) for xx in re.findall(r"(\d+)", x.rsplit("/")[-1])]
        # lambda x: int(re.search(r".*?(\d+)(?!.*\d)", x).group(1))
    )  # search for last numerical value
    pattern = str(pattern)
    if "*" not in pattern:
        assert os.path.isdir(pattern), f"Check {pattern=}"
        pattern = f"{pattern}/*"
    paths = glob.glob(pattern)
    if exts is not None:
        paths = [p for p in paths if any(p.endswith(ext) for ext in exts)]
    return sorted(paths, key=sort_fn)


def load_args(path):
    return argparse.Namespace(**yaml.load(open(path), Loader=yaml.UnsafeLoader))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
