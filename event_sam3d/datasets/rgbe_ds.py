import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from event_sam3d.config import RGBE_DIR
from event_sam3d.utils.common_utils import cast_to_numpy, cast_to_torch
from event_sam3d.utils.data_utils import (
    get_sam3d_path_from_rgb,
    load_sam3_res,
    load_sam3d_res,
)
from event_sam3d.utils.events_representations import Tencode
from event_sam3d.utils.io import load_color
from event_sam3d.utils.misc_utils import get_ordered_paths, print_cls


class RGBEDataset(Dataset):
    def __init__(
        self,
        split,
        root=RGBE_DIR,
        height=260,
        width=346,
        event_window_ms=50,
        transform=None,
        use_masks=True,
        use_sam3d=False,
        use_vg_event_repr=False,
        obj_name="person",
        test_subsplit=None,
        len_limit=None,
        include_only_if_enough_events=False,
        min_num_events=500,
        dirnames=None,
    ):
        self.use_masks = use_masks
        self.use_sam3d = use_sam3d
        self.use_vg_event_repr = use_vg_event_repr
        self.include_only_if_enough_events = include_only_if_enough_events
        self.height = height
        self.width = width
        self.event_window_ms = event_window_ms
        self.transform = transform
        self.obj_name = obj_name
        self.len_limit = len_limit
        self.min_num_events = min_num_events
        self.test_subsplit = test_subsplit
        self.split = split

        self.root = f"{root}/{split}"
        self.hw = (height, width)
        self.image_pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.image_pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        self.evimg_pixel_mean = self.image_pixel_mean
        self.evimg_pixel_std = self.image_pixel_std

        if split == "train":
            rgb_paths = [
                line.rstrip()
                for line in open(os.path.join(self.root, "eventsam_split.txt"))
            ]
            if dirnames is not None:
                rgb_paths = [p for p in rgb_paths if p.split("/")[0] in dirnames]
            self.rgb_paths = [f"{self.root}/{p}" for p in rgb_paths]
            self.data_dirs = list(set([p.split("/")[0] for p in rgb_paths]))
        else:
            assert test_subsplit is not None
            self.data_dirs = [
                f"{self.root}/{line.rstrip()}"
                for line in open(os.path.join(self.root, f"{test_subsplit}.txt"))
            ]
            if dirnames is not None:
                self.data_dirs = [
                    d for d in self.data_dirs if d.split("/")[-1] in dirnames
                ]
            self.rgb_paths = []
            for dirpath in self.data_dirs:
                rgb_paths = get_ordered_paths(Path(dirpath) / "rgb_image/*")
                self.rgb_paths.extend(rgb_paths)

        if use_masks:
            mask_paths = [
                p.replace("rgb_image/", f"sam3/{obj_name}_").replace(".jpg", ".pt")
                for p in self.rgb_paths
            ]
            self.rgb_paths = [
                p for p, m in zip(self.rgb_paths, mask_paths) if os.path.exists(m)
            ]
        self.num_frames = len(self.rgb_paths) if len_limit is None else len_limit
        if use_vg_event_repr:
            self.vg = Tencode(height=self.hw[0], width=self.hw[1])

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return print_cls(
            self,
            excluded_attrs=[
                "rgb_paths",
                "data_dirs",
                "image_pixel_mean",
                "image_pixel_std",
                "evimg_pixel_mean",
                "evimg_pixel_std",
            ],
            extra_str=f"{len(self.rgb_paths)=} {self.rgb_paths[:5]=} {self.rgb_paths[-5:]=}\n{len(self.data_dirs)=} {self.data_dirs[:5]=} {self.data_dirs[-5:]=}",
        )

    def __getitem__(self, index):
        image_path = self.rgb_paths[index]
        evimg_path = image_path.replace("rgb_image/", "voxel_image/")
        image = load_color(image_path)
        evimg = load_color(evimg_path)
        sample = {
            "rgb": image,
            "events": evimg,
            "rgb_path": image_path,
            "frame_name": Path(image_path).stem,
        }
        if self.use_masks:
            sam3_res_path = f'{image_path.replace("rgb_image/", f"sam3/{self.obj_name}_").replace(".jpg", ".pt")}'
            mask = load_sam3_res(sam3_res_path)
            sample["mask"] = mask
        if self.use_sam3d:
            sam3d_res_path = get_sam3d_path_from_rgb(image_path, self.obj_name)
            sam3d_res = load_sam3d_res(sam3d_res_path)
            sample["t"] = sam3d_res

        if self.use_vg_event_repr:
            events_path = Path(
                image_path.replace("rgb_image/", "event/")
                .replace(".png", ".npz")
                .replace(".jpg", ".npz")
            )
            events = np.load(events_path, allow_pickle=True)
            event_repr = self.vg.convert(
                x=cast_to_torch(events["x"]),
                y=cast_to_torch(events["y"]),
                t=cast_to_torch(events["t"]),
                p=cast_to_torch(events["p"]),
            )
            event_repr = self.vg.to_rgb_mono(event_repr)
            sample["events"] = event_repr
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
