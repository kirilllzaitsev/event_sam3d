import os
from pathlib import Path

import numpy as np

from event_sam3d.config import REPLICA_DIR
from event_sam3d.utils.common_utils import cast_to_torch
from event_sam3d.utils.data_utils import load_sam3_res
from event_sam3d.utils.events_representations import Tencode
from event_sam3d.utils.io import load_color
from event_sam3d.utils.misc_utils import get_ordered_paths, print_cls


class EventReplicaDataset:
    def __init__(
        self,
        seq_name,
        root=REPLICA_DIR,
        height=256,
        width=448,
        event_window_ms=50,
        transform=None,
        event_representation=None,
        nr_temporal_bins=5,
        use_masks=True,
        use_vg_event_repr=False,
        obj_name="barrel",
        len_limit=None,
        include_only_if_enough_events=False,
        min_num_events=500,
    ):
        self.seq_name = seq_name
        self.root = root
        self.event_representation = event_representation
        self.nr_temporal_bins = nr_temporal_bins
        self.obj_name = obj_name
        self.height = height
        self.width = width
        self.len_limit = len_limit
        self.transform = transform
        self.min_num_events = min_num_events

        self.use_masks = use_masks
        self.use_vg_event_repr = use_vg_event_repr
        self.include_only_if_enough_events = include_only_if_enough_events

        self.hw = (height, width)
        self.half_event_window_us = (event_window_ms // 2) * 1e3

        self.input_folder = Path(self.root) / f"{self.seq_name}"
        self.rgb_paths = get_ordered_paths(
            f"{self.input_folder}/original_images/*.png")
        self.event_paths = get_ordered_paths(
            f"{self.input_folder}/event/*.npz")
        self.img_timestamps = np.loadtxt(
            f"{self.input_folder}/timestamps.txt")[:, 1]
        if use_vg_event_repr:
            self.vg = Tencode(height=self.hw[0], width=self.hw[1])
        if use_masks:
            mask_paths = []
            for p in self.rgb_paths:
                mp = p.replace("original_images/", f"sam3/{obj_name}_").replace(".png", ".pt").replace(".jpg", ".pt")
                if os.path.exists(mp):
                    mask_paths.append(mp)
            mask_frame_names = set(Path(p).stem.replace(f"{obj_name}_", "") for p in mask_paths)
            target_idxs = [
                idx for idx, p in enumerate(self.rgb_paths) if Path(p).stem in mask_frame_names
            ]
            self.rgb_paths = [self.rgb_paths[i] for i in target_idxs]
            self.img_timestamps = self.img_timestamps[target_idxs]
        self.num_frames = len(self.rgb_paths)
        if len_limit is not None:
            self.num_frames = min(self.num_frames, len_limit)

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return print_cls(
            self,
            excluded_attrs=[
                "rgb_paths",
                "sharp_rgb_paths",
                "event_paths",
                "img_timestamps",
            ],
            extra_str=f"{len(self.rgb_paths)=} {self.rgb_paths[:5]=} {self.rgb_paths[-5:]=}",
        )

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        sharp_image_path = rgb_path.replace(
            'original_images', 'sharp_images')
        event_path = Path(
            rgb_path).parents[1] / "event" / rgb_path.split('/')[-1].replace('.png', '.npz')
        rgb = load_color(rgb_path)
        rgb_clean = load_color(sharp_image_path)
        events = np.load(event_path, allow_pickle=True)
        sample = {
            "rgb": rgb,
            "rgb_clean": rgb_clean,
            "events": events,
            "rgb_path": rgb_path,
            "frame_name": Path(rgb_path).stem,
        }
        if self.use_masks:
            sam3_res_path = f'{rgb_path.replace("original_images/", f"sam3/{self.obj_name}_").replace(".jpg", ".pt")}'
            mask = load_sam3_res(sam3_res_path)
            sample["mask"] = mask
        if self.use_vg_event_repr:
            event_repr = self.vg.convert(
                x=cast_to_torch(events['x']),
                y=cast_to_torch(events['y']),
                t=cast_to_torch(events['t']),
                p=cast_to_torch(events['p']),
            )
            event_repr = self.vg.to_rgb_mono(event_repr)
            sample["events"] = event_repr
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
