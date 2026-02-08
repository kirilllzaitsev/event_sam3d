import os
import sys
from pathlib import Path

import numpy as np

from event_sam3d.config import CO3D_DIR, RELATED_DIR, REPLICA_DIR
from event_sam3d.utils.common_utils import cast_to_torch
from event_sam3d.utils.data_utils import (
    get_sam3_path_from_rgb,
    get_sam3d_path_from_rgb,
    load_sam3_res,
    load_sam3d_res,
)
from event_sam3d.utils.events_representations import Tencode
from event_sam3d.utils.io import load_color, load_mask
from event_sam3d.utils.misc_utils import get_ordered_paths, print_cls


class CO3DDataset:
    def __init__(
        self,
        seq_name,
        root=CO3D_DIR,
        height=260,
        width=346,
        event_window_ms=50,
        transform=None,
        event_representation=None,
        nr_temporal_bins=5,
        use_masks=True,
        use_sam3_masks=True,
        use_sam3d=False,
        use_vg_event_repr=False,
        len_limit=None,
        include_only_if_enough_events=False,
        min_num_events=500,
        use_blurry_rgb=False,
        input_frame_rate=30,
        **kwargs,
    ):
        sys.path.insert(0, f"{RELATED_DIR}/data/v2e")
        from v2ecore.data_utils.aedat2_reader import AEDat2Reader

        self.seq_name = seq_name
        self.root = root
        self.event_representation = event_representation
        self.nr_temporal_bins = nr_temporal_bins
        self.height = height
        self.width = width
        self.len_limit = len_limit
        self.transform = transform
        self.min_num_events = min_num_events

        self.use_masks = use_masks
        self.use_sam3_masks = use_sam3_masks
        self.use_sam3d = use_sam3d
        self.use_vg_event_repr = use_vg_event_repr
        self.include_only_if_enough_events = include_only_if_enough_events
        self.use_blurry_rgb = use_blurry_rgb

        self.hw = (height, width)
        self.half_event_window_us = (event_window_ms // 2) * 1e3
        self.obj_name = seq_name.split("/")[0]

        self.input_folder = Path(self.root) / f"{self.seq_name}"
        self.num_frames_skipped = 1
        self.rgb_paths = get_ordered_paths(f"{self.input_folder}/images/*.jpg")[
            self.num_frames_skipped :
        ]
        if use_vg_event_repr:
            self.vg = Tencode(height=self.hw[0], width=self.hw[1])
        if use_sam3d:
            self.filter_by_obj_name(use_masks=False)
        self.num_frames = len(self.rgb_paths)
        if len_limit is not None:
            self.num_frames = min(self.num_frames, len_limit)

        self.img_timestamps = (
            np.arange(
                self.num_frames_skipped,
                self.num_frames + self.num_frames_skipped,
                dtype=np.float64,
            )
            / input_frame_rate
        )

        # -- event reader (cache timestamps once) -----------------------------
        self.reader = AEDat2Reader(
            f"{self.input_folder}/events.aedat", auto_detect_size=False
        )
        self.reader.set_sensor_size(width=width, height=height)
        self.reader.load_timestamps_us()  # pre-cache for O(log N) lookups

    def filter_by_obj_name(self, use_masks=True):
        mask_paths = []
        subdir = "masks" if use_masks else "sam3d_sparse"
        for p in self.rgb_paths:
            mp = p.replace("images/", f"{subdir}/").replace(
                ".jpg", ".png" if use_masks else ".pt"
            )
            if os.path.exists(mp):
                mask_paths.append(mp)
        mask_frame_names = set(Path(p).stem for p in mask_paths)
        target_idxs = [
            idx
            for idx, p in enumerate(self.rgb_paths)
            if Path(p).stem in mask_frame_names
        ]
        self.rgb_paths = [self.rgb_paths[i] for i in target_idxs]
        # self.img_timestamps = self.img_timestamps[target_idxs]

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return print_cls(
            self,
            excluded_attrs=[
                "rgb_paths",
                "event_paths",
                "img_timestamps",
            ],
            extra_str=f"{len(self.rgb_paths)=} {self.rgb_paths[:5]=} {self.rgb_paths[-5:]=}",
        )

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        # sharp_image_path = rgb_path.replace("original_images", "sharp_images")
        event_path = (
            Path(rgb_path).parents[1]
            / "event"
            / rgb_path.split("/")[-1].replace(".png", ".npz")
        )
        rgb = load_color(rgb_path)
        # rgb_clean = load_color(sharp_image_path)
        rgb_clean = rgb.copy()
        ts = self.img_timestamps[index]
        events = self.reader.get_event_window_fast(
            ts, self.half_event_window_us / 1e6, "xytp"
        )
        sample = {
            "rgb": rgb if self.use_blurry_rgb else rgb_clean,
            "rgb_clean": rgb_clean,
            "events": events,
            "rgb_path": rgb_path,
            "frame_name": Path(rgb_path).stem,
            "ts": ts,
        }
        if self.use_masks:
            use_orig_mask = not self.use_sam3_masks
            if self.use_sam3_masks:
                sam3_res_path = rgb_path.replace("images/", "sam3/").replace(
                    ".jpg", ".pt"
                )
                if os.path.exists(sam3_res_path):
                    mask = load_sam3_res(sam3_res_path)
                else:
                    use_orig_mask = True
            if use_orig_mask:
                mask_path = rgb_path.replace("images/", "masks/").replace(
                    ".jpg", ".png"
                )
                mask = load_mask(mask_path)
            sample["mask"] = mask.astype(np.uint8)
        if self.use_sam3d:
            sam3d_res_path = get_sam3d_path_from_rgb(rgb_path, self.obj_name)
            sam3d_res = load_sam3d_res(sam3d_res_path)
            sample["t"] = sam3d_res
        if self.use_vg_event_repr:
            event_repr = self.vg.convert(
                x=cast_to_torch(events[:, 0]),
                y=cast_to_torch(events[:, 1]),
                t=cast_to_torch(events[:, 2]),
                p=cast_to_torch(events[:, 3]),
            )
            event_repr = self.vg.to_rgb_mono(event_repr)
            sample["events"] = event_repr
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
