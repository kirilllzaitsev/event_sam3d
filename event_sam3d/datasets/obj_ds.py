import os
import sys
from pathlib import Path

import numpy as np

from event_sam3d.config import OBJ_DIR, RELATED_DIR
from event_sam3d.nb_utils_static import wrap_with_futures
from event_sam3d.utils.common_utils import adjust_img_for_plt, cast_to_torch
from event_sam3d.utils.data_utils import (
    get_sam3d_path_from_rgb,
    load_sam3_res,
    load_sam3d_res,
)
from event_sam3d.utils.events_representations import Tencode
from event_sam3d.utils.io import load_color, load_mask
from event_sam3d.utils.misc_utils import get_ordered_paths, print_cls


class ObjDataset:
    def __init__(
        self,
        seq_name,
        root=OBJ_DIR,
        height=260,
        width=346,
        event_window_ms=50,
        transform=None,
        event_representation=None,
        nr_temporal_bins=5,
        use_masks=True,
        use_sam3_masks=False,
        use_event_masks=True,
        use_sam3d=False,
        use_vg_event_repr=False,
        len_limit=None,
        include_only_if_enough_events=False,
        min_num_events=500,
        use_blurry_rgb=False,
        input_frame_rate=30,
        blur_severity=None,
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
        self.blur_severity = blur_severity

        self.use_masks = use_masks
        self.use_sam3_masks = use_sam3_masks
        self.use_event_masks = use_event_masks
        self.use_sam3d = use_sam3d
        self.use_vg_event_repr = use_vg_event_repr
        self.include_only_if_enough_events = include_only_if_enough_events
        self.use_blurry_rgb = use_blurry_rgb

        self.hw = (height, width)
        self.half_event_window_us = (event_window_ms // 2) * 1e3
        self.obj_name = seq_name.split("/")[0]

        self.input_folder = Path(self.root) / f"{self.seq_name}"
        self.img_dirname = "images"
        self.num_frames_skipped = 1
        self.rgb_paths = get_ordered_paths(
            f"{self.input_folder}/{self.img_dirname}/*.png"
        )[self.num_frames_skipped :]
        if use_vg_event_repr:
            self.vg = Tencode(height=self.hw[0], width=self.hw[1])
            self.reader = AEDat2Reader(
                f"{self.input_folder}/events.aedat", auto_detect_size=False
            )
            self.reader.set_sensor_size(width=width, height=height)
            self.reader.load_timestamps_us()
        self.num_frames = len(self.rgb_paths)
        self.img_timestamps = (
            np.arange(
                self.num_frames_skipped,
                self.num_frames + self.num_frames_skipped,
                dtype=np.float64,
            )
            / input_frame_rate
        )
        if use_masks:
            self.filter_by_obj_name(use_masks=True)
        if use_sam3d:
            self.filter_by_obj_name(use_masks=False)

        if blur_severity is not None:
            # take only imgs whose frame_name is in images_blur_{blur_severity}/
            blur_frame_names = [
                Path(p).stem
                for p in get_ordered_paths(
                    f"{self.input_folder}/images_blur_{blur_severity}/*.png"
                )
            ]
            target_idxs = [
                idx
                for idx, p in enumerate(self.rgb_paths)
                if Path(p).stem in blur_frame_names
            ]
            self.rgb_paths = [self.rgb_paths[i] for i in target_idxs]
            self.img_timestamps = self.img_timestamps[target_idxs]
        self.num_frames = len(self.rgb_paths)
        if len_limit is not None:
            self.num_frames = min(self.num_frames, len_limit)

    def filter_by_obj_name(self, use_masks=True):
        mask_paths = []
        subdir = "masks" if use_masks else "sam3d_sparse"
        for p in self.rgb_paths:
            mp = p.replace(f"{self.img_dirname}/", f"{subdir}/").replace(
                ".png", ".png" if use_masks else ".pt"
            )
            if os.path.exists(mp) and use_masks:
                mask_paths.append(mp)
        valid_paths = []

        def filter_invalid_mask(mp):
            m = load_mask(mp)
            return is_mask_valid(m)

        valid = wrap_with_futures(mask_paths, filter_invalid_mask, disable_tqdm=True)

        if self.use_event_masks:

            def filter_invalid_mask(index):
                ts = self.img_timestamps[index]
                events = self.reader.get_event_window_fast(
                    ts, self.half_event_window_us / 1e6, "xytp"
                )
                return len(events) > 400

            valid_e = wrap_with_futures(
                list(range(len(mask_paths))), filter_invalid_mask, disable_tqdm=True
            )
            valid = [vm and ve for vm, ve in zip(valid, valid_e)]

        valid_paths = [mp for mp, valid in zip(mask_paths, valid) if valid]

        mask_frame_names = set(Path(p).stem for p in valid_paths)
        target_idxs = [
            idx
            for idx, p in enumerate(self.rgb_paths)
            if Path(p).stem in mask_frame_names
        ]
        self.rgb_paths = [self.rgb_paths[i] for i in target_idxs]
        self.img_timestamps = self.img_timestamps[target_idxs]

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
        sample = {
            "rgb": rgb if self.use_blurry_rgb else rgb_clean,
            "rgb_clean": rgb_clean,
            "rgb_path": rgb_path,
            "frame_name": Path(rgb_path).stem,
            "ts": ts,
        }
        if self.use_masks:
            use_orig_mask = not self.use_sam3_masks
            if self.use_sam3_masks:
                sam3_res_path = rgb_path.replace(
                    f"{self.img_dirname}/", "sam3/"
                ).replace(".png", ".pt")
                if os.path.exists(sam3_res_path):
                    mask = load_sam3_res(sam3_res_path)
                else:
                    use_orig_mask = True
            if use_orig_mask:
                mask_path = rgb_path.replace(f"{self.img_dirname}/", "masks/").replace(
                    ".png", ".png"
                )
                mask = load_mask(mask_path)
            sample["mask"] = mask.astype(np.uint8)
        if self.use_sam3d:
            sam3d_res_path = get_sam3d_path_from_rgb(rgb_path, self.obj_name)
            sam3d_res = load_sam3d_res(sam3d_res_path)
            sample["t"] = sam3d_res
        if self.use_vg_event_repr:
            events = self.reader.get_event_window_fast(
                ts, self.half_event_window_us / 1e6, "xytp"
            )
            event_repr = self.vg.convert(
                x=cast_to_torch(events[:, 0]),
                y=cast_to_torch(events[:, 1]),
                t=cast_to_torch(events[:, 2]),
                p=cast_to_torch(events[:, 3]),
            )
            event_repr=adjust_img_for_plt(event_repr)
            # event_repr = self.vg.to_rgb_mono(event_repr)
            sample["events"] = event_repr
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def is_mask_valid(m):
    h, w = m.shape
    n_px = m.sum()
    is_mask_valid = not (n_px < h * w * 0.05 or n_px > h * w * 0.9)
    return is_mask_valid
