import os
import sys
from pathlib import Path

import cv2
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

        self.use_blurry_rgb = use_blurry_rgb or blur_severity is not None
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
            assert self.use_blurry_rgb
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

        valid = [True for _ in range(len(self.rgb_paths))]

        if self.use_event_masks:

            def filter_invalid_mask(index):
                ts = self.img_timestamps[index]
                events = self.reader.get_event_window_fast(
                    ts, self.half_event_window_us / 1e6, "xytp"
                )
                return len(events) > self.min_num_events

            valid_e = wrap_with_futures(
                list(range(len(valid))), filter_invalid_mask, disable_tqdm=True
            )
            valid = [vm and ve for vm, ve in zip(valid, valid_e)]

        valid_paths = [mp for mp, valid in zip(self.rgb_paths, valid) if valid]

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
        if self.blur_severity is None:
            rgb = load_color(rgb_path)
            rgb_clean = rgb.copy()
        else:
            rgb = load_color(
                rgb_path.replace(
                    f"{self.img_dirname}/", f"images_blur_{self.blur_severity}/"
                )
            )
            rgb_clean = load_color(rgb_path)

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
                rgb[np.all(rgb == (26, 26, 26), axis=-1)] = 0
                mask = np.any(rgb > 0, axis=-1)
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
                )
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
            event_repr = adjust_img_for_plt(event_repr)
            sample["events"] = event_repr
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ObjBlurSharpEventDataset:
    """Dataset yielding (blurry, sharp, events) triplets from OBJ sequences.

    blurry  – pixel-wise mean of `blur_window` consecutive frames
    sharp   – center frame of that window
    events  – voxel-grid encoded events in a `event_window_ms` window
               centered on the sharp frame's timestamp

    Windows start at 0, blur_step, 2*blur_step, … (default blur_step = blur_window // 2).
    Windows whose center frame has fewer than `min_num_events` are dropped.
    """

    def __init__(
        self,
        seq_name,
        root=OBJ_DIR,
        height=260,
        width=346,
        event_window_ms=100,
        blur_window=20,
        blur_step=None,
        nr_temporal_bins=5,
        transform=None,
        input_frame_rate=30,
        min_num_events=500,
        len_limit=None,
        **kwargs,
    ):
        sys.path.insert(0, f"{RELATED_DIR}/data/v2e")
        from v2ecore.data_utils.aedat2_reader import AEDat2Reader

        self.seq_name = seq_name
        self.root = root
        self.height = height
        self.width = width
        self.hw = (height, width)
        self.nr_temporal_bins = nr_temporal_bins
        self.transform = transform
        self.blur_window = blur_window
        self.half_blur = blur_window // 2
        self.half_event_window_us = (event_window_ms / 2) * 1e3
        self.min_num_events = min_num_events

        self.input_folder = Path(self.root) / seq_name
        self.img_dirname = "images"
        num_frames_skipped = 1
        all_rgb_paths = get_ordered_paths(
            f"{self.input_folder}/{self.img_dirname}/*.png"
        )[num_frames_skipped:]

        num_all = len(all_rgb_paths)
        self.all_rgb_paths = all_rgb_paths
        self.all_timestamps = (
            np.arange(num_frames_skipped, num_all + num_frames_skipped, dtype=np.float64)
            / input_frame_rate
        )

        self.vg = Tencode(height=self.hw[0], width=self.hw[1])
        self.reader = AEDat2Reader(
            f"{self.input_folder}/events.aedat", auto_detect_size=False
        )
        self.reader.set_sensor_size(width=width, height=height)
        self.reader.load_timestamps_us()

        step = blur_step if blur_step is not None else blur_window // 2
        candidate_starts = range(0, max(0, num_all - blur_window + 1), step)
        self.valid_starts = [
            s
            for s in candidate_starts
            if len(
                self.reader.get_event_window_fast(
                    self.all_timestamps[s + self.half_blur],
                    self.half_event_window_us / 1e6,
                    "xytp",
                )
            )
            >= min_num_events
        ]
        if len_limit is not None:
            self.valid_starts = self.valid_starts[:len_limit]
        self.num_frames = len(self.valid_starts)

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return print_cls(
            self,
            excluded_attrs=["all_rgb_paths", "all_timestamps", "valid_starts"],
            extra_str=f"{self.num_frames=}",
        )

    def __getitem__(self, index):
        start = self.valid_starts[index]
        frame_paths = self.all_rgb_paths[start : start + self.blur_window]
        sharp_path = frame_paths[self.half_blur]
        sharp_ts = self.all_timestamps[start + self.half_blur]

        frames = np.stack([load_color(p) for p in frame_paths], axis=0)
        blurry = frames.mean(axis=0).astype(np.uint8)
        sharp = frames[self.half_blur]

        events = self.reader.get_event_window_fast(
            sharp_ts, self.half_event_window_us / 1e6, "xytp"
        )
        event_repr = self.vg.convert(
            x=cast_to_torch(events[:, 0]),
            y=cast_to_torch(events[:, 1]),
            t=cast_to_torch(events[:, 2]),
            p=cast_to_torch(events[:, 3]),
        )
        event_repr = adjust_img_for_plt(event_repr)

        sample = {
            "blurry": blurry,
            "sharp": sharp,
            "events": event_repr,
            "sharp_path": sharp_path,
            "frame_name": Path(sharp_path).stem,
            "ts": sharp_ts,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def is_mask_valid(m):
    h, w = m.shape
    n_px = m.sum()
    is_mask_valid = not (n_px < h * w * 0.01 or n_px > h * w * 0.9)
    return is_mask_valid
