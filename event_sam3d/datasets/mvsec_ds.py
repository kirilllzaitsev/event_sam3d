import os
import re

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from event_sam3d.config import MVSEC_DIR
from event_sam3d.datasets.transforms import Transform
from event_sam3d.utils.common_utils import cast_to_numpy, cast_to_torch
from event_sam3d.utils.event_utils import VoxelGrid
from event_sam3d.utils.events_representations import Tencode
from event_sam3d.utils.misc_utils import get_ordered_paths, print_cls


class MVSECDataset(Dataset):
    def __init__(
        self,
        seq_name,
        root=MVSEC_DIR,
        height=260,
        width=346,
        event_window_ms=50,
        transform_names=None,
        mode="train",
        event_representation=None,
        nr_temporal_bins=5,
        use_masks=True,
        use_vg_event_repr=False,
        obj_name="barrel",
        len_limit=None,
    ):
        """ """
        self.seq_name = seq_name
        self.root = root
        self.event_representation = event_representation
        self.nr_temporal_bins = nr_temporal_bins
        self.mode = mode
        self.obj_name = obj_name
        self.height = height
        self.width = width
        self.len_limit = len_limit

        self.use_masks = use_masks
        self.use_vg_event_repr = use_vg_event_repr

        self.hw = (height, width)
        self.half_event_window_us = (event_window_ms // 2) * 1e3

        if transform_names is None:
            self.transform = None
        else:
            assert use_vg_event_repr
            self.transform = Transform(names=transform_names)

        self.hdf5_path = os.path.join(self.root, f"{self.seq_name}.hdf5")
        self.dataset = h5py.File(self.hdf5_path, "r")
        self.num_frames = self.dataset["davis/left"]["image_raw"].shape[0]
        self.num_events = self.dataset["davis/left"]["events"].shape[0]
        self.frame_ts = (
            np.asarray(self.dataset["davis/left/image_raw_ts"]) * 1e6
        ).astype("int64")
        self.frame_ids = list(range(self.num_frames))
        self.event_ts = (self.dataset["davis/left/events"][:, 2] * 1e6).astype("int64")

        if use_masks:
            paths = get_ordered_paths(
                f"{self.root}/{self.seq_name}/sam3/{obj_name}*.pt"
            )
            target_frame_ids = {
                int(re.search("\d+", x).group())
                for x in (set([x.split("_")[-1] for x in paths]))
            }
            matched_frame_idxs = [
                i for i, fid in enumerate(self.frame_ids) if fid in target_frame_ids
            ]
            self.frame_ids = [self.frame_ids[i] for i in matched_frame_idxs]
            self.frame_ts = self.frame_ts[matched_frame_idxs]
            self.num_frames = len(self.frame_ids)
        if use_vg_event_repr:
            self.vg = Tencode(height=self.hw[0], width=self.hw[1])
        if len_limit is not None:
            self.num_frames = len_limit

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return print_cls(
            self,
            excluded_attrs=["dataset", "frame_ids"],
            extra_str=f"{len(self.frame_ids)=} {self.frame_ids[:5]=} {self.frame_ids[-5:]=}",
        )

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        closest_event_id = self.dataset["davis/left"]["image_raw_event_inds"][frame_id]
        closest_event_ts = self.event_ts[closest_event_id]
        start_event_id = np.searchsorted(
            self.event_ts, closest_event_ts - self.half_event_window_us, side="left"
        )
        end_event_id = np.searchsorted(
            self.event_ts,
            closest_event_ts + self.half_event_window_us,
            side="right",
        )
        events = self.dataset["davis/left"]["events"][start_event_id:end_event_id]

        gray = self.dataset["davis/left"]["image_raw"][frame_id]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        sample = {
            "rgb": rgb,
            "events": events,
            "closest_event_id": closest_event_id,
            "start_event_id": start_event_id,
            "end_event_id": end_event_id,
        }
        if self.use_masks:
            sam3_res = torch.load(
                f"{self.root}/{self.seq_name}/sam3/{self.obj_name}_{frame_id:06d}.pt",
                map_location="cpu",
            )
            masks = cast_to_numpy(sam3_res["masks"].squeeze(1))
            largest_mask_idx = np.argmax([(np.sum(m)) for m in masks])
            sample["mask"] = masks[largest_mask_idx]
        if self.use_vg_event_repr:
            event_repr = self.vg.convert(
                x=cast_to_torch(events[:, 0]),
                y=cast_to_torch(events[:, 1]),
                t=cast_to_torch(events[:, 2]),
                p=cast_to_torch(events[:, 3]),
            )
            event_repr = self.vg.to_rgb_mono(event_repr)
            # sample["events_raw"] = events
            sample["events"] = event_repr

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
