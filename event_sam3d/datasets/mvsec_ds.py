import os
import re

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from event_sam3d.config import MVSEC_DIR
from event_sam3d.utils.common_utils import cast_to_numpy
from event_sam3d.utils.event_utils import VoxelGrid
from event_sam3d.utils.misc_utils import get_ordered_paths


class MVSECDataset(Dataset):
    def __init__(
        self,
        seq_name,
        root=MVSEC_DIR,
        height=260,
        width=346,
        nr_events_window=30_000,
        augmentation=False,
        mode="train",
        event_representation=None,
        nr_temporal_bins=5,
        use_masks=True,
        use_vg_event_repr=False,
        obj_name="barrel",
    ):
        """ """
        self.seq_name = seq_name
        self.root = root
        self.event_representation = event_representation
        self.nr_events_window = nr_events_window
        self.nr_temporal_bins = nr_temporal_bins
        self.mode = mode
        self.augmentation = augmentation
        self.obj_name = obj_name

        self.use_masks = use_masks
        self.use_vg_event_repr = use_vg_event_repr

        if mode == "train":
            self.use_labels = False
        elif mode == "val":
            self.use_labels = True
        elif mode == "test":
            self.use_labels = True

        self.height = height
        self.width = width
        self.hw = (height, width)
        self.original_height = 260
        self.original_width = 346

        self.dataset = h5py.File(os.path.join(self.root, f"{self.seq_name}.hdf5"), "r")
        self.num_frames = self.dataset["davis/left"]["image_raw"].shape[0]
        self.num_events = self.dataset["davis/left"]["events"].shape[0]
        self.frame_ids = list(range(self.num_frames))

        if use_masks:
            paths = get_ordered_paths(
                f"{self.root}/{self.seq_name}/sam3/{obj_name}*.pt"
            )
            self.frame_ids = sorted(
                [
                    int(re.search("\d+", x).group())
                    for x in (set([x.split("_")[-1] for x in paths]))
                ]
            )
        if use_vg_event_repr:
            self.vg = VoxelGrid(3, self.hw[0], self.hw[1])

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        closest_event_id = self.dataset["davis/left"]["image_raw_event_inds"][frame_id]
        start_event_id = max(closest_event_id - self.nr_events_window // 2, 0)
        if closest_event_id + self.nr_events_window // 2 >= self.num_events:
            start_event_id = self.num_frames - self.nr_events_window

        events = self.dataset["davis/left"]["events"][
            start_event_id : (start_event_id + self.nr_events_window)
        ][()]
        gray = self.dataset["davis/left"]["image_raw"][frame_id]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        sample = {
            "rgb": rgb,
            "events": events,
            "closest_event_id": closest_event_id,
            "start_event_id": start_event_id,
        }
        if self.use_masks:
            sam3_res = torch.load(
                f"{self.root}/{self.seq_name}/sam3/{self.obj_name}_{frame_id:06d}.pt"
            )
            masks = cast_to_numpy(sam3_res["masks"].squeeze(1))
            sample["mask"] = masks[0]
        if self.use_vg_event_repr:
            event_repr = self.vg.convert(
                event_dict={
                    "x": events[:, 0],
                    "y": events[:, 1],
                    "t": events[:, 2],
                    "p": events[:, 3],
                }
            )
            sample["events"] = event_repr
        return sample
