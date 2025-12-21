import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

from event_sam3d.config import MVSEC_DIR


class MVSEC(Dataset):
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
    ):
        """ """
        self.seq_name = seq_name
        self.root = root
        self.event_representation = event_representation
        self.nr_events_window = nr_events_window
        self.nr_temporal_bins = nr_temporal_bins
        self.mode = mode
        self.augmentation = augmentation

        if mode == "train":
            self.use_labels = False
        elif mode == "val":
            self.use_labels = True
        elif mode == "test":
            self.use_labels = True

        self.height = height
        self.width = width
        self.original_height = 260
        self.original_width = 346

        self.dataset = h5py.File(os.path.join(self.root, f"{self.seq_name}.hdf5"), "r")
        self.num_frames = self.dataset["davis/left"]["image_raw"].shape[0]
        self.num_events = self.dataset["davis/left"]["events"].shape[0]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        closest_event_id = self.dataset["davis/left"]["image_raw_event_inds"][idx]
        start_event_id = max(closest_event_id - self.nr_events_window // 2, 0)
        if closest_event_id + self.nr_events_window // 2 >= self.num_events:
            start_event_id = self.num_frames - self.nr_events_window

        events = self.dataset["davis/left"]["events"][
            start_event_id : (start_event_id + self.nr_events_window)
        ][()]
        gray = self.dataset["davis/left"]["image_raw"][idx]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return {
            "rgb": rgb,
            "events": events,
            "closest_event_id": closest_event_id,
            "start_event_id": start_event_id,
        }
        }
