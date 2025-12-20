import os

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

        self.frame_file_list = []
        self.dataset = h5py.File(os.path.join(self.root, f"{self.seq_name}.hdf5"), "r")
        self.extract_data()

    def __len__(self):
        return len(self.frame_file_list)

    def extract_data(self):
        num_frames = self.dataset["davis/left"]["image_raw"].shape[0]
        num_events = self.dataset["davis/left"]["events"].shape[0]

        gray_ts = np.array(
            self.dataset["davis"]["left"]["image_raw_ts"], dtype=np.float64
        )
        for i_file, img_ts in enumerate(gray_ts):
            closest_event_id = self.dataset["davis/left"]["image_raw_event_inds"][
                int(i_file)
            ]
            start_event_id = max(closest_event_id - self.nr_events_window // 2, 0)
            if closest_event_id + self.nr_events_window // 2 >= num_events:
                start_event_id = num_frames - self.nr_events_window
            frame_list = [int(i_file), start_event_id]
            self.frame_file_list.append(frame_list)

    def __getitem__(self, idx):
        frame_id, start_event_id = self.frame_file_list[idx]

        events = self.dataset["davis/left"]["events"][
            start_event_id : (start_event_id + self.nr_events_window)
        ][()]
        return {
            "events": events,
            "frame_id": frame_id,
            "start_event_id": start_event_id,
        }
