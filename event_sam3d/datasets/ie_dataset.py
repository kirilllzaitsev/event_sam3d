import numpy as np
import torch

from event_sam3d.utils.common_utils import cast_to_torch
from event_sam3d.utils.events_representations import Tencode
from event_sam3d.utils.misc_utils import print_cls


class IEDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):

        self.datasets = datasets

        self.dataset_lengths = [len(v) for k, v in datasets.items()]
        self.cum_lengths = np.cumsum(self.dataset_lengths)

        self.first_ds = list(datasets.values())[0]
        self.hw = (self.first_ds.height, self.first_ds.width)
        assert all((d.height, d.width) == self.hw for d in datasets.values()), [
            (k, d.height, d.width) for k, d in datasets.items()
        ]
        self.vg = Tencode(height=self.hw[0], width=self.hw[1])

    def __len__(self):
        return sum(self.dataset_lengths)

    def __repr__(self):
        return print_cls(
            self, excluded_attrs=["datasets"], extra_str=f"{self.datasets.keys()=}"
        )

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_lengths[dataset_idx - 1]

        dataset_key = list(self.datasets.keys())[dataset_idx]
        sample = self.datasets[dataset_key][sample_idx]

        if isinstance(sample["events"], dict):
            event_repr = self.vg.convert(
                x=cast_to_torch(sample["events"][:, 0]),
                y=cast_to_torch(sample["events"][:, 1]),
                t=cast_to_torch(sample["events"][:, 2]),
                p=cast_to_torch(sample["events"][:, 3]),
            )
            sample["events"] = event_repr
        sample["mask"] = (sample["mask"] * 255).astype(np.uint8)

        return sample
