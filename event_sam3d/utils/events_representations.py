# based on https://github.com/bartn8/depthanyevent/tree/main/dataset/events

import json
from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F

from .events_visualizations import (
    histogram_mono_to_rgb,
    histogram_stereo_to_rgb,
    tencode_mono_to_rgb,
    tencode_stereo_to_rgb,
    voxel_grid_mono_to_rgb,
    voxel_grid_stereo_to_rgb,
)


class EventRepresentation(ABC):

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    @abstractmethod
    def convert(
        self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor
    ):
        pass

    @abstractmethod
    def get_dataset_file_name(self):
        pass

    @abstractmethod
    def to_rgb_stereo(self, representation_left, representation_right):
        pass

    @abstractmethod
    def to_rgb_mono(self, representation):
        pass

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def convert_from_voxels(self, voxels):
        pass


class VoxelGrid(EventRepresentation):
    def __init__(
        self, height: int, width: int, normalize: bool = True, channels: int = 3
    ):
        super().__init__(height, width)
        self.voxel_grid = torch.zeros(
            (channels, height, width), dtype=torch.float, requires_grad=False
        )
        self.channels = channels
        self.normalize = normalize
        self.height = height
        self.width = width

    @classmethod
    def from_configuration(cls, configuration):
        assert configuration["representation_type"] == "voxel_grid"
        return cls(
            channels=int(configuration["channels"]),
            height=int(configuration["height"]),
            width=int(configuration["width"]),
            normalize=bool(configuration["normalize"]),
        )

    def convert(
        self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, t: torch.Tensor
    ):
        assert x.shape == y.shape == p.shape == t.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(p.device)
            voxel_grid = self.voxel_grid.clone()

            if x.shape[0] == 0:
                return voxel_grid

            t_norm = t
            t_norm = (
                (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])
            )  # This normalizes t between 0 and C-1

            x0 = x.int()  # Let's make the x an integer
            y0 = y.int()  # Let's make the y an integer
            t0 = t_norm.int()  # Let's make the normalized time an integer

            value = 2 * p - 1  # Let's make pol from in [0; 1] to in [-1; 1]

            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    for tlim in [t0, t0 + 1]:

                        mask = (
                            (xlim < W)
                            & (xlim >= 0)
                            & (ylim < H)
                            & (ylim >= 0)
                            & (tlim >= 0)
                            & (tlim < self.channels)
                        )
                        interp_weights = (
                            value
                            * (1 - (xlim - x).abs())
                            * (1 - (ylim - y).abs())
                            * (1 - (tlim - t_norm).abs())
                        )
                        interp_weights = interp_weights.float()

                        index = H * W * tlim.long() + W * ylim.long() + xlim.long()

                        voxel_grid.put_(
                            index[mask], interp_weights[mask], accumulate=True
                        )

            if self.normalize:
                voxel_grid = self.normalize_fn(voxel_grid)

        return voxel_grid

    def normalize_fn_enrico(self, voxel_grid):
        mask = torch.nonzero(voxel_grid, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = voxel_grid[mask].mean()
            std = voxel_grid[mask].std()
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

    def normalize_fn(self, voxel_grid):
        nonzero_ev = voxel_grid != 0
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = (
                torch.sum(voxel_grid, dtype=torch.float32) / num_nonzeros
            )  # force torch.float32 to prevent overflows when using 16-bit precision
            stddev = torch.sqrt(
                torch.sum(voxel_grid**2, dtype=torch.float32) / num_nonzeros - mean**2
            )
            mask = nonzero_ev.type_as(voxel_grid)
            voxel_grid = mask * (voxel_grid - mean) / stddev

        return voxel_grid

    def get_dataset_file_name(self, extra_str=""):
        return f"voxel_grid_{extra_str}_{self.channels}"

    def to_rgb_stereo(self, representation_left, representation_right):
        return voxel_grid_stereo_to_rgb(representation_left, representation_right)

    def to_rgb_mono(self, representation):
        return voxel_grid_mono_to_rgb(representation)

    def info(self):
        output = {
            "representation_type": "voxel_grid",
            "channels": self.channels,
            "normalize": self.normalize,
            "height": self.height,
            "width": self.width,
        }
        return output

    def convert_from_voxels(self, voxel_grid):
        _, h, w = voxel_grid.shape

        # Compress/Expand original voxelgrid channels into self.channels... Use F.interpolate 5D
        voxel_grid = F.interpolate(
            voxel_grid.unsqueeze(0).unsqueeze(0),
            size=(self.channels, h, w),
            mode="nearest",
        )[0, 0]

        if self.normalize:
            voxel_grid = self.normalize_fn(voxel_grid)
        return voxel_grid


class Histogram(EventRepresentation):

    def __init__(self, height: int, width: int, remove_int_artifact: bool = True):
        super().__init__(height, width)
        self.height = height
        self.width = width
        self.remove_int_artifact = remove_int_artifact

    @classmethod
    def from_configuration(cls, configuration):
        assert configuration["representation_type"] == "histogram"
        return cls(
            height=int(configuration["height"]),
            width=int(configuration["width"]),
            remove_int_artifact=bool(configuration["remove_int_artifact"]),
        )

    def convert(
        self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor
    ):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        mask = (x < self.width) & (x >= 0) & (y < self.height) & (y >= 0)

        x = x[mask]
        y = y[mask]
        pol = pol[mask]

        histo = torch.zeros(
            (2, self.height, self.width), dtype=torch.float, requires_grad=False
        )

        if x.shape[0] == 0:
            return histo

        with torch.no_grad():
            pol = pol.int()  # Let's make the polarity an integer

            x0 = x.int()  # Let's make the x an integer
            y0 = y.int()  # Let's make the y an integer

            if self.remove_int_artifact:
                for x_lim in [x0, x0 + 1]:
                    for y_lim in [y0, y0 + 1]:
                        interp_weights = (
                            (1 - (x_lim - x).abs()) + (1 - (y_lim - y).abs())
                        ) / 4
                        index = (
                            pol * self.height * self.width
                            + self.width * y_lim.long()
                            + x_lim.long()
                        )
                        mask = index < self.height * self.width * 2
                        histo.put_(index[mask], interp_weights[mask], accumulate=True)
            else:
                index = (
                    pol * self.height * self.width + self.width * y0.long() + x0.long()
                )
                histo.put_(
                    index, torch.ones_like(index, dtype=histo.dtype), accumulate=True
                )
            # CLIP:
            histo[histo > 200] = 200
        return histo

    def get_dataset_file_name(self, extra_str=""):
        return f"histogram_{extra_str}"

    def to_rgb_stereo(self, histo_left, histo_right):
        return histogram_stereo_to_rgb(histo_left, histo_right)

    def to_rgb_mono(self, histo):
        return histogram_mono_to_rgb(histo)

    def info(self):
        output = {
            "representation_type": "histogram",
            "remove_int_artifact": self.remove_int_artifact,
            "height": self.height,
            "width": self.width,
        }
        return output

    def convert_from_voxels(self, voxels):
        n, h, w = voxels.shape
        histogram = torch.zeros((2, h, w), dtype=torch.float)
        for i in range(n):
            histogram[0, voxels[i] > 0] += voxels[i][voxels[i] > 0]
            histogram[1, voxels[i] < 0] += -voxels[i][voxels[i] < 0]
        return histogram


class Tencode(EventRepresentation):

    def __init__(
        self, height: int, width: int, normalize: bool = True, white_frame: bool = False
    ):
        super().__init__(height, width)
        self.height = height
        self.width = width
        self.normalize = normalize
        self.white_frame = white_frame

    @classmethod
    def from_configuration(cls, configuration):
        assert configuration["representation_type"] == "tencode"
        _white_frame = bool(configuration.get("white_frame", False))
        print(f"TENCODE White frame: {_white_frame}")
        return cls(
            height=int(configuration["height"]),
            width=int(configuration["width"]),
            normalize=bool(configuration["normalize"]),
            white_frame=_white_frame,
        )

    def convert(
        self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, t: torch.Tensor
    ):
        assert x.shape == y.shape == p.shape == t.shape
        assert x.ndim == 1

        if self.white_frame:
            tencode = torch.full(
                (3, self.height, self.width),
                255.0,
                dtype=torch.float,
                requires_grad=False,
            )
        else:
            tencode = torch.zeros(
                (3, self.height, self.width), dtype=torch.float, requires_grad=False
            )

        if x.shape[0] == 0:
            return tencode

        with torch.no_grad():
            p = p.int().clip(0)  # Let's make the polarity an integer {0,1}

            t_norm = t
            t_norm = (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

            index_red = (
                (0 * self.width * self.height) + (y.long() * self.width) + x.long()
            )
            index_green = (
                (1 * self.width * self.height) + (y.long() * self.width) + x.long()
            )
            index_blue = (
                (2 * self.width * self.height) + (y.long() * self.width) + x.long()
            )

            mask_red = (
                (x < self.width)
                & (x >= 0)
                & (y < self.height)
                & (y >= 0)
                & (index_red >= 0)
                & (index_red < 3 * self.height * self.width)
            )
            mask_green = (
                (x < self.width)
                & (x >= 0)
                & (y < self.height)
                & (y >= 0)
                & (index_green >= 0)
                & (index_green < 3 * self.height * self.width)
            )
            mask_blue = (
                (x < self.width)
                & (x >= 0)
                & (y < self.height)
                & (y >= 0)
                & (index_blue >= 0)
                & (index_blue < 3 * self.height * self.width)
            )

            tencode.put_(index_red[mask_red], 255.0 * p[mask_red], accumulate=False)
            tencode.put_(
                index_green[mask_green],
                (255.0 * (1 - t_norm[mask_green])).float(),
                accumulate=False,
            )
            tencode.put_(
                index_blue[mask_blue], 255.0 * (1 - p[mask_blue]), accumulate=False
            )

            if self.normalize:
                tencode = tencode / 255.0

        return tencode

    def get_dataset_file_name(self, extra_str=""):
        return f"tencode_{extra_str}"

    def to_rgb_stereo(self, representation_left, representation_right):
        return tencode_stereo_to_rgb(representation_left, representation_right)

    def to_rgb_mono(self, representation):
        return tencode_mono_to_rgb(representation)

    def info(self):
        output = {
            "representation_type": "tencode",
            "height": self.height,
            "width": self.width,
        }
        return output

    def convert_from_voxels(self, voxels):
        n, h, w = voxels.shape
        g_map = [255.0 * (1 - i) for i in torch.linspace(0, 1, n)]

        if self.white_frame:
            tencode = torch.full((3, h, w), 255.0, dtype=torch.float)
        else:
            tencode = torch.zeros((3, h, w), dtype=torch.float)

        for i in range(n):
            tencode[0, voxels[i] > 0] = 255.0
            tencode[1, voxels[i] > 0] = g_map[i]
            tencode[2, voxels[i] > 0] = 0.0
            tencode[0, voxels[i] < 0] = 0.0
            tencode[1, voxels[i] < 0] = g_map[i]
            tencode[2, voxels[i] < 0] = 255.0

        # If there are no events, we set the frame to white
        if self.white_frame:
            tencode[:, tencode.sum(dim=0) == 0] = 255.0

        if self.normalize:
            tencode = tencode / 255.0

        return tencode


def representation_from_config_file(info_json_path: str):
    # Let's read the dataset info.json file
    with open(info_json_path, "r") as info_json:
        representation_info = json.load(info_json)
    if representation_info["representation_type"] == "voxel_grid":
        return VoxelGrid.from_configuration(representation_info)
    elif representation_info["representation_type"] == "histogram":
        return Histogram.from_configuration(representation_info)
    elif representation_info["representation_type"] == "tencode":
        return Tencode.from_configuration(representation_info)
    else:
        raise ValueError(
            f"Unknown representation type {representation_info['representation_type']}"
        )
