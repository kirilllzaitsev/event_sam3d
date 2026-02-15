import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from event_sam3d.config import RELATED_DIR

from .common_utils import cast_to_torch
from .kpt_utils import get_queries, load_kpt_det_and_match


def get_events_from_samples(sample_events):
    xs = []
    ys = []
    ps = []
    ts = []
    for i, event in enumerate(sample_events):
        xs.append(event["x"])
        ys.append(event["y"])
        ps.append(event["p"])
        ts.append(event["t"])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    ts = np.concatenate(ts)
    return {"x": xs, "y": ys, "p": ps, "t": ts}


def edict_to_arr(events_dict):
    x, y, p, t = events_dict["x"], events_dict["y"], events_dict["p"], events_dict["t"]
    events = np.stack([y, x, t, p], axis=1)
    return events


class EventTracker:
    def __init__(self, hw=(256, 448), use_kpts=True, device="cuda"):

        self.use_kpts = use_kpts
        self.device = device

        prj_path = f"{RELATED_DIR}/kpts/ETAP"
        sys.path.append(prj_path)
        from src.model.etap.model import Etap
        from src.representations import MixedDensityEventStack
        from src.utils import Visualizer

        self.ckpt_path = Path(f"{RELATED_DIR}/kpts/ETAP/weights/ETAP_v1_cvpr25.pth")
        num_bins = 10
        self.hw = hw

        # Object to convert raw event data into grid representations
        self.converter = MixedDensityEventStack(
            image_shape=(self.hw),
            num_stacks=num_bins,
            interpolation="bilinear",
            channel_overlap=True,
            centered_channels=False,
        )

        # Load the model
        self.tracker = Etap(num_in_channels=num_bins, stride=4, window_len=8)
        weights = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        self.tracker.load_state_dict(weights)
        self.tracker = self.tracker.to(device)
        self.tracker.eval()
        self.num_events = 60_000
        self.visib_th = 0.8

        if use_kpts:
            features = "superpoint"
            extractor, matcher = load_kpt_det_and_match(
                features, filter_threshold=0.8, max_num_keypoints=256
            )
            self.extractor, self.matcher = extractor.cuda(), matcher.cuda()
        else:
            self.extractor = None

    def track_two_event_samples(self, events_prev_prev, events_prev_cur, image_prev):
        events = get_events_from_samples([events_prev_prev, events_prev_cur])
        ts = events["t"]
        t_start = ts[0] + 1e-4  # seconds
        t_end = ts[-1] - 1e-4  # seconds
        t_mid = (t_start + t_end) / 2
        num_slices = 24
        # important to provide sufficiently many events to the tracker, even if these events come from the previous batch
        tracking_timestamps = np.linspace(t_mid, t_end, num_slices)
        xy = np.vstack((events["x"], events["y"])).T
        p = events["p"]
        t = ts

        assert (
            t_start > t[0]
        ), "Start time must be greater than the first event timestamp"
        assert t_end < t[-1], "End time must be less than the last event timestamp"
        assert t_start < t_end, "Start time must be less than end time"
        assert (
            xy.shape[0] == p.shape[0] == t.shape[0]
        ), "Event data arrays must have the same length"

        event_indices = np.searchsorted(t, tracking_timestamps)
        event_representations = []

        # At each tracking timestep, we take the last num_events events and convert
        # them into a grid representation.
        for i_end in tqdm(
            event_indices, desc="Creating grid representations", disable=True
        ):
            i_start = max(i_end - self.num_events, 0)

            events = np.stack(
                [
                    xy[i_start:i_end, 1],
                    xy[i_start:i_end, 0],
                    t[i_start:i_end],
                    p[i_start:i_end],
                ],
                axis=1,
            )
            ev_repr = self.converter(events)
            event_representations.append(ev_repr)

        voxels = np.stack(event_representations, axis=0)
        voxels = torch.from_numpy(voxels)[None].float().to(self.device)

        # query
        queries = get_queries(
            image_prev,
            use_kpts=self.use_kpts,
            hw=(self.hw),
            extractor=self.extractor,
        )

        # run tracker
        with torch.no_grad():
            result = self.tracker(voxels, queries, iters=6)
            predictions, visibility = (
                result["coords_predicted"],
                result["vis_predicted"],
            )

        visibility = visibility > self.visib_th

        idx0, idx1 = 0, num_slices - 1
        visib_mask = visibility[0, idx0] & visibility[0, idx1]
        mkpts0 = predictions[0, idx0][visib_mask]
        mkpts1 = predictions[0, idx1][visib_mask]
        return {"mkpts0": mkpts0, "mkpts1": mkpts1, "visib_mask": visib_mask}


class VoxelGrid:
    # bflow/data/utils/representations.py

    def __init__(self, channels: int, height: int, width: int):
        assert channels > 1
        assert height > 1
        assert width > 1
        self.nb_channels = channels
        self.height = height
        self.width = width

    def get_extended_time_window(self, t0_center: int, t1_center: int):
        dt = self._get_dt(t0_center, t1_center)
        t_start = math.floor(t0_center - dt)
        t_end = math.ceil(t1_center + dt)
        return t_start, t_end

    def _construct_empty_voxel_grid(self):
        return torch.zeros(
            (self.nb_channels, self.height, self.width),
            dtype=torch.float,
            requires_grad=False,
            device=torch.device("cpu"),
        )

    def _get_dt(self, t0_center: int, t1_center: int):
        assert t1_center > t0_center
        return (t1_center - t0_center) / (self.nb_channels - 1)

    def _normalize_time(self, time: torch.Tensor, t0_center: int, t1_center: int):
        # time_norm < t0_center will be negative
        # time_norm == t0_center is 0
        # time_norm > t0_center is positive
        # time_norm == t1_center is (nb_channels - 1)
        # time_norm > t1_center is greater than (nb_channels - 1)
        return (time - t0_center) / (t1_center - t0_center) * (self.nb_channels - 1)

    @staticmethod
    def _is_int_tensor(tensor: torch.Tensor) -> bool:
        return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)

    def convert(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        pol: torch.Tensor = None,
        time: torch.Tensor = None,
        event_dict=None,
        t0_center: Optional[int] = None,
        t1_center: Optional[int] = None,
    ):

        if event_dict is not None:
            time = event_dict["t"]
            x = event_dict["x"].astype("int16")
            y = event_dict["y"].astype("int16")
            pol = event_dict["p"].astype("int8")

        if pol.min() == -1:
            pol = pol.clip(0)

        # assert x.device == y.device == pol.device == time.device == torch.device("cpu")
        assert type(t0_center) == type(t1_center)
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        x = cast_to_torch(x)
        y = cast_to_torch(y)
        pol = cast_to_torch(pol)
        time = cast_to_torch(time)
        # assert self._is_int_tensor(time)

        is_int_xy = self._is_int_tensor(x)
        if is_int_xy:
            assert self._is_int_tensor(y)

        voxel_grid = self._construct_empty_voxel_grid()
        ch, ht, wd = self.nb_channels, self.height, self.width
        # assert pol.min() == 0, pol.min()
        with torch.no_grad():
            t0_center = t0_center if t0_center is not None else time[0]
            t1_center = t1_center if t1_center is not None else time[-1]
            t_norm = self._normalize_time(time, t0_center, t1_center)

            t0 = t_norm.floor().int()
            value = 2 * pol.float() - 1

            if is_int_xy:
                for tlim in [t0, t0 + 1]:
                    mask = (tlim >= 0) & (tlim < ch)
                    interp_weights = value * (1 - (tlim - t_norm).abs()).float()

                    index = ht * wd * tlim.long() + wd * y.long() + x.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)
            else:
                x0 = x.floor().int()
                y0 = y.floor().int()
                for xlim in [x0, x0 + 1]:
                    for ylim in [y0, y0 + 1]:
                        for tlim in [t0, t0 + 1]:

                            mask = (
                                (xlim < wd)
                                & (xlim >= 0)
                                & (ylim < ht)
                                & (ylim >= 0)
                                & (tlim >= 0)
                                & (tlim < ch)
                            )
                            interp_weights = (
                                value
                                * (1 - (xlim - x).abs())
                                * (1 - (ylim - y).abs())
                                * (1 - (tlim - t_norm).abs())
                            )

                            index = (
                                ht * wd * tlim.long() + wd * ylim.long() + xlim.long()
                            )

                            voxel_grid.put_(
                                index[mask], interp_weights[mask], accumulate=True
                            )

        return voxel_grid


def event2event_image(events, num_img, H, W, device="cuda"):
    zero_img = torch.zeros(num_img - 1, H, W, device=device)
    t, x, y, p = events["t"], events["x"], events["y"], events["p"]
    t = torch.from_numpy(t)
    x = torch.from_numpy(x).to(device).long()
    y = torch.from_numpy(y).to(device).long()
    p = torch.from_numpy(p).to(device).float()

    num_events = t.shape[0]
    indices = torch.arange(num_events, device=device)
    split_indices = torch.tensor_split(indices, num_img - 1)
    chunk_labels = torch.cat(
        [
            torch.full((len(sub_indices),), i)
            for i, sub_indices in enumerate(split_indices)
        ]
    )
    chunk_labels = chunk_labels.long()

    zero_img.index_put_((chunk_labels, y, x), p, accumulate=True)
    tstamp_index = [0] + [inx[-1].item() for inx in split_indices]
    tstamp = t[tstamp_index]
    norm_tstamp = (tstamp - tstamp[0]) / (tstamp[-1] - tstamp[0])

    return zero_img, norm_tstamp.to(torch.float32).to(device)


# rgb is [3, H, W]
def EDI_rgb(blurry_image, events, H, W, threshold=0.3):
    assert blurry_image.shape[0] == 3, blurry_image.shape
    device = blurry_image.device
    bin_num = events.shape[0]
    blurry_image = blurry_image.permute(1, 2, 0) * 255.0
    event_sum = torch.zeros(H, W, device=device)
    EDI = torch.ones(H, W, device=device)
    for i in range(bin_num):
        event_sum = event_sum + events[i]
        EDI = EDI + torch.exp(threshold * event_sum)
    EDI = torch.stack([EDI, EDI, EDI], axis=-1)
    img = (bin_num + 1) * blurry_image / EDI
    img = img.permute(2, 0, 1) / 255.0

    return img


def deblur_with_edi(original_image, events, device='cuda'):
    hw = original_image.shape[-2:]
    event_image, norm_ts = event2event_image(events, num_img=8, H=hw[0], W=hw[1], device=device)
    gt_image = EDI_rgb(
        original_image,
        event_image,
        hw[0],
        hw[1],
    )
    return gt_image.clip(0, 1)
