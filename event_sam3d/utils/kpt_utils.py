import time
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .common_utils import adjust_img_for_plt, adjust_img_for_torch, rbd

try:
    from lightglue import ALIKED, DISK, SIFT, DoGHardNet, LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
except ImportError:
    print("lightglue not installed, some funcs not available")


try:
    from cotracker.predictor import CoTrackerOnlinePredictor
    from cotracker.utils.visualizer import Visualizer
except ImportError:
    print("cotracker not installed, some funcs not available")


def load_kpt_det_and_match(features, filter_threshold=0.1, max_num_keypoints=1024):
    # filter_threshold=keep matches with confidence > thresh
    extractor = load_extractor(features, max_num_keypoints=max_num_keypoints)

    matcher = LightGlue(
        features=features, filter_threshold=filter_threshold
    )  # load the matcher

    for p in extractor.parameters():
        p.requires_grad = False
    for p in matcher.parameters():
        p.requires_grad = False
    extractor.eval()
    matcher.eval()

    return extractor, matcher


def load_extractor(features, max_num_keypoints=1024):
    if features == "superpoint":
        extractor = SuperPoint(
            max_num_keypoints=max_num_keypoints
        ).eval()  # load the extractor
        for n, p in extractor.named_parameters():
            if "convPa" in n or "convPb" in n:
                p.requires_grad = False
    elif features == "disk":
        # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
        extractor = DISK(
            max_num_keypoints=max_num_keypoints
        ).eval()  # load the extractor
    elif features == "sift":
        extractor = SIFT(
            max_num_keypoints=max_num_keypoints
        ).eval()  # load the extractor
    elif features == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval()
    elif features == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval()
    else:
        raise ValueError(features)
    return extractor


def extract_kpts(x, extractor, do_normalize=False, use_zeros_for_pad=True):
    bs, c, h, w = x.shape
    memory_key_padding_mask = None
    if bs > 1:
        extracted_kpts = [extractor.extract(x[i : i + 1]) for i in range(bs)]
        extracted_kpts = [{k: v[0] for k, v in kpts.items()} for kpts in extracted_kpts]
        # pad with zeros up to the max number of keypoints
        max_kpts = max([len(kpts["keypoints"]) for kpts in extracted_kpts])
        memory_key_padding_mask = torch.zeros(
            bs, max_kpts, dtype=torch.bool, device=x.device
        )
        # extracted_kpts = copy.deepcopy(extracted_kpts)
        for bidx, kpts in enumerate(extracted_kpts):
            for k in ["keypoints", "descriptors"]:
                pad_len = max_kpts - len(kpts[k])
                if pad_len > 0:
                    if use_zeros_for_pad:
                        memory_key_padding_mask[bidx, -pad_len:] = True
                        kpts[k] = F.pad(kpts[k], (0, 0, 0, pad_len), value=0)
                    else:
                        # duplicate random pts
                        pad_idxs = torch.randint(0, len(kpts[k]), (pad_len,))
                        kpts[k] = torch.cat([kpts[k], kpts[k][pad_idxs]])
        extracted_kpts = {
            k: torch.stack([v[k] for v in extracted_kpts], dim=0)
            for k in ["keypoints", "descriptors", "score_map"]
        }
    else:
        extracted_kpts = extractor.extract(x)

    if do_normalize:
        extracted_kpts["keypoints"] = extracted_kpts["keypoints"] / torch.tensor(
            [w, h], dtype=extracted_kpts["keypoints"].dtype
        ).to(extracted_kpts["keypoints"].device)

    return {**extracted_kpts, "padding_mask": memory_key_padding_mask}


def get_matches(image0_rgb, image1_rgb, extractor, matcher):

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = image0_rgb.cuda()
    image1 = image1_rgb.cuda()

    times = []
    for _ in range(1):
        start = time.time()
        # extract local features
        feats0 = extractor.extract(
            image0
        )  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)

        # match the features
        matches01 = matcher({"image0": feats0, "image1": feats1})
        times.append(time.time() - start)

    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension
    matches = matches01["matches"]  # indices with shape (K,2)
    mkpts0 = feats0["keypoints"][
        matches[..., 0]
    ].cpu()  # coordinates in image #0, shape (K,2)
    mkpts1 = feats1["keypoints"][
        matches[..., 1]
    ].cpu()  # coordinates in image #1, shape (K,2)

    return {
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "scores": matches01["scores"],
        "times": times,
    }


class Cotracker:
    def __init__(
        self,
        checkpoint=None,
        device="cuda",
        use_online=True,
        model=None,
    ):
        from events_3dgs.config import RELATED_DIR
        checkpoint = (
            f"{RELATED_DIR}/kpts/co-tracker/checkpoints/scaled_online.pth"
            if checkpoint is None
            else checkpoint
        )
        self.model = (
            get_cotracker(checkpoint, device, use_online) if model is None else model
        )
        self.step = self.model.step

    def process_step(
        self,
        window_frames,
        is_first_step,
        queries=None,
        grid_size=None,
        grid_query_frame=None,
    ):
        video_chunk = torch.tensor(
            np.stack(window_frames[-self.model.step * 2 :]), device="cuda"
        ).float()[
            None
        ]  # (1, T, 3, H, W)
        if video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 1, 4, 2, 3)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            # add_support_grid=True,
        )

    def process_video(
        self, frames, queries=None, grid_size=None, grid_query_frame=None
    ):
        assert queries is not None or (
            grid_size is not None and grid_query_frame is not None
        )
        assert frames[0].max() > 1.0, frames[0].max()  # gets normalized by the model
        is_first_step = True
        window_frames = []
        for i, frame in enumerate(tqdm(frames)):
            if i % self.model.step == 0 and i != 0:
                pred_tracks, pred_visibility = self.process_step(
                    window_frames,
                    is_first_step,
                    queries=queries,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                )
                is_first_step = False
            window_frames.append(frame)
        # Processing the final video frames in case video length is not a multiple of self.model.step
        # TODO: check if queries are handled correctly here
        pred_tracks, pred_visibility = self.process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1 :],
            is_first_step,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
        return {
            "pred_tracks": pred_tracks,
            "pred_visibility": pred_visibility,
            "window_frames": window_frames,
        }

    def vis(
        self,
        window_frames=None,
        pred_tracks=None,
        pred_visibility=None,
        res=None,
        save_dir="./saved_videos",
        filename=None,
        n=int(1e9),
        query_frame=0,
    ):
        if res is not None:
            pred_tracks, pred_visibility, window_frames = (
                res["pred_tracks"],
                res["pred_visibility"],
                res["window_frames"],
            )
        else:
            assert all(
                x is not None for x in [window_frames, pred_tracks, pred_visibility]
            )

        if len(window_frames) != pred_tracks.shape[1]:
            print(f"WARNING: {len(window_frames)=} {pred_tracks.shape=}")

        video = torch.tensor(
            np.stack([adjust_img_for_plt(x) for x in window_frames])[
                : min(n, pred_tracks.shape[1])
            ]
        ).permute(0, 3, 1, 2)[None]
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
        res = vis.visualize(
            video.cpu(),
            pred_tracks[:, :n].cpu(),
            pred_visibility[:, :n].cpu(),
            query_frame=query_frame,
            filename=filename,
            save_video=filename is not None,
        )
        return res


def get_cotracker(checkpoint=None, device="cuda", use_online=True):
    if checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=checkpoint)
    else:
        name = "cotracker3_online" if use_online else "cotracker3"
        model = torch.hub.load("facebookresearch/co-tracker", name)

    model = model.to(device)
    return model


def get_queries(
    frame,
    use_kpts,
    grid_size=10,
    grid_query_frame=0,
    hw=None,
    extractor=None,
    use_frame_num_col=True,
):

    if use_kpts:
        assert extractor is not None
        # run extractor
        feats = extractor.extract(adjust_img_for_torch(frame).cuda())
        kpts = feats["keypoints"][0]
        queries = torch.cat(
            [torch.ones_like(kpts[:, :1]) * grid_query_frame, kpts], dim=1
        ).unsqueeze(0)
    else:
        assert hw is not None
        # sample on the grid
        grid_pts = get_points_on_a_grid(
            grid_size,
            # model.interp_shape,
            hw,
            device="cuda",
        )
        queries = torch.cat(
            [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
            dim=2,
        )

    if not use_frame_num_col:
        queries = queries[:, :, 1:]

    return queries


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""cotracker.models.core.model_utils"""
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
