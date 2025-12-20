from collections import defaultdict

import numpy as np
import torch

from event_sam3d.utils.common_utils import cast_to_numpy, get_transpose_func, istensor


def calc_rt_errors(pred_rt, gt_rt, pts=None):
    """Calculate rotation and translation errors between two poses.
    Can handle symmetries in Linemod objects.
    """
    pred_rt = cast_to_numpy(pred_rt).squeeze()
    gt_rt = cast_to_numpy(gt_rt).squeeze()

    if len(pred_rt.shape) == 3:
        errors = defaultdict(list)
        for i in range(pred_rt.shape[0]):
            error = calc_rt_errors(pred_rt[i], gt_rt[i])
            for k, v in error.items():
                errors[k].append(v)
        return {k: np.mean(v) for k, v in errors.items()}

    T1 = pred_rt[:3, 3]
    T2 = gt_rt[:3, 3]
    rot_pred = pred_rt[:3, :3]
    rot_gt = gt_rt[:3, :3]

    theta = calc_r_error(rot_pred, rot_gt)
    shift = calc_t_error(T1, T2)
    result = {"r_err": theta, "t_err": shift}

    if pts is not None:
        add = calc_add(pred_rt, gt_rt, pts)
        result["add"] = add

    return result


def calc_t_error(T1, T2, do_reduce=True):
    if istensor(T1):
        res = torch.linalg.norm(T1 - T2, dim=-1)
    else:
        res = np.linalg.norm(T1 - T2, axis=-1)
    if do_reduce:
        res = res.mean()
    return res


def calc_r_error(rot_pred, rot_gt, do_reduce=True, do_return_deg=True):

    if isinstance(rot_pred, np.ndarray):
        rot_pred = torch.tensor(rot_pred).float()
    if isinstance(rot_gt, np.ndarray):
        rot_gt = torch.tensor(rot_gt).float()

    if rot_pred.ndim == 3:
        thetas = [
            calc_r_error(
                rot_pred[i],
                rot_gt[i],
                do_return_deg=do_return_deg,
                do_reduce=do_reduce,
            )
            for i in range(rot_pred.shape[0])
        ]
        thetas = torch.stack(thetas)
        if do_reduce:
            thetas = thetas.mean()
        return thetas

    R_rel = rot_pred.transpose(-1, -2) @ rot_gt
    trace = torch.clamp((torch.einsum("...ii", R_rel) - 1) / 2, -1.0, 1.0)
    theta = torch.acos(trace)

    if do_reduce:
        theta = theta.mean()
    if do_return_deg:
        theta = theta * 180 / torch.pi
    return theta


def normalize_rotation_matrix(matrix):
    if istensor(matrix):
        U, _, Vt = torch.linalg.svd(matrix)
        return torch.matmul(U, Vt)
    else:
        U, _, Vt = np.linalg.svd(matrix)
        return np.dot(U, Vt)


def calc_add(pred_rt, gt_rt, pts):
    pts1 = transform_pts(pts, rt=pred_rt)
    pts2 = transform_pts(pts, rt=gt_rt)
    return np.linalg.norm(pts1 - pts2, axis=-1).mean()


def transform_pts(pts, r=None, t=None, rt=None):
    """
    Returns:
        nx3 ndarray with transformed 3D points.
    """
    if rt is not None:
        r = rt[..., :3, :3]
        t = rt[..., :3, 3]
    t_func = get_transpose_func(pts)
    if pts.shape[-1] == 3:
        pts = t_func(pts)
    new_pts = r @ pts + t[..., None]
    return t_func(new_pts)
