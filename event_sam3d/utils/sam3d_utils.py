import torch

from event_sam3d.utils.common_utils import detach_and_cpu


def save_sam3d_sparse_pred(save_path, output):
    pts_key = 'ss' if 'ss' in output else 'voxel'
    assert pts_key in output
    torch.save(
        {
            k: v
            for k, v in detach_and_cpu(output).items()
            if k
            in [
                "6drotation_normalized",
                "rotation",
                "scale",
                "shape",
                "translation",
                "translation_scale",
                pts_key,
            ]
        },
        save_path,
    )
