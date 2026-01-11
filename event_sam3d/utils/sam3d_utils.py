import torch

from event_sam3d.utils.common_utils import detach_and_cpu


def save_sam3d_sparse_pred(save_path, output):
    torch.save(
        {
            k: v
            for k, v in detach_and_cpu(output).items()
            if k
            in [
                "6drotation_normalized",
                "scale",
                "shape",
                "translation",
                "translation_scale",
                "voxel",
            ]
        },
        save_path,
    )
