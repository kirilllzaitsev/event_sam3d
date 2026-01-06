import argparse
import sys

import numpy as np
import open_clip
import torch
import trimesh
import yaml
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

from event_sam3d.config import RELATED_DIR
from event_sam3d.utils.common_utils import cast_to_torch


def pc_norm(pc):
    """pc: NxC, return NxC"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_rgb_from_3dgs_ply(plydata):
    v = plydata["vertex"]

    # 1. Extract the DC (Degree 0) coefficients
    # 'f_dc_0', 'f_dc_1', 'f_dc_2' correspond to R, G, B channels in SH space
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)

    # 2. Apply SH Constant (0.28209...)
    # Degree 0 SH basis function is a constant: 1/2 * sqrt(1/pi)
    SH_C0 = 0.28209479177387814
    rgb = sh_dc * SH_C0

    # 3. Add 0.5 offset (SH standard) and Clip?
    # Actually, official 3DGS often treats these just as logits if 'activation' is used.
    # But a common simplified conversion for visualization is:
    rgb = rgb + 0.5

    # Ideally, ensure range [0, 1]
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb  # Shape (N, 3)


class Uni3DScorer:
    def __init__(self) -> None:
        sys.path.append("/cluster/home/kzaitse/event_sam3d/related_work/rec/Uni3D")
        from models.uni3d import create_uni3d

        args = argparse.Namespace(
            **yaml.load(
                open(f"{RELATED_DIR}/rec/Uni3D/config.yaml"), Loader=yaml.UnsafeLoader
            )
        )
        args.pretrained_pc = f"{RELATED_DIR}/rec/Uni3D/checkpoints/uni3d-s.pt"
        args.pretrained_pc = ""
        args.distributed = False
        args.ckpt_path = f"{RELATED_DIR}/rec/Uni3D/checkpoints/uni3d-s.pt"

        model = create_uni3d(args)
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        print("loaded checkpoint {}".format(args.ckpt_path))
        sd = checkpoint["module"]
        if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        self.model = model

        clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
            args.clip_model,  # The specific CLIP model Uni3D-Giant uses
            pretrained=f"{RELATED_DIR}/rec/Uni3D/clip_model/open_clip_pytorch_model.bin",
            # force_custom_clip=True
        )
        device = "cuda"
        clip_model.eval().to(device)
        model.eval().to(device)
        self.clip_model = clip_model
        self.preprocess = preprocess_val

    def score(self, mask_obj, pts1):
        image_input = self.preprocess(Image.fromarray(mask_obj)).unsqueeze(0).cuda()
        pc_input = cast_to_torch(pc_norm(pts1)).unsqueeze(0).cuda()
        pc_input = torch.cat(
            [pc_input, torch.ones_like(pc_input).float() * 0.4], dim=-1
        )

        with torch.no_grad():

            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pc_features = self.model.encode_pc(pc_input)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ pc_features.T).item()
        return similarity


def compute_viou(pts1, pts2):

    mesh_1 = trimesh.points.PointCloud(pts1).convex_hull
    mesh_2 = trimesh.points.PointCloud(pts2).convex_hull

    # Voxelize both meshes to pitch 0.01 (e.g., 1cm)
    vox_1 = mesh_1.voxelized(pitch=0.01).fill()
    vox_2 = mesh_2.voxelized(pitch=0.01).fill()

    intersection = np.logical_1nd(vox_1.matrix, vox_2.matrix).sum()
    union = np.logical_or(vox_1.matrix, vox_2.matrix).sum()
    iou = intersection / union
    return iou


def voxelize_point_cloud(points, resolution=64, bounds=(-0.5, 0.5)):
    """
    Voxelizes a point cloud into a fixed boolean grid.

    Args:
        points (np.ndarray): (N, 3) Point cloud.
        resolution (int): Grid size (e.g., 64 for 64x64x64).
        bounds (tuple): The spatial bounds of the grid (min, max).
                        Standard generative models usually output in [-0.5, 0.5].

    Returns:
        voxel_grid (np.ndarray): (res, res, res) Boolean array.
    """
    min_bound, max_bound = bounds
    scene_scale = max_bound - min_bound

    # 1. Normalize points to [0, 1] range relative to the bounding box
    # Formula: (x - min) / (max - min)
    norm_points = (points - min_bound) / scene_scale

    # 2. Scale to Integer Coordinates [0, resolution-1]
    # We use floor() to map 0.99 -> 0, 1.01 -> 1, etc.
    grid_coords = np.floor(norm_points * resolution).astype(int)

    # 3. Filter Out-of-Bounds points
    # (Some points might be slightly outside the box due to noise)
    mask = ((grid_coords >= 0) & (grid_coords < resolution)).all(axis=1)
    grid_coords = grid_coords[mask]

    # 4. Create the Grid (Histogram)
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Advanced indexing to set occupied voxels to True
    # We only care *if* a voxel is occupied, not *how many* points (boolean)
    voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] = True

    return voxel_grid


def compute_viou_np(pts_pred, pts_gt):
    """
    Calculates Volumetric Intersection over Union.
    """
    grid_pred = voxelize_point_cloud(pts_pred, resolution=64, bounds=(-0.5, 0.5))
    grid_gt = voxelize_point_cloud(pts_gt, resolution=64, bounds=(-0.5, 0.5))
    intersection = np.logical_and(grid_pred, grid_gt).sum()

    union = np.logical_or(grid_pred, grid_gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


def load_gaussian_centers(plydata=None, plypath=None, opacity_threshold=0.5):
    """Extracts XYZ centers from a 3DGS PLY file, filtering by opacity."""
    if plydata is None:
        assert plypath is not None
        plydata = PlyData.read(plypath)
    v = plydata["vertex"]

    # 3DGS usually stores opacity as 'opacity' (logit) or scale
    # This varies by implementation; adjust 'opacity' key as needed
    if "opacity" in v:
        opacities = 1 / (1 + np.exp(-v["opacity"]))  # Sigmoid if stored as logit
        mask = opacities > opacity_threshold
        points = np.stack((v["x"], v["y"], v["z"]), axis=-1)[mask]
    else:
        # Fallback: take all points
        points = np.stack((v["x"], v["y"], v["z"]), axis=-1)

    return points


def load_mesh_samples(mesh_path, num_samples=30000):
    """Loads a mesh and samples points from its surface."""
    mesh = trimesh.load(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points


def compute_chamfer_distance(p1, p2):
    """Computes symmetric Chamfer Distance."""
    # Build KD-Trees for fast nearest neighbor search
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)

    # Distances from P1 to nearest in P2
    dist1, _ = tree2.query(p1, k=1)
    # Distances from P2 to nearest in P1
    dist2, _ = tree1.query(p2, k=1)

    return dist1, dist2


def compute_chamfer_distance_acc(p1, p2, th=0.01):
    dist1, dist2 = compute_chamfer_distance(p1, p2)
    dist = np.mean(dist1**2) + np.mean(dist2**2)
    pr = (dist1 < th).mean()
    rec = (dist2 < th).mean()
    f1 = 2 * (pr * rec) / (pr + rec + 1e-8)
    return {"precision": pr, "recall": rec, "f1": f1, "dist": dist}
