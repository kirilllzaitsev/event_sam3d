import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree


def load_gaussian_centers(plydata, opacity_threshold=0.5):
    """Extracts XYZ centers from a 3DGS PLY file, filtering by opacity."""
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
