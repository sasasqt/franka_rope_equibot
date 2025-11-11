import json
import glob
import os
import numpy as np


# ---------- Geometry helpers ----------

def quat_to_rot_matrix(q):
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    w, x, y, z = q / n

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ], dtype=float)


def points_in_box(points, center, quat, scale):
    """
    Check which points lie inside an oriented box.

    center: (3,)
    quat: [w, x, y, z]
    scale: full side lengths -> half extents = scale / 2
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    he = 0.5 * np.asarray(scale, dtype=float)  # half extents
    R = quat_to_rot_matrix(quat)               # local -> world

    # Convert world -> local for row-vector points.
    local = (points - center) @ R

    return np.all(np.abs(local) <= he + 1e-8, axis=-1)


def build_grid(pc_a, pc_b, margin=0.05, resolution=80):
    """
    Build a cubic voxel grid covering both point clouds with some margin.
    resolution: number of steps along the largest dimension.
    """
    pc_a = np.asarray(pc_a, dtype=float)
    pc_b = np.asarray(pc_b, dtype=float)

    all_pts = np.concatenate([pc_a, pc_b], axis=0)
    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin

    extent = maxs - mins
    max_dim = float(extent.max())
    step = max_dim / float(resolution)

    xs = np.arange(mins[0], maxs[0] + 0.5 * step, step)
    ys = np.arange(mins[1], maxs[1] + 0.5 * step, step)
    zs = np.arange(mins[2], maxs[2] + 0.5 * step, step)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    return grid


def occupancy_t_shape(grid, vbar_pos, vbar_quat, vbar_scale,
                      hbar_pos, hbar_quat, hbar_scale):
    """
    Boolean occupancy of union of vbar and hbar for all grid points.
    """
    inside = np.zeros(len(grid), dtype=bool)

    inside |= points_in_box(grid, vbar_pos, vbar_quat, vbar_scale)
    inside |= points_in_box(grid, hbar_pos, hbar_quat, hbar_scale)

    return inside


# ---------- IoU per frame ----------

def volumetric_iou_for_entry(entry, resolution=80, margin=0.05):
    """
    Compute volumetric IoU between T and Target_T for one
    Isaac Sim Data entry.
    """
    T = entry["data"]["T"]
    TT = entry["data"]["Target_T"]

    pc = np.asarray(T["pc"], dtype=float)
    pc_tgt = np.asarray(TT["Target_pc"], dtype=float)

    grid = build_grid(pc, pc_tgt, margin=margin, resolution=resolution)

    # T shape (prediction/current)
    occ_T = occupancy_t_shape(
        grid,
        T["vbar_world_position"],
        T["vbar_world_orientation"],
        T["vbar_world_scale"],
        T["hbar_world_position"],
        T["hbar_world_orientation"],
        T["hbar_world_scale"],
    )

    # Target T shape (ground truth)
    occ_GT = occupancy_t_shape(
        grid,
        TT["vbar_world_position"],
        TT["vbar_world_orientation"],
        TT["vbar_world_scale"],
        TT["hbar_world_position"],
        TT["hbar_world_orientation"],
        TT["hbar_world_scale"],
    )

    inter = np.logical_and(occ_T, occ_GT).sum()
    union = np.logical_or(occ_T, occ_GT).sum()

    if union == 0:
        return 0.0

    return inter / union  # voxel volume cancels


# ---------- Helpers for your requested metric ----------

def get_start_index_after_first_drop(frames):
    """
    Find index of first frame AFTER current_time_step decreases.
    Example: ... 560, 561, 567, 210, 211 ...
    -> returns index of the 210 frame.
    If no drop is found, returns None.
    """
    if not frames:
        return None

    prev = frames[0]["current_time_step"]
    for i in range(1, len(frames)):
        cur = frames[i]["current_time_step"]
        if cur < prev:
            return i  # start counting from this frame
        prev = cur

    return None  # no reset found


def metric_for_file(path, resolution=80, margin=0.05,
                    min_count=20, verbose=False):
    """
    For a given JSON log file:

    1. Find the first time current_time_step goes from higher -> lower.
    2. Use ONLY frames from that point onward.
    3. Compute IoUs for those frames.
    4. Return the maximum IoU threshold tau such that
       at least `min_count` frames have IoU >= tau.
       (this is the `min_count`-th largest IoU)

    Returns:
        tau (float or None), num_used_frames (int)
    """
    with open(path, "r") as f:
        data = json.load(f)

    frames = data["Isaac Sim Data"]

    start_idx = get_start_index_after_first_drop(frames)
    if start_idx is None:
        # No drop -> specification says "only counts after" the drop,
        # so we treat this as having no valid frames.
        if verbose:
            print(os.path.basename(path), "NO_DROP_FOUND")
        return None, 0

    # Compute IoUs for frames from start_idx to end.
    ious = [
        volumetric_iou_for_entry(frames[i], resolution=resolution, margin=margin)
        for i in range(start_idx, len(frames))
    ]

    num_frames = len(ious)

    if num_frames < min_count:
        if verbose:
            print(
                os.path.basename(path),
                f"NOT_ENOUGH_FRAMES_AFTER_DROP ({num_frames} < {min_count})"
            )
        return None, num_frames

    # tau is the min_count-th largest IoU
    sorted_ious = sorted(ious, reverse=True)
    tau = sorted_ious[min_count - 1]

    if verbose:
        print(os.path.basename(path), "tau:", tau,
              f"(from {num_frames} frames after drop)")

    return tau, num_frames

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Cube/TCube volumetric IoU metric over JSON logs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing JSON log files."
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum number of frames for tau (default: 20)."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=80,
        help="Voxel grid resolution along largest dim (default: 80)."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Spatial margin around point clouds (default: 0.05)."
    )
    parser.add_argument(
        "--target-z-offset",
        type=float,
        default=0.0,
        help="Z offset added to TCube before IoU (default: 0.0)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details."
    )

    args = parser.parse_args()

    for path in sorted(glob.glob(os.path.join(args.log_dir, "*.json"))):
        tau, n = metric_for_file(
            path,
            resolution=args.resolution,
            margin=args.margin,
            min_count=args.min_count,
            verbose=args.verbose,
        )
        if tau is None:
            print(os.path.basename(path),
                  "-> metric: N/A (used frames:", n, ")")
        else:
            print(os.path.basename(path),
                  "-> metric:", tau, "(used frames:", n, ")")
