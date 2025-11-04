"""
Max overlap (Z-tolerant), single maximum, Z-rotation aware.

What this does
--------------
- Parses Isaac Sim JSON (file/folder/string/dict) with light cleanup.
- Computes overlaps between two yaw-rotated boxes per frame.
- Z-tolerance gate:
    Let the Z-intervals be [cz-hz, cz+hz] and [tz-hz2, tz+hz2]
    Define separation:
        sep = 0 if intervals overlap
            = (cz-hz) - (tz+hz2) if A above B and disjoint
            = (tz-hz2) - (cz+hz) if B above A and disjoint
    A frame "passes" if sep <= z_eps (default 1e-4). Otherwise intersection=0.
- Picks **one** maximum frame (strict argmax with deterministic tie-break).
- Prints **Cube** and **TCube** volumes at that max frame (informational).
- Supports:
    * custom body keys: --cube-key / --tcube-key
    * Z shift of second body: --tcube-dz
    * metric selection: --metric {area,norm-cube-xy}
        - area          : **Volumetric IoU (3D)** = inter_volume / union_volume, with Z-tolerance gating
        - norm-cube-xy  : **XY IoU (2D)**         = inter_area   / union_area,   with Z-tolerance gating
    * per-file-only printing for folders

Assumptions
-----------
- Boxes are axis-aligned in their LOCAL frame; rotation is only about global Z (yaw).
- Roll/pitch (if present) are ignored. If quaternion is missing and 'pc' exists, yaw is
  estimated from top-face PCA.

Requires: numpy
"""

import json
import re
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
import numpy as np
import argparse
import sys
import math

# ============================
# JSON cleanup & loading
# ============================

_COMMENT_BLOCKS = re.compile(r"/\*.*?\*/", flags=re.S)
_COMMENT_LINES  = re.compile(r"(^|[^:])//.*?$", flags=re.M)
_TRAILING_COMMAS = re.compile(r",\s*([}\]])")
_ELLIPSES = re.compile(r"\.\.\.")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def _clean_json_text(text: str) -> str:
    text = _COMMENT_BLOCKS.sub("", text)
    text = _COMMENT_LINES.sub(r"\1", text)
    text = _ELLIPSES.sub("", text)
    text = _TRAILING_COMMAS.sub(r"\1", text)
    text = _CONTROL_CHARS.sub("", text)
    return text.strip()

def _read_text(source) -> str:
    if isinstance(source, (dict, list)):
        return json.dumps(source)
    if hasattr(source, "read"):
        return source.read()
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
        return str(source)
    raise TypeError("Unsupported source type. Provide a file path, JSON string, dict/list, or file-like object.")

def _extract_frame_objects_from_text(text: str) -> List[Dict[str, Any]]:
    s = _clean_json_text(text)
    objs = []
    i, n, depth, start = 0, len(s), 0, -1
    in_str, esc = False, False
    while i < n:
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{':
                if depth == 0: start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    block = s[start:i+1]
                    try:
                        obj = json.loads(block)
                        objs.append(obj)
                    except Exception:
                        pass
                    start = -1
        i += 1
    for o in objs:
        if isinstance(o, dict) and "Isaac Sim Data" in o and isinstance(o["Isaac Sim Data"], list):
            return o["Isaac Sim Data"]
    frames = []
    for o in objs:
        if isinstance(o, dict) and "data" in o:
            frames.append(o)
    return frames

def load_isaac_frames(source,
                      cube_key: str = "Cube",
                      tcube_key: str = "TCube") -> List[Dict[str, Any]]:
    text = _read_text(source)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = json.loads(_clean_json_text(text))
        except Exception:
            data = None

    frames = []
    if isinstance(data, dict) and "Isaac Sim Data" in data and isinstance(data["Isaac Sim Data"], list):
        frames = data["Isaac Sim Data"]
    elif isinstance(data, list):
        frames = data
    elif data is None:
        frames = _extract_frame_objects_from_text(text)

    good = []
    for f in frames:
        if not isinstance(f, dict): continue
        d = f.get("data", {})
        c, t = d.get(cube_key, {}), d.get(tcube_key, {})
        def _has_valid(b):
            return ("pc" in b and isinstance(b["pc"], list) and len(b["pc"]) >= 8) or \
                   ("cube_world_position" in b and "cube_world_orientation" in b)
        if _has_valid(c) and _has_valid(t):
            good.append(f)
    return good

# ============================
# Math & geometry helpers
# ============================

def _guess_quat_layout(q: np.ndarray) -> Tuple[float,float,float,float]:
    q = np.asarray(q, dtype=float).flatten()
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have 4 components")
    if abs(q[0]) >= abs(q[-1]):
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0: return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)

def _yaw_from_quaternion(q) -> float:
    w, x, y, z = _guess_quat_layout(np.asarray(q, dtype=float))
    t0 = 2.0*(w*z + x*y)
    t1 = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(t0, t1)

def _edge_lengths_from_pc(pc: np.ndarray, tol: float = 1e-6) -> Tuple[float,float,float]:
    pts = np.asarray(pc, dtype=float)
    n = pts.shape[0]
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > tol:
                dists.append(d)
    dists = np.array(sorted(dists))
    uniq = []
    for d in dists:
        if not uniq or abs(d - uniq[-1]) > tol:
            uniq.append(d)
        if len(uniq) == 3: break
    if len(uniq) == 0: return (0.0, 0.0, 0.0)
    if len(uniq) == 1: return (uniq[0], uniq[0], uniq[0])
    if len(uniq) == 2: return (uniq[0], uniq[0], uniq[1])
    return tuple(uniq[:3])

def _body_dims_and_pose(frame: Dict[str,Any],
                        key: str,
                        default_half_extent: float = 0.03,
                        yaw_fallback_from_pc: bool = True) -> Tuple[float,float,float,float,float,float,float]:
    """
    Returns (cx, cy, cz, hx, hy, hz, yaw).
    """
    d = frame.get("data", {}).get(key, {})
    pos = np.asarray(d.get("cube_world_position", [0,0,0]), dtype=float)
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

    he = None
    if "half_extents" in d:
        he = np.asarray(d["half_extents"], dtype=float).reshape(3)
    elif "half_extent" in d:
        he = np.asarray(d["half_extent"], dtype=float).reshape(3)
    elif "size" in d:
        he = (np.asarray(d["size"], dtype=float).reshape(3))/2.0

    if he is not None:
        hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
    elif "pc" in d and isinstance(d["pc"], list) and len(d["pc"]) >= 8:
        pc = np.asarray(d["pc"], dtype=float)
        zmin, zmax = np.min(pc[:,2]), np.max(pc[:,2])
        hz = float(0.5*(zmax - zmin))
        a,b,c = _edge_lengths_from_pc(pc)
        edges = [a,b,c]
        diffs = [abs(e - 2.0*hz) for e in edges]
        idxs = np.argsort(diffs)[::-1][:2]
        e_xy = [edges[i] for i in idxs]
        hx, hy = 0.5*min(e_xy), 0.5*max(e_xy)
    else:
        hx = hy = hz = float(default_half_extent)

    if "cube_world_orientation" in d:
        yaw = _yaw_from_quaternion(d["cube_world_orientation"])
    elif yaw_fallback_from_pc and "pc" in d and isinstance(d["pc"], list) and len(d["pc"]) >= 8:
        pc = np.asarray(d["pc"], dtype=float)
        zmax = np.max(pc[:,2])
        top = pc[np.abs(pc[:,2] - zmax) < 1e-5][:, :2]
        if top.shape[0] >= 4:
            mu = np.mean(top, axis=0)
            X = top - mu
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            axis = Vt[0]
            yaw = float(math.atan2(axis[1], axis[0]))
        else:
            yaw = 0.0
    else:
        yaw = 0.0

    return cx, cy, cz, hx, hy, hz, yaw

# ============================
# XY rectangle intersection (Z-rotation aware)
# ============================

def _rect_xy_corners(cx: float, cy: float, hx: float, hy: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],[s, c]], dtype=float)
    local = np.array([[-hx, -hy], [-hx, hy], [hx, hy], [hx, -hy]], dtype=float)  # CCW
    return (local @ R.T) + np.array([cx, cy])

def _is_ccw(poly: np.ndarray) -> bool:
    x, y = poly[:,0], poly[:,1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0.0

def _clip_against_edge(subject: List[np.ndarray], a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    def inside(p):
        return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= 0.0
    def intersect(p1, p2):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = a; x4,y4 = b
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-18:
            return p2
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py], dtype=float)
    if not subject:
        return []
    out = []
    prev = subject[-1]; prev_in = inside(prev)
    for curr in subject:
        curr_in = inside(curr)
        if curr_in:
            if not prev_in: out.append(intersect(prev, curr))
            out.append(curr)
        elif prev_in:
            out.append(intersect(prev, curr))
        prev, prev_in = curr, curr_in
    return out

def _poly_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    if not _is_ccw(poly1): poly1 = poly1[::-1]
    if not _is_ccw(poly2): poly2 = poly2[::-1]
    subject = [p for p in poly1]
    clipper = [p for p in poly2]
    out = subject
    for i in range(len(clipper)):
        out = _clip_against_edge(out, clipper[i], clipper[(i+1) % len(clipper)])
        if not out:
            return 0.0
    P = np.array(out, dtype=float)
    x, y = P[:,0], P[:,1]
    area = 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)

# ============================
# Z helpers & overlaps
# ============================

def _z_separation(cz: float, hz: float, tz: float, hz2: float) -> float:
    """
    Minimum positive separation between the two Z-intervals.
    Returns 0 if intervals overlap; otherwise the positive gap.
    """
    a_lo, a_hi = cz - hz, cz + hz
    b_lo, b_hi = tz - hz2, tz + hz2
    if a_hi < b_lo:
        return b_lo - a_hi
    if b_hi < a_lo:
        return a_lo - b_hi
    return 0.0

def _z_overlap_len(cz: float, hz: float, tz: float, hz2: float) -> float:
    a_lo, a_hi = cz - hz, cz + hz
    b_lo, b_hi = tz - hz2, tz + hz2
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))

# ============================
# Overlap stats with Z tolerance
# ============================

def _overlap_stats_if_z_ok(frame: Dict[str,Any],
                           cube_key: str, tcube_key: str,
                           default_half_extent: float,
                           tcube_dz: float,
                           z_eps: float) -> Dict[str, float]:
    """
    Compute XY area & volumetric stats.
    - If Z separation > z_eps or Cube volume < TCube volume: intersections => 0.
    - XY IoU uses gated 2D intersection.
    - Volumetric IoU uses actual Z-overlap length (0 if slabs don't overlap in Z).
    """
    cx, cy, cz, hx, hy, hz, yaw    = _body_dims_and_pose(frame, cube_key,  default_half_extent)
    tx, ty, tz, hx2, hy2, hz2, yw2 = _body_dims_and_pose(frame, tcube_key, default_half_extent)
    tz += float(tcube_dz)

    sep = _z_separation(cz, hz, tz, hz2)

    poly1 = _rect_xy_corners(cx, cy, hx, hy, yaw)
    poly2 = _rect_xy_corners(tx, ty, hx2, hy2, yw2)
    area_xy = _poly_intersection_area(poly1, poly2)

    cube_area_xy  = float((2*hx)*(2*hy))
    tcube_area_xy = float((2*hx2)*(2*hy2))
    cube_volume   = float((2*hx)*(2*hy)*(2*hz))
    tcube_volume  = float((2*hx2)*(2*hy2)*(2*hz2))

    qualifies_volume = (cube_volume >= tcube_volume)

    # Gate intersections
    if sep <= z_eps and qualifies_volume:
        inter_area_xy = area_xy
        z_overlap     = _z_overlap_len(cz, hz, tz, hz2)  # true Z overlap (0 if just "near" but disjoint)
        inter_vol     = inter_area_xy * z_overlap
    else:
        inter_area_xy = 0.0
        z_overlap     = 0.0
        inter_vol     = 0.0

    # XY IoU
    union_xy = cube_area_xy + tcube_area_xy - inter_area_xy
    iou_xy   = (inter_area_xy / union_xy) if union_xy > 0.0 else 0.0

    # Volumetric IoU
    union_vol = cube_volume + tcube_volume - inter_vol
    iou_3d    = (inter_vol / union_vol) if union_vol > 0.0 else 0.0

    return {
        "z_separation": float(sep),
        "z_overlap_len": float(z_overlap),

        "cube_area_xy": float(cube_area_xy),
        "tcube_area_xy": float(tcube_area_xy),

        "cube_volume": float(cube_volume),
        "tcube_volume": float(tcube_volume),

        "inter_area_xy": float(inter_area_xy),
        "union_area_xy": float(union_xy),
        "area_iou_xy": float(iou_xy),

        "inter_volume": float(inter_vol),
        "union_volume": float(union_vol),
        "volume_iou_3d": float(iou_3d),
    }

# ============================
# SINGLE strict argmax (one maximum only)
# ============================

def _unique_argmax(values: List[float], times: List[float], steps: List[int]) -> int:
    """
    Return a single index of the strict maximum.
    Tie-breaks (exact ties within 1e-12):
      1) earliest time, 2) smallest time_step, 3) smallest index.
    """
    if not values:
        return -1
    m = max(values)
    tol = 1e-12
    cand = [i for i, v in enumerate(values) if math.isclose(v, m, rel_tol=tol, abs_tol=0.0)]
    if len(cand) == 1:
        return cand[0]
    return min(cand, key=lambda i: (times[i], steps[i], i))

# ============================
# Analyzer
# ============================

def analyze_frames(isaac_json: Dict[str, Any],
                   cube_key: str = "Cube",
                   tcube_key: str = "TCube",
                   z_eps: float = 1e-4,
                   default_half_extent: float = 0.03,
                   tcube_dz: float = 0.0) -> Dict[str, Any]:
    """
    Compute per-frame overlaps (with Z tolerance) and return ONE maximum for:
      - 'area'           -> volumetric IoU (3D)
      - 'norm-cube-xy'   -> XY IoU (2D)
    """
    if isinstance(isaac_json, dict) and "Isaac Sim Data" in isaac_json:
        frames_in = isaac_json["Isaac Sim Data"]
    elif isinstance(isaac_json, list):
        frames_in = isaac_json
    else:
        frames_in = []

    results = []
    for idx, frame in enumerate(frames_in):
        try:
            ts = int(frame.get("current_time_step"))
            t  = float(frame.get("current_time"))

            stats = _overlap_stats_if_z_ok(frame, cube_key, tcube_key, default_half_extent, tcube_dz, z_eps)

            results.append({
                "index": idx,
                "time_step": ts,
                "time": t,
                **stats,  # includes area_iou_xy and volume_iou_3d
                "tcube_dz": float(tcube_dz),
                "z_eps": float(z_eps),
                # Back-compat keys (naming kept):
                "area_norm_by_cube_xy": float(stats["area_iou_xy"]),
            })
        except Exception:
            continue

    if not results:
        return {
            "max_area_xy": 0.0,                  # will represent volumetric IoU now
            "max_frame": None,
            "max_area_norm_by_cube_xy": 0.0,     # XY IoU
            "max_frame_norm_by_cube_xy": None,
            "per_frame": []
        }

    # 'area' metric now means volumetric IoU
    raw_vals  = [r["volume_iou_3d"] for r in results]
    norm_vals = [r["area_norm_by_cube_xy"] for r in results]  # XY IoU
    times     = [r["time"] for r in results]
    steps     = [r["time_step"] for r in results]

    i_raw  = _unique_argmax(raw_vals,  times, steps)
    i_norm = _unique_argmax(norm_vals, times, steps)

    max_raw  = raw_vals[i_raw]
    max_norm = norm_vals[i_norm]

    max_frame_raw = {
        "time_step": results[i_raw]["time_step"],
        "time": results[i_raw]["time"],
        "value": results[i_raw]["volume_iou_3d"],  # volumetric IoU
        "cube_volume": results[i_raw]["cube_volume"],
        "tcube_volume": results[i_raw]["tcube_volume"],
        "z_separation": results[i_raw]["z_separation"],
    }
    max_frame_norm = {
        "time_step": results[i_norm]["time_step"],
        "time": results[i_norm]["time"],
        "value": results[i_norm]["area_norm_by_cube_xy"],  # XY IoU
        "cube_volume": results[i_norm]["cube_volume"],
        "tcube_volume": results[i_norm]["tcube_volume"],
        "z_separation": results[i_norm]["z_separation"],
    }

    return {
        "max_area_xy": max_raw,                       # volumetric IoU (semantic change)
        "max_frame": max_frame_raw,                   # volumetric IoU details
        "max_area_norm_by_cube_xy": max_norm,         # XY IoU
        "max_frame_norm_by_cube_xy": max_frame_norm,  # XY IoU details
        "per_frame": results
    }

# ============================
# Folder utilities
# ============================

def _iter_json_files(root: Union[str, Path],
                     pattern: str = "*.json",
                     recursive: bool = True) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    files = list(root.rglob(pattern) if recursive else root.glob(pattern))
    files.sort(key=lambda p: (p.name.lower(), str(p.parent).lower()))
    return files

def load_all_jsons_from_folder(folder: Union[str, Path],
                               pattern: str = "*.json",
                               recursive: bool = True,
                               cube_key: str = "Cube",
                               tcube_key: str = "TCube") -> Dict[str, Any]:
    datasets, errors = [], []
    for fp in _iter_json_files(folder, pattern, recursive):
        try:
            frames = load_isaac_frames(fp, cube_key=cube_key, tcube_key=tcube_key)
            datasets.append({"path": str(fp), "frames": frames})
        except Exception as e:
            errors.append({"path": str(fp), "error": repr(e)})
    return {"datasets": datasets, "errors": errors, "num_files": len(datasets) + len(errors)}

def max_overlap_from_folder(folder: Union[str, Path],
                            pattern: str = "*.json",
                            recursive: bool = True,
                            z_eps: float = 1e-4,
                            default_half_extent: float = 0.03,
                            tcube_dz: float = 0.0,
                            cube_key: str = "Cube",
                            tcube_key: str = "TCube",
                            metric: str = "area") -> Dict[str, Any]:
    """
    metric:
      - 'area'           : volumetric IoU (3D) with Z tolerance
      - 'norm-cube-xy'   : XY IoU (2D) with Z tolerance
    """
    load_res = load_all_jsons_from_folder(folder, pattern, recursive, cube_key=cube_key, tcube_key=tcube_key)
    per_file = []
    overall_max = 0.0
    occurrence = None  # keep only one

    key_val   = "max_area_xy" if metric == "area" else "max_area_norm_by_cube_xy"
    key_frame = "max_frame"   if metric == "area" else "max_frame_norm_by_cube_xy"

    for ds in load_res["datasets"]:
        summary = analyze_frames({"Isaac Sim Data": ds["frames"]},
                                 cube_key=cube_key, tcube_key=tcube_key,
                                 z_eps=z_eps, default_half_extent=default_half_extent, tcube_dz=tcube_dz)
        per_file.append({"path": ds["path"], **summary})
        m = summary[key_val]
        f = summary[key_frame]
        if f is None:
            continue
        if m > overall_max:
            overall_max = m
            occurrence = (ds["path"], f["time_step"], f["time"], f["value"])
        elif math.isclose(m, overall_max, rel_tol=1e-12, abs_tol=0.0) and occurrence is not None:
            # tie: keep earliest (time, then step)
            cand = (ds["path"], f["time_step"], f["time"], f["value"])
            occurrence = min([occurrence, cand], key=lambda x: (x[2], x[1]))

    return {
        "overall_max": overall_max,
        "metric": metric,
        "occurrence": occurrence,   # a single (path, step, time, value) or None
        "per_file": per_file,
        "errors": load_res["errors"],
        "num_files": load_res["num_files"],
    }

def per_file_max_overlaps(folder: Union[str, Path],
                          pattern: str = "*.json",
                          recursive: bool = True,
                          z_eps: float = 1e-4,
                          default_half_extent: float = 0.03,
                          tcube_dz: float = 0.0,
                          cube_key: str = "Cube",
                          tcube_key: str = "TCube",
                          metric: str = "area"):
    res = max_overlap_from_folder(folder, pattern, recursive,
                                  z_eps=z_eps, default_half_extent=default_half_extent, tcube_dz=tcube_dz,
                                  cube_key=cube_key, tcube_key=tcube_key, metric=metric)
    return res["per_file"], res["errors"], res["num_files"]

# ============================
# CLI printing
# ============================

def _print_one_max(prefix: str, frame: Dict[str, Any]) -> None:
    print(prefix + f"at step={frame['time_step']}, time={frame['time']:.6f}, value={frame['value']}")
    print(f"{prefix}Cube volume : {frame['cube_volume']}")
    print(f"{prefix}TCube volume: {frame['tcube_volume']}")
    print(f"{prefix}Z separation (m): {frame['z_separation']}")

def _print_file_summary(path: str, summary: Dict[str, Any], metric: str = "area") -> None:
    print(f"\nFile: {path}")
    if metric == "area":
        print(f"  Max volumetric IoU (Z≤eps): {summary['max_area_xy']}")
        if summary["max_frame"]:
            _print_one_max("    ", summary["max_frame"])
    else:
        print(f"  Max XY IoU (Z≤eps): {summary['max_area_norm_by_cube_xy']}")
        if summary["max_frame_norm_by_cube_xy"]:
            _print_one_max("    ", summary["max_frame_norm_by_cube_xy"])

def _print_folder_summary(res: Dict[str, Any]) -> None:
    label = "Volumetric IoU (Z≤eps)" if res.get("metric","area")=="area" else "XY IoU (Z≤eps)"
    print("\n=== Folder Summary ===")
    print(f"Files processed: {res['num_files']}")
    print(f"Overall max ({label}): {res['overall_max']}")
    if res["occurrence"]:
        path, ts, t, v = res["occurrence"]
        print(f"Occurrence: file={path}, step={ts}, time={t:.6f}, value={v}")
    if res["errors"]:
        print("\nErrors:")
        for e in res["errors"]:
            print(f"  {e['path']}: {e['error']}")

def _print_per_file_only(res: Dict[str, Any]) -> None:
    metric = res.get("metric", "area")
    key_val   = "max_area_xy" if metric == "area" else "max_area_norm_by_cube_xy"
    key_frame = "max_frame"   if metric == "area" else "max_frame_norm_by_cube_xy"
    label = "Volumetric IoU (Z≤eps)" if metric == "area" else "XY IoU (Z≤eps)"

    print(f"\n=== Per-file maxima ({label}) ===")
    for item in res["per_file"]:
        print(f"File: {item['path']}")
        print(f"  Max: {item[key_val]}")
        if item[key_frame]:
            _print_one_max("    ", item[key_frame])
    if res["errors"]:
        print("\nErrors:")
        for e in res["errors"]:
            print(f"  {e['path']}: {e['error']}")

# ============================
# CLI
# ============================

def main():
    parser = argparse.ArgumentParser(description="Max overlap (Z tolerance, single maximum), Z-rotation aware.")
    parser.add_argument("path", help="Path to a JSON file or a folder containing JSON files.")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern for JSON files in folder (default: *.json)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subfolders")
    parser.add_argument("--half", type=float, default=0.03, help="Default half-extent (m) if not present in JSON")
    parser.add_argument("--tcube-dz", type=float, default=0.0,
                        help="Translate the SECOND body along +Z by this many meters (positive=up, negative=down).")
    parser.add_argument("--z-eps", type=float, default=1e-4,
                        help="Z-axis tolerance (meters) to accept overlap when slabs are separated (default: 1e-4).")
    parser.add_argument("--cube-key", type=str, default="Cube",
                        help="Name of the FIRST body inside frame['data'] (default: 'Cube').")
    parser.add_argument("--tcube-key", type=str, default="TCube",
                        help="Name of the SECOND body inside frame['data'] (default: 'TCube').")
    parser.add_argument("--metric", choices=["area", "norm-cube-xy"], default="area",
                        help="Max metric: 'area'=Volumetric IoU (3D), 'norm-cube-xy'=XY IoU (2D).")
    parser.add_argument("--per-file-only", action="store_true",
                        help="Print only each JSON's max (suppress overall summary).")
    args = parser.parse_args()

    p = Path(args.path)
    recursive = not args.no_recursive

    if p.is_file():
        try:
            frames = load_isaac_frames(p, cube_key=args.cube_key, tcube_key=args.tcube_key)
            summary = analyze_frames({"Isaac Sim Data": frames},
                                     cube_key=args.cube_key, tcube_key=args.tcube_key,
                                     z_eps=args.z_eps, default_half_extent=args.half, tcube_dz=args.tcube_dz)
            _print_file_summary(str(p), summary, metric=args.metric)
        except Exception as e:
            print(f"Error processing file {p}: {e}", file=sys.stderr)
            sys.exit(1)
    elif p.is_dir():
        try:
            res = max_overlap_from_folder(
                p, pattern=args.pattern, recursive=recursive,
                z_eps=args.z_eps, default_half_extent=args.half, tcube_dz=args.tcube_dz,
                cube_key=args.cube_key, tcube_key=args.tcube_key, metric=args.metric
            )
            if args.per_file_only:
                _print_per_file_only(res)
            else:
                _print_folder_summary(res)
        except Exception as e:
            print(f"Error processing folder {p}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Path not found: {p}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example:
# python cube.py --tcube-dz 0.03 --per-file-only --metric area --z-eps 0.005  /path/to/folder     # volumetric IoU
# python cube.py --tcube-dz 0.03 --per-file-only --metric norm-cube-xy --z-eps 0.005 /path/to/folder  # XY IoU
