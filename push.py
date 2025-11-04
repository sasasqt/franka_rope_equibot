"""
Max XY-overlap (Z-tolerant), single maximum, Z-rotation aware.
Supports Cube/TCube and T/Target_T (vbar + hbar). Includes --xy-pad and --debug-pairs.

What this does
--------------
- Parses Isaac Sim JSON (file/folder/string/dict) with light cleanup.
- Each body is a union of 1..2 rotated rectangles in XY:
    * Cube/TCube: 1 rectangle (old format).
    * T/Target_T: 2 rectangles (vbar + hbar) via ..._world_position/orientation/scale.
- Computes the **XY intersection area** between those two unions per frame.
- **Z-tolerant**: pair contributes only if pairwise Z gap <= z_eps.
- Picks **one** maximum frame (strict argmax with deterministic tie-break).
- Reports "volumes" (sum of bar volumes minus internal overlap) for both bodies.
- New:
    * --xy-pad      : pads X/Y half-extents of every rectangle (treat near-misses as contact).
    * --debug-pairs : for a single file, prints per-pair overlaps (v∩v, v∩h, …) on top-N frames.

CLI keeps: --cube-key / --tcube-key / --tcube-dz / --metric / --z-eps / folder mode.

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
from itertools import combinations

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
    raise TypeError("Unsupported source type.")

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
    """
    Flexible loader: accepts frames whenever BOTH body keys are present in frame['data'].
    (We no longer require specific inner fields here; analysis step will validate.)
    """
    text = _read_text(source)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = json.loads(_clean_json_text(text))
        except Exception:
            data = None

    if isinstance(data, dict) and "Isaac Sim Data" in data:
        frames = data["Isaac Sim Data"]
    elif isinstance(data, list):
        frames = data
    elif data is None:
        frames = _extract_frame_objects_from_text(text)
    else:
        frames = []

    good = []
    for f in frames:
        if not isinstance(f, dict): 
            continue
        d = f.get("data", {})
        if isinstance(d, dict) and cube_key in d and tcube_key in d:
            good.append(f)
    return good

# ============================
# Math & geometry helpers
# ============================

def _guess_quat_layout(q: np.ndarray) -> Tuple[float,float,float,float]:
    q = np.asarray(q, dtype=float).flatten()
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have 4 components")
    # Heuristic: prefer w-first if it looks larger in magnitude
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

# ---- polygons ----

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

def _poly_intersection(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """Return intersection polygon (possibly empty) of two convex polygons."""
    if poly1.size == 0 or poly2.size == 0:
        return np.zeros((0,2), dtype=float)
    P1 = poly1.copy()
    P2 = poly2.copy()
    if not _is_ccw(P1): P1 = P1[::-1]
    if not _is_ccw(P2): P2 = P2[::-1]
    subject = [p for p in P1]
    clipper = [p for p in P2]
    out = subject
    for i in range(len(clipper)):
        out = _clip_against_edge(out, clipper[i], clipper[(i+1) % len(clipper)])
        if not out:
            return np.zeros((0,2), dtype=float)
    return np.array(out, dtype=float)

def _poly_area(poly: np.ndarray) -> float:
    if poly.size == 0:
        return 0.0
    x, y = poly[:,0], poly[:,1]
    return 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _poly_intersection_many(polys: List[np.ndarray]) -> np.ndarray:
    if not polys:
        return np.zeros((0,2), dtype=float)
    res = polys[0]
    for p in polys[1:]:
        res = _poly_intersection(res, p)
        if res.size == 0:
            break
    return res

def _area_union(polys: List[np.ndarray]) -> float:
    """
    Union area via inclusion–exclusion for up to 4 convex polygons.
    Robust for our use (<=4 convex intersection polygons).
    """
    polys = [p for p in polys if p.size != 0]
    n = len(polys)
    if n == 0: return 0.0
    if n == 1: return _poly_area(polys[0])
    total = 0.0
    for k in range(1, n+1):
        sign = 1.0 if k % 2 == 1 else -1.0
        for idxs in combinations(range(n), k):
            inter = _poly_intersection_many([polys[i] for i in idxs])
            total += sign * _poly_area(inter)
    return float(max(total, 0.0))

# ============================
# Bodies -> rectangles
# ============================

def _z_separation(cz: float, hz: float, tz: float, hz2: float) -> float:
    a_lo, a_hi = cz - hz, cz + hz
    b_lo, b_hi = tz - hz2, tz + hz2
    if a_hi < b_lo:
        return b_lo - a_hi
    if b_hi < a_lo:
        return a_lo - b_hi
    return 0.0

def _dims_from_pc(d: Dict[str,Any], default_half_extent: float) -> Tuple[float,float,float,float,float,float,float]:
    # Fallback single box via 'pc' (old format)
    pos = np.asarray(d.get("cube_world_position", [0,0,0]), dtype=float)
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

    if "pc" in d and isinstance(d["pc"], list) and len(d["pc"]) >= 8:
        pc = np.asarray(d["pc"], dtype=float)
        zmin, zmax = np.min(pc[:,2]), np.max(pc[:,2])
        hz = float(0.5*(zmax - zmin))
        a,b,c = _edge_lengths_from_pc(pc)
        edges = [a,b,c]
        diffs = [abs(e - 2.0*hz) for e in edges]
        idxs = np.argsort(diffs)[::-1][:2]
        e_xy = [edges[i] for i in idxs]
        hx, hy = 0.5*min(e_xy), 0.5*max(e_xy)
        # yaw from top PCA
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
        hx = hy = hz = float(default_half_extent)
        yaw = _yaw_from_quaternion(d.get("cube_world_orientation",[1,0,0,0]))
    return cx, cy, cz, hx, hy, hz, yaw

def _rects_from_body(frame: Dict[str,Any],
                     key: str,
                     default_half_extent: float = 0.03,
                     xy_pad: float = 0.0) -> List[Tuple[float,float,float,float,float,float,float]]:
    """
    Return a list of rectangles for the body 'key'.
    Each rectangle is (cx, cy, cz, hx, hy, hz, yaw).
    Supports:
      - Cube-like: 'cube_world_position' + 'cube_world_orientation' (+ size/half_extents/pc).
      - T-shape  : 'vbar_*' and 'hbar_*' with '..._world_position/orientation/scale'.
    """
    d = frame.get("data", {}).get(key, {})
    rects = []

    # T-shape: vbar + hbar (return in deterministic order: v, h)
    has_v = all(f in d for f in ("vbar_world_position","vbar_world_orientation","vbar_world_scale"))
    has_h = all(f in d for f in ("hbar_world_position","hbar_world_orientation","hbar_world_scale"))
    if has_v and has_h:
        for prefix in ("vbar","hbar"):
            pos = np.asarray(d[f"{prefix}_world_position"], dtype=float)
            ori = d[f"{prefix}_world_orientation"]
            scl = np.asarray(d[f"{prefix}_world_scale"], dtype=float)  # full extents
            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
            hx, hy, hz = float(scl[0])*0.5, float(scl[1])*0.5, float(scl[2])*0.5
            hx += xy_pad; hy += xy_pad  # apply padding
            yaw = _yaw_from_quaternion(ori)
            rects.append((cx, cy, cz, hx, hy, hz, yaw))
        return rects

    # Cube-like (old)
    if "cube_world_position" in d:
        pos = np.asarray(d.get("cube_world_position",[0,0,0]), dtype=float)
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

        # Half-extents
        if "half_extents" in d:
            he = np.asarray(d["half_extents"], dtype=float).reshape(3)
            hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
        elif "half_extent" in d:
            he = np.asarray(d["half_extent"], dtype=float).reshape(3)
            hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
        elif "size" in d:
            he = (np.asarray(d["size"], dtype=float).reshape(3))/2.0
            hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
        elif "pc" in d:
            cx, cy, cz, hx, hy, hz, _yaw = _dims_from_pc(d, default_half_extent)
            yaw = _yaw_from_quaternion(d.get("cube_world_orientation",[1,0,0,0])) if "cube_world_orientation" in d else _yaw
            hx += xy_pad; hy += xy_pad
            rects.append((cx, cy, cz, hx, hy, hz, yaw))
            return rects
        else:
            hx = hy = hz = float(default_half_extent)

        yaw = _yaw_from_quaternion(d.get("cube_world_orientation",[1,0,0,0]))
        hx += xy_pad; hy += xy_pad
        rects.append((cx, cy, cz, hx, hy, hz, yaw))
        return rects

    # Fallback via 'pc' (rare; old format without explicit pose)
    if "pc" in d and isinstance(d["pc"], list) and len(d["pc"]) >= 8:
        cx, cy, cz, hx, hy, hz, yaw = _dims_from_pc(d, default_half_extent)
        hx += xy_pad; hy += xy_pad
        rects.append((cx, cy, cz, hx, hy, hz, yaw))
        return rects

    raise ValueError(f"Body '{key}' in frame lacks required fields (cube_* or *bar_* or pc).")

# ============================
# XY-overlap with Z tolerance for unions
# ============================

def _union_xy_area_with_z_gate(rects_a, rects_b, z_eps: float, z_shift_b: float = 0.0) -> Tuple[float,float]:
    """
    rects_a / rects_b: lists of (cx, cy, cz, hx, hy, hz, yaw).
    Returns (area_all_pairs, area_z_gated)
      - area_all_pairs: union area of all pairwise intersections (no Z gating).
      - area_z_gated  : same but only counting pairs with Z separation <= z_eps.
    """
    # Build rectangles (polygons) + z slabs
    A = []
    for (cx,cy,cz,hx,hy,hz,yaw) in rects_a:
        poly = _rect_xy_corners(cx, cy, hx, hy, yaw)
        A.append((poly, cz, hz))
    B = []
    for (cx,cy,cz,hx,hy,hz,yaw) in rects_b:
        poly = _rect_xy_corners(cx, cy, hx, hy, yaw)
        B.append((poly, cz + z_shift_b, hz))

    # All pairwise intersections as polygons
    inter_all = []
    inter_pass = []
    for (pa, cza, hza) in A:
        for (pb, czb, hzb) in B:
            P = _poly_intersection(pa, pb)
            if P.size == 0:
                continue
            inter_all.append(P)
            sep = _z_separation(cza, hza, czb, hzb)
            if sep <= z_eps:
                inter_pass.append(P)

    return _area_union(inter_all), _area_union(inter_pass)

def _xy_area_of_union(rects) -> float:
    """Area of union of 1..N rectangles (same-body bars), no Z gating."""
    polys = [_rect_xy_corners(cx,cy,hx,hy,yaw) for (cx,cy,cz,hx,hy,hz,yaw) in rects]
    # union of convex polygons via inclusion–exclusion (N<=2 for T)
    return _area_union(polys)

def _z_overlap_length(cz1,hz1, cz2,hz2) -> float:
    lo = max(cz1 - hz1, cz2 - hz2)
    hi = min(cz1 + hz1, cz2 + hz2)
    return max(0.0, hi - lo)

def _internal_overlap_volume(rects) -> float:
    """
    3D overlap between bars within the SAME body (for T this is vbar∩hbar).
    We do: volume = (XY intersection area of those two bars) * (Z overlap length).
    If only 1 rect, returns 0.
    """
    if len(rects) < 2:
        return 0.0
    (cx1,cy1,cz1,hx1,hy1,hz1,yaw1) = rects[0]
    (cx2,cy2,cz2,hx2,hy2,hz2,yaw2) = rects[1]
    P1 = _rect_xy_corners(cx1,cy1,hx1,hy1,yaw1)
    P2 = _rect_xy_corners(cx2,cy2,hx2,hy2,yaw2)
    P = _poly_intersection(P1,P2)
    area_xy = _poly_area(P)
    z_len = _z_overlap_length(cz1,hz1, cz2,hz2)
    return float(area_xy * z_len)

def _volume_of_body(rects) -> float:
    """Sum of bar volumes minus internal overlap."""
    vols = [float((2*hx)*(2*hy)*(2*hz)) for (cx,cy,cz,hx,hy,hz,yaw) in rects]
    vol = sum(vols)
    vol -= _internal_overlap_volume(rects)
    return float(max(vol, 0.0))

# ============================
# SINGLE strict argmax (one maximum only)
# ============================

def _unique_argmax(values: List[float], times: List[float], steps: List[int]) -> int:
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
                   tcube_dz: float = 0.0,
                   xy_pad: float = 0.0) -> Dict[str, Any]:
    """
    Compute per-frame XY overlap (with Z tolerance) for two bodies that can each
    be a union of 1..2 bars.
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

            rects_a = _rects_from_body(frame, cube_key, default_half_extent, xy_pad)
            rects_b = _rects_from_body(frame, tcube_key, default_half_extent, xy_pad)

            # XY intersection of unions
            area_all, area_pass = _union_xy_area_with_z_gate(rects_a, rects_b, z_eps, z_shift_b=float(tcube_dz))

            # Areas / volumes for reporting & normalization
            a_xy = _xy_area_of_union(rects_a)
            vol_a = _volume_of_body(rects_a)
            vol_b = _volume_of_body([(x,y,z+float(tcube_dz),hx,hy,hz,yaw) for (x,y,z,hx,hy,hz,yaw) in rects_b])

            norm_by_a_xy = (area_pass / a_xy) if a_xy > 0 else 0.0

            # Z separation summary: minimum pairwise separation (diagnostic)
            min_sep = float("inf")
            for (cxa,cya,cza,hxa,hya,hza,ya) in rects_a:
                for (cxb,cyb,czb,hxb,hyb,hzb,yb) in rects_b:
                    sep = _z_separation(cza, hza, czb + float(tcube_dz), hzb)
                    if sep < min_sep: min_sep = sep
            if min_sep == float("inf"):
                min_sep = 0.0

            results.append({
                "index": idx,
                "time_step": ts,
                "time": t,
                "area_xy": float(area_all),
                "area_xy_pass": float(area_pass),
                "z_separation": float(min_sep),
                "cube_area_xy": float(a_xy),         # 'cube' kept for backward-compatible field name
                "tcube_area_xy": None,               # not used; union metric normalizes by first body
                "cube_volume": float(vol_a),
                "tcube_volume": float(vol_b),
                "area_norm_by_cube_xy": float(norm_by_a_xy),
                "tcube_dz": float(tcube_dz),
                "z_eps": float(z_eps),
            })
        except Exception:
            continue

    if not results:
        return {
            "max_area_xy": 0.0,
            "max_frame": None,
            "max_area_norm_by_cube_xy": 0.0,
            "max_frame_norm_by_cube_xy": None,
            "per_frame": []
        }

    raw_vals  = [r["area_xy_pass"] for r in results]
    norm_vals = [r["area_norm_by_cube_xy"] for r in results]
    times     = [r["time"] for r in results]
    steps     = [r["time_step"] for r in results]

    i_raw  = _unique_argmax(raw_vals,  times, steps)
    i_norm = _unique_argmax(norm_vals, times, steps)

    max_raw  = raw_vals[i_raw]
    max_norm = norm_vals[i_norm]

    max_frame_raw = {
        "time_step": results[i_raw]["time_step"],
        "time": results[i_raw]["time"],
        "value": results[i_raw]["area_xy_pass"],
        "cube_volume": results[i_raw]["cube_volume"],
        "tcube_volume": results[i_raw]["tcube_volume"],
        "z_separation": results[i_raw]["z_separation"],
    }
    max_frame_norm = {
        "time_step": results[i_norm]["time_step"],
        "time": results[i_norm]["time"],
        "value": results[i_norm]["area_norm_by_cube_xy"],
        "cube_volume": results[i_norm]["cube_volume"],
        "tcube_volume": results[i_norm]["tcube_volume"],
        "z_separation": results[i_norm]["z_separation"],
    }

    return {
        "max_area_xy": max_raw,
        "max_frame": max_frame_raw,
        "max_area_norm_by_cube_xy": max_norm,
        "max_frame_norm_by_cube_xy": max_frame_norm,
        "per_frame": results
    }

# ============================
# Folder utilities (unchanged API)
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
                            metric: str = "area",
                            xy_pad: float = 0.0) -> Dict[str, Any]:
    load_res = load_all_jsons_from_folder(folder, pattern, recursive, cube_key=cube_key, tcube_key=tcube_key)
    per_file = []
    overall_max = 0.0
    occurrence = None

    key_val   = "max_area_xy" if metric == "area" else "max_area_norm_by_cube_xy"
    key_frame = "max_frame"   if metric == "area" else "max_frame_norm_by_cube_xy"

    for ds in load_res["datasets"]:
        summary = analyze_frames({"Isaac Sim Data": ds["frames"]},
                                 cube_key=cube_key, tcube_key=tcube_key,
                                 z_eps=z_eps, default_half_extent=default_half_extent,
                                 tcube_dz=tcube_dz, xy_pad=xy_pad)
        per_file.append({"path": ds["path"], **summary})
        m = summary[key_val]
        f = summary[key_frame]
        if f is None:
            continue
        if m > overall_max:
            overall_max = m
            occurrence = (ds["path"], f["time_step"], f["time"], f["value"])
        elif math.isclose(m, overall_max, rel_tol=1e-12, abs_tol=0.0) and occurrence is not None:
            cand = (ds["path"], f["time_step"], f["time"], f["value"])
            occurrence = min([occurrence, cand], key=lambda x: (x[2], x[1]))

    return {
        "overall_max": overall_max,
        "metric": metric,
        "occurrence": occurrence,
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
                          metric: str = "area",
                          xy_pad: float = 0.0):
    res = max_overlap_from_folder(folder, pattern, recursive,
                                  z_eps=z_eps, default_half_extent=default_half_extent, tcube_dz=tcube_dz,
                                  cube_key=cube_key, tcube_key=tcube_key, metric=metric, xy_pad=xy_pad)
    return res["per_file"], res["errors"], res["num_files"]

# ============================
# Debug helpers
# ============================

def _pairwise_breakdown(rects_a, rects_b, z_eps: float, z_shift_b: float = 0.0):
    """
    Returns list of dicts: {pair, area, area_pass, z_sep}
    Where pair ∈ {v∩v, v∩h, h∩v, h∩h, box∩box, ...}
    """
    labels_a = ["v","h"] if len(rects_a) == 2 else ["box"]
    labels_b = ["v","h"] if len(rects_b) == 2 else ["box"]
    out = []
    for i,(cxa,cya,cza,hxa,hya,hza,ya) in enumerate(rects_a):
        Pa = _rect_xy_corners(cxa,cya,hxa,hya,ya)
        for j,(cxb,cyb,czb,hxb,hyb,hzb,yb) in enumerate(rects_b):
            Pb = _rect_xy_corners(cxb,cyb,hxb,hyb,yb)
            P  = _poly_intersection(Pa,Pb)
            area = _poly_area(P)
            sep = _z_separation(cza,hza, czb+z_shift_b, hzb)
            out.append({
                "pair": f"{labels_a[i]}∩{labels_b[j]}",
                "area": float(area),
                "area_pass": float(area if sep <= z_eps else 0.0),
                "z_sep": float(sep),
            })
    return out

def _debug_print_top_pairs(frames, cube_key, tcube_key, default_half_extent,
                           z_eps, tcube_dz, xy_pad, top_k):
    # Rank by non-gated union area
    scored = []
    cache_rects = []
    for idx, frame in enumerate(frames):
        try:
            rects_a = _rects_from_body(frame, cube_key, default_half_extent, xy_pad)
            rects_b = _rects_from_body(frame, tcube_key, default_half_extent, xy_pad)
        except Exception:
            continue
        area_all, _ = _union_xy_area_with_z_gate(rects_a, rects_b, z_eps, z_shift_b=float(tcube_dz))
        scored.append((area_all, idx))
        cache_rects.append((idx, rects_a, rects_b))
    scored.sort(key=lambda x: x[0], reverse=True)
    print("\n=== Debug: Pairwise overlaps on top frames (by non-gated union area) ===")
    for rank, (score, idx) in enumerate(scored[:top_k], start=1):
        # Find cached rects
        rects_a = rects_b = None
        for i, ra, rb in cache_rects:
            if i == idx:
                rects_a, rects_b = ra, rb
                break
        f = frames[idx]
        ts = f.get("current_time_step")
        t  = f.get("current_time")
        # min z sep (diagnostic)
        min_sep = float("inf")
        for (cxa,cya,cza,hxa,hya,hza,ya) in rects_a:
            for (cxb,cyb,czb,hxb,hyb,hzb,yb) in rects_b:
                sep = _z_separation(cza, hza, czb + float(tcube_dz), hzb)
                if sep < min_sep: min_sep = sep
        if min_sep == float("inf"):
            min_sep = 0.0
        # pairwise
        pairs = _pairwise_breakdown(rects_a, rects_b, z_eps, z_shift_b=float(tcube_dz))
        print(f"\n[#{rank}] step={ts}, time={t:.6f}, non-gated union area={score:.6g}, min Z sep={min_sep:.6g}")
        for p in pairs:
            print(f"  {p['pair']}: area={p['area']:.6g}, gated_area={p['area_pass']:.6g}, z_sep={p['z_sep']:.6g}")

# ============================
# CLI printing
# ============================

def _print_one_max(prefix: str, frame: Dict[str, Any]) -> None:
    print(prefix + f"at step={frame['time_step']}, time={frame['time']:.6f}, value={frame['value']}")
    print(f"{prefix}First body volume : {frame['cube_volume']}")
    print(f"{prefix}Second body volume: {frame['tcube_volume']}")
    print(f"{prefix}Z separation (m): {frame['z_separation']}")

def _print_file_summary(path: str, summary: Dict[str, Any], metric: str = "area") -> None:
    print(f"\nFile: {path}")
    if metric == "area":
        print(f"  Max XY-overlap area (Z≤eps): {summary['max_area_xy']}")
        if summary["max_frame"]:
            _print_one_max("    ", summary["max_frame"])
    else:
        print(f"  Max XY-overlap (normalized by first body's XY area): {summary['max_area_norm_by_cube_xy']}")
        if summary["max_frame_norm_by_cube_xy"]:
            _print_one_max("    ", summary["max_frame_norm_by_cube_xy"])

def _print_folder_summary(res: Dict[str, Any]) -> None:
    label = "XY area (Z≤eps)" if res.get("metric","area")=="area" else "XY area / First-body XY"
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
    label = "XY area (Z≤eps)" if metric == "area" else "XY area / First-body XY"

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
    parser = argparse.ArgumentParser(description="Max XY-overlap with Z tolerance (single maximum), Z-rotation aware. Supports Cube/TCube and T/Target_T.")
    parser.add_argument("path", help="Path to a JSON file or a folder containing JSON files.")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern for JSON files in folder (default: *.json)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subfolders")
    parser.add_argument("--half", type=float, default=0.03, help="Default half-extent (m) if not present in JSON")
    parser.add_argument("--tcube-dz", type=float, default=0.0,
                        help="Translate the SECOND body along +Z by this many meters (positive=up, negative=down).")
    parser.add_argument("--z-eps", type=float, default=1e-4,
                        help="Z-axis tolerance (meters) to accept XY overlap when slabs are separated.")
    parser.add_argument("--cube-key", type=str, default="Cube",
                        help="Name of the FIRST body inside frame['data'] (e.g., 'Cube' or 'T').")
    parser.add_argument("--tcube-key", type=str, default="TCube",
                        help="Name of the SECOND body inside frame['data'] (e.g., 'TCube' or 'Target_T').")
    parser.add_argument("--metric", choices=["area", "norm-cube-xy"], default="area",
                        help="Max metric: 'area'=raw XY area, 'norm-cube-xy'=XY area / first body's XY area.")
    parser.add_argument("--per-file-only", action="store_true",
                        help="Print only each JSON's max (suppress overall summary).")
    parser.add_argument("--xy-pad", type=float, default=0.0,
                        help="Pad each rectangle half-extent in X and Y by this many meters (default 0).")
    parser.add_argument("--debug-pairs", type=int, default=0,
                        help="For a single file: print per-pair overlaps on top-N frames (ranked by non-gated union area).")
    args = parser.parse_args()

    p = Path(args.path)
    recursive = not args.no_recursive

    if p.is_file():
        try:
            frames = load_isaac_frames(p, cube_key=args.cube_key, tcube_key=args.tcube_key)
            summary = analyze_frames({"Isaac Sim Data": frames},
                                     cube_key=args.cube_key, tcube_key=args.tcube_key,
                                     z_eps=args.z_eps, default_half_extent=args.half,
                                     tcube_dz=args.tcube_dz, xy_pad=args.xy_pad)
            _print_file_summary(str(p), summary, metric=args.metric)

            if args.debug_pairs and frames:
                _debug_print_top_pairs(frames, args.cube_key, args.tcube_key, args.half,
                                       args.z_eps, args.tcube_dz, args.xy_pad, args.debug_pairs)
        except Exception as e:
            print(f"Error processing file {p}: {e}", file=sys.stderr)
            sys.exit(1)
    elif p.is_dir():
        try:
            res = max_overlap_from_folder(
                p, pattern=args.pattern, recursive=recursive,
                z_eps=args.z_eps, default_half_extent=args.half, tcube_dz=args.tcube_dz,
                cube_key=args.cube_key, tcube_key=args.tcube_key, metric=args.metric, xy_pad=args.xy_pad
            )
            if args.per_file_only:
                _print_per_file_only(res)
            else:
                _print_folder_summary(res)
            if args.debug_pairs:
                print("\n(Note) --debug-pairs is only supported in single-file mode to avoid excessive output.")
        except Exception as e:
            print(f"Error processing folder {p}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Path not found: {p}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

# python push.py /home/workstation/project/franka_rope_equibot/logs/eval/tmp/2025-09-07_10-43-20.json --cube-key T --tcube-key Target_T   --metric norm-cube-xy  --z-eps 0.005
