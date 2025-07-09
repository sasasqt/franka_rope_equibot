import json
import numpy as np
import os
from math import sqrt
import math
import hydra
import random
from scipy.spatial.transform import Rotation as R
from collections import defaultdict


# len(data) = 1
# data[0].keys()=Isaac Sim Data
# len(data[0]['Isaac Sim Data']) = saved steps
# data[0]['Isaac Sim Data'] = list
# data[0]['Isaac Sim Data'][i].keys() = ['futent_time', 'futent_time_step', 'data']
# data[0]['Isaac Sim Data'][i]['data'].keys()=dict_keys(['Left', 'Right', 'Rope', 'extras', 'Datetime'])
# data[0]['Isaac Sim Data'][i]['data']['Left']=dict_keys(['Left_joint_positions', 'applied_joint_positions', 'Left_end_effector_world_position',
#   'Left_end_effector_world_orientation', 'Left_end_effector_local_position', 'Left_end_effector_local_orientation', 'Left_target_world_position',
#   'Left_target_world_orientation', 'Left_target_local_position', 'Left_target_local_orientation'])
# data[0]['Isaac Sim Data'][i]['data']['Rope']=dict_keys(['Rope_world_position', 'Rope_world_orientation'])
# data[0]['Isaac Sim Data'][i]['data']['extras']=dict_keys(['left_reposition_pressed', 'right_reposition_pressed'])
# recording={}
# recording["pc"]=[]
# recording["action"]=[]
# recording["eef_pos"]=[]



@hydra.main(config_path="equibot/policies/configs", config_name="franka_base")
def main(cfg):

    gt_dir = cfg.eval_chart.gt_dir
    # pred_file = cfg.eval_chart.pred_file
    pred_dir=cfg.eval_chart.pred_dir
    # output_dir = cfg.eval_chart.output_dir

    rotation = cfg.eval_chart.rotation or None
    translation = cfg.eval_chart.translation or None

    if rotation:
        rotation = R.from_euler('z', rotation[2], degrees=True).as_matrix()
    else:
        rotation=np.eye(3)
    if translation:
        translation=np.array(translation)
    else:
        translation=np.array([0.0,0.0,0.0,])

    # if not os.path.isfile(pred_file):
    #     return
    gt_gripper_world_poss=[]
    gt_gripper_world_oris=[]
    for ep, filename in enumerate(os.listdir(gt_dir)):
        if not filename.endswith(".json"):
            continue

        file = os.path.join(gt_dir, filename)

        data = []
        with open(file, "rb") as f:
            for line in f:
                data.append(json.loads(line))
                
        # to mimic saved npz with keys pc, rgb?, action, eef_pos
        _gt_gripper_world_poss=[]
        _gt_gripper_world_oris=[]
        for i, _fut in enumerate(data[0]["Isaac Sim Data"]):
            gripper_world_pos=data[0]["Isaac Sim Data"][i]["data"]["Left"]["Left_target_world_position"]
            gripper_world_ori=data[0]["Isaac Sim Data"][i]["data"]["Left"]["Left_target_world_orientation"] # quant
        
            _gt_gripper_world_poss.append(gripper_world_pos)
            _gt_gripper_world_oris.append(gripper_world_ori)
            # print(gripper_world_pos)
        gt_gripper_world_poss.append(_gt_gripper_world_poss)
        gt_gripper_world_oris.append(_gt_gripper_world_oris)
    results=defaultdict(dict)
    for ep, filename in enumerate(os.listdir(pred_dir)):
        if not (filename.endswith(".json") and filename.startswith("my")):
            continue

        print(filename)
        pred_file = os.path.join(pred_dir, filename)

        data = []
        with open(pred_file, "r") as f:
            for line in f:
                data.append(json.loads(line))

        gripper_world_poss=[]
        gripper_world_oris=[]
            
        for i, _fut in enumerate(data[0]):
            # print(i)
            if i>110:
                break
            # gripper_world_pos=data[0][str(i)]["abs_t"]
            # gripper_world_ori=data[0][str(i)]["abs_q"] # quant
            # # gripper_world_pos=np.array(gripper_world_pos)+translation

            # # gripper_world_ori=rotation@R.from_quat(np.array(gripper_world_ori),scalar_first=True).as_matrix()
            # # gripper_world_ori=R.from_matrix(gripper_world_ori).as_quat(scalar_first=True)
            # gripper_world_pos=np.array(gripper_world_pos)-translation
            # gripper_world_ori= np.linalg.inv(rotation)@R.from_quat(np.array(gripper_world_ori),scalar_first=True).as_matrix()
            # gripper_world_ori=R.from_matrix(gripper_world_ori).as_quat(scalar_first=True)
            # gripper_world_poss.append(gripper_world_pos)
            # gripper_world_oris.append(gripper_world_ori)

            gripper_world_pos=data[0][str(i)]["undo_abs_t"]
            gripper_world_ori=data[0][str(i)]["undo_abs_q"] # quant

            gripper_world_poss.append(gripper_world_pos)
            gripper_world_oris.append(gripper_world_ori)

            # print(gripper_world_pos)
            # break


                
        res1,res2=min_frechet_rotvec(gripper_world_poss,gripper_world_oris,gt_gripper_world_poss,gt_gripper_world_oris,)
        # result=emd_distance(np.asarray(gripper_world_poss), np.asarray(gt_gripper_world_poss))
        # print("Wasserstein‑1 / EMD distance (different sizes):", result)
        # print(res2)

        results[len(results)]=res2
    total = sum(val for val in results.values())
    results['avg']=total/len(results)
    print(f">>> FINAL: {results['avg']} <<<")
    with open(os.path.join(pred_dir, f'_lcs_union_scores.json'), 'w') as f:
        json.dump(results,f)

from math import acos


def _quat_geodesic(q1, q2):
    """Shortest-arc angle (rad) between two unit quaternions."""
    return 2.0 * acos(np.clip(abs(np.sum(np.dot(q1, q2))), -1.0, 1.0))

# ----------------------------------------------------------------------
# Iterative Fréchet in SE(3) with rot-vectors
# ----------------------------------------------------------------------

import numpy as np

def frechet_avg_se3_rotvec(P_pos, P_ori,
                           Q_pos, Q_ori,
                           *, lambda_rot=1.0):
    """
    Average (integral / L¹) discrete Fréchet distance for 6-DoF trajectories.

    Each pose is split into
      P_pos, Q_pos : (N,3) and (M,3)   Cartesian metres
      P_ori, Q_ori : (N,4) and (M,4)   unit quaternions  (w,x,y,z)

    Returns
    -------
    float
        Mean SE(3) separation along the optimal monotone coupling path.

    Complexity
    ----------
    Time  : O(N·M)
    Memory: O(N·M)  (can be reduced to O(min{N,M}) if needed)

    Notes
    -----
    • The classical max-Fréchet recurrence “max (min, min, min, d)” is replaced
      with the additive DTW-style recurrence
          ca[i,j] = d[i,j] + min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1]).
    • The final sum is divided by the path length (N+M-1) to obtain a true mean.
    """

    # --- pair-wise SE(3) distances -----------------------------------
    N, M = len(P_pos), len(Q_pos)
    dist = np.empty((N, M), dtype=P_pos.dtype)

    for i in range(N):
        for j in range(M):
            dp  = P_pos[i] - Q_pos[j]
            ang = _quat_geodesic(P_ori[i], Q_ori[j])     # rad
            dist[i, j] = np.sqrt(np.dot(dp, dp) + (lambda_rot * ang) ** 2)
    # -----------------------------------------------------------------

    # dynamic-programming table: cumulative *sums*
    ca = np.empty((N, M), dtype=P_pos.dtype)
    ca[0, 0] = dist[0, 0]

    # initialise first row / column
    for j in range(1, M):
        ca[0, j] = dist[0, j] + ca[0, j - 1]
    for i in range(1, N):
        ca[i, 0] = dist[i, 0] + ca[i - 1, 0]

    # DP fill (add + min, not max + min)
    for i in range(1, N):
        for j in range(1, M):
            ca[i, j] = dist[i, j] + min(
                ca[i - 1, j],
                ca[i - 1, j - 1],
                ca[i, j - 1]
            )

    # divide by path length to get the mean
    path_len = N + M - 1         # number of grid cells visited
    return float(ca[N - 1, M - 1] / path_len)


def frechet_distance_se3_rotvec(P_pos,P_ori, Q_pos,Q_ori, *, lambda_rot=1.0,separate=False):
    """
    Discrete Fréchet distance for 6-DoF poses stored as
    [x, y, z, rx, ry, rz] (metres, axis–angle radians).

    No recursion: O(N·M) time, O(N·M) memory.
    """
    P_pos=P_pos
    Q_pos=Q_pos
    
    P_q = P_ori          # (N,4)
    Q_q = Q_ori          # (M,4)

    n, m = len(P_pos), len(Q_pos)
    ca = np.empty((n, m), dtype=P_pos.dtype)   # dynamic-programming table

    # --- pre-compute pairwise SE(3) point distances ------------------
    dist = np.empty((n, m), dtype=P_pos.dtype)
    for i in range(n):
        for j in range(m):
            dp   = P_pos[i] - Q_pos[j]
            ang  = _quat_geodesic(P_q[i], Q_q[j])
            dist[i, j] = np.sqrt(np.dot(dp, dp) + (lambda_rot * ang) ** 2)
    # ----------------------------------------------------------------

    # DP initialisation
    ca[0, 0] = dist[0, 0]
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])

    # DP fill
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                dist[i, j]
            )

    return float(ca[n - 1, m - 1])

from tqdm.auto import tqdm

def dtw_distance_se3_rotvec(P_pos, P_ori,
                            Q_pos, Q_ori,
                            *, lambda_rot=1.0,
                            average=True):
    """
    Dynamic-Time-Warping (DTW) distance for 6-DoF trajectories whose poses are
    stored as
        P_pos, Q_pos : (N,3) and (M,3)   --  metres
        P_ori, Q_ori : (N,4) and (M,4)   --  unit quaternions  (w, x, y, z)

    Parameters
    ----------
    lambda_rot : float
        Weight that converts radians of rotational error into translational
        metres before the Euclidean √(dx²+…+λ²·dθ²) combination.
    average : bool, default False
        • False → return *total* DTW cost (classic definition).
        • True  → return *mean* cost along the optimal warping path
                   (useful to compare sequences of very different duration).

    Returns
    -------
    float
        DTW total or mean cost.

    Complexity
    ----------
    Time  :  O(N·M)
    Memory:  O(N·M)  (drop to O(min(N,M)) with a rolling-array trick).
    """

    N= len(P_pos)
    M=100#, len(Q_pos)

    # ------------------------------------------------------------
    # Pair-wise SE(3) distance matrix: √(‖Δp‖² + (λ·Δθ)²)
    # ------------------------------------------------------------
    dist = np.empty((N, M), dtype=P_pos.dtype)
    for i in range(N):
        for j in range(M):
            dp  = P_pos[i] - Q_pos[j]
            ang = _quat_geodesic(P_ori[i], Q_ori[j])      # radians
            dist[i, j] = np.sqrt(np.dot(dp, dp) + (lambda_rot * ang) ** 2)

    # ------------------------------------------------------------
    # Dynamic-programming table – cumulative *sums*
    # ------------------------------------------------------------
    ca = np.empty((N, M), dtype=P_pos.dtype)
    ca[0, 0] = dist[0, 0]

    for j in range(1, M):
        ca[0, j] = dist[0, j] + ca[0, j - 1]          # can “stay” on P[0]
    for i in range(1, N):
        ca[i, 0] = dist[i, 0] + ca[i - 1, 0]          # can “stay” on Q[0]

    for i in range(1, N):
        for j in range(1, M):
            ca[i, j] = dist[i, j] + min(
                ca[i - 1, j],        # vertical   (repeat Q_j)
                ca[i, j - 1],        # horizontal (repeat P_i)
                ca[i - 1, j - 1]     # diagonal   (advance both)
            )

    total_cost = float(ca[N - 1, M - 1])
    if not average:
        return total_cost

    # ------------------------------------------------------------
    # Recover mean cost = total / path-length
    # We track a second DP table with shortest path lengths so we
    # don’t need an explicit back-trace.
    # ------------------------------------------------------------
    steps = np.empty((N, M), dtype=np.int32)
    steps[0, 0] = 1
    for j in range(1, M):
        steps[0, j] = steps[0, j - 1] + 1
    for i in range(1, N):
        steps[i, 0] = steps[i - 1, 0] + 1
    for i in range(1, N):
        for j in range(1, M):
            prev = np.argmin([ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]])
            if prev == 0:
                steps[i, j] = steps[i - 1, j] + 1
            elif prev == 1:
                steps[i, j] = steps[i, j - 1] + 1
            else:
                steps[i, j] = steps[i - 1, j - 1] + 1

    return total_cost / float(steps[N - 1, M - 1])

def rmse(pos,q, pos_refs,q_refs,lambda_rot=0.0):
    """
    pos, pos_ref : (N, D) ndarray
        Sampled positions at identical time stamps.
    Returns
    -------
    float           scalar RMSE over all axes and time steps
    ndarray shape (D,)   per-axis RMSE (optional convenience)
    """
    min=float("inf")
    for pos_ref in pos_refs:
        err = pos - pos_ref                    # (N, D)
        total    = np.sqrt(np.mean(np.sum(err**2, 1)))
        if min>total:
            min=total
    return min

def min_frechet_rotvec(P_pos,P_ori, Q_poss,Q_oris, lambda_rot=0.1):
    """
    Oracle (min) Fréchet over K stochastic samples for xyz + rot-vec input.
    """
    # Convert GT once
    P_pos_arr = np.asarray(P_pos, dtype=float)
    P_ori_arr = np.asarray(P_ori, dtype=float)

    best1 = float("inf")
    best2 = 0.0

    it = zip(Q_poss, Q_oris)
    pbar = tqdm(it,
                total=len(Q_poss),        # tqdm can now display % done
                desc="min-Fréchet")
    
    for Q_pos, Q_ori in it: 
        # d = dtw_distance_se3_rotvec(
        #         P_pos_arr,
        #         P_ori_arr,
        #         np.asarray(Q_pos, dtype=float),
        #         np.asarray(Q_ori, dtype=float),
        #         lambda_rot=0.1
        #     )
        # if d < best1:
        #     best1 = d

        d=multi_lcs_union_length(P_pos_arr[:100],np.asarray(Q_pos, dtype=float),P_ori_arr[:100],np.asarray(Q_ori, dtype=float),)
        if d > best2:
            best2 = d

        pbar.update(1)
    return best1,best2



import numpy as np
from scipy.optimize import linprog

def emd_distance(P: np.ndarray,
                 Q: np.ndarray,
                 a: np.ndarray | None = None,
                 b: np.ndarray | None = None) -> float:
    """
    Earth‑Mover / Wasserstein‑1 distance for two point‑clouds **of
    different cardinalities**.  Solves the full optimal‑transport
    linear program with SciPy’s HiGHS backend.

    Parameters
    ----------
    P, Q : (n, d) and (m, d) ndarrays
        Coordinates of the two clouds.
    a, b : 1‑D arrays of shape (n,) and (m,), optional
        Non‑negative weights (masses) on P and Q.  If omitted
        they default to uniform and are automatically normalised.

    Returns
    -------
    float
        The Wasserstein‑1 (average ground) distance.
    """
    n, d = P.shape
    m, d2 = Q.shape
    if d != d2:
        raise ValueError("Point dimensionality mismatch")

    # Default to uniform weights and make sure both sum to 1.
    if a is None:
        a = np.ones(n) / n
    a = np.asarray(a, dtype=float)
    a /= a.sum()

    if b is None:
        b = np.ones(m) / m
    b = np.asarray(b, dtype=float)
    b /= b.sum()

    # Pair‑wise ground cost ‖p_i − q_j‖₂
    C = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    c_vec = C.ravel()

    # Build equality‑constraint matrix  [ T 1_m = a ; Tᵀ 1_n = b ]
    A_eq = np.zeros((n + m, n * m))
    b_eq = np.concatenate([a, b])

    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1          # row‑sum constraints
    for j in range(m):
        A_eq[n + j, j::m] = 1                   # col‑sum constraints

    bounds = [(0, None)] * (n * m)
    res = linprog(c_vec, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError("OT solver failed: " + res.message)

    T = res.x.reshape(n, m)
    return float((T * C).sum())


import numpy as np

def lcss(A, B,P,Q, peps=0.1, qeps=np.deg2rad(100), delta=np.inf):
    """
    A, B : (m, d) and (n, d) arrays
    eps  : spatial tolerance
    delta: max time-index gap (int) or np.inf
    Returns similarity in [0,1].
    """
    # m, n = len(A), len(B)
    m=len(A)
    n=110

    L = np.zeros((m+1, n+1), dtype=int)
    for i in range(1, m+1):
        for j in range(max(1, i-delta), min(n, i+delta)+1):
            ang_err = _quat_geodesic(P[i-1], Q[j-1])
            pos_err = np.linalg.norm(A[i-1] - B[j-1])

            if pos_err <= peps and ang_err<=qeps:
                L[i, j] = 1 + L[i-1, j-1]
            else:
                L[i, j] = max(L[i-1, j], L[i, j-1])
    return L[m, n] / min(m, n)          # normalised score



import numpy as np

def multi_lcs_union_length(
    A: np.ndarray,
    B: np.ndarray,
    P,Q,
    eps_pos: float = 0.1,
    eps_ori: float = np.deg2rad(10),
    delta: float | int = 50,#np.inf,        # timing slack in sample indices
    min_run: int = 3,                   # ignore matches shorter than this
) -> int:
    """Return the union‑length of *all* disjoint longest common subsequences.

    Each element of *A* and *B* is allowed to appear in **at most one** of the
    extracted subsequences.  The algorithm iteratively removes one LCS from
    the remaining slices until no matches are left and sums their lengths.
    """

    # ------------------------------------------------------------------
    # inline pose‑match predicate (fast; no extra function look‑ups)
    # ------------------------------------------------------------------
    def _match(i: int, j: int) -> bool:
        if np.linalg.norm(A[i] - B[j]) > eps_pos:
            return False

        dot = abs(np.dot(P[i], Q[j]))
        ang = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
        return ang <= eps_ori



    # effective DP band width from timing slack
    band_global = int(delta) if np.isfinite(delta) else max(len(A), len(B))

    # ------------------------------------------------------------------
    # inner function: compute ONE LCS over A[a0:a1] × B[b0:b1]
    # returns list[(ia, jb)] with absolute indices.
    # ------------------------------------------------------------------
    def _one_lcs(a0: int, a1: int, b0: int, b1: int):
        if a0 >= a1 or b0 >= b1:
            return []
        m, n = a1 - a0, b1 - b0
        band = min(band_global, max(m, n))
        dp = np.zeros((m + 1, n + 1), dtype=np.uint32)

        # fill DP within band
        for ii in range(m - 1, -1, -1):
            j_max = min(n - 1, ii + band)
            j_min = max(0, ii - band)
            for jj in range(j_max, j_min - 1, -1):
                if _match(a0 + ii, b0 + jj):
                    dp[ii, jj] = 1 + dp[ii + 1, jj + 1]
                else:
                    dp[ii, jj] = max(dp[ii + 1, jj], dp[ii, jj + 1])

        # back‑trace ONE path
        ia = ja = 0
        pairs = []
        while ia < m and ja < n:
            if _match(a0 + ia, b0 + ja) and dp[ia, ja] == dp[ia + 1, ja + 1] + 1:
                pairs.append((a0 + ia, b0 + ja))
                ia += 1
                ja += 1
            elif dp[ia + 1, ja] >= dp[ia, ja + 1]:
                ia += 1
            else:
                ja += 1
        return pairs

    # ------------------------------------------------------------------
    # main loop – iterative (explicit stack) so we never blow the call stack
    # ------------------------------------------------------------------
    total_points = 0
    stack: list[tuple[int, int, int, int]] = [(0, len(A), 0, len(B))]

    while stack:
        a0, a1, b0, b1 = stack.pop()
        pairs = _one_lcs(a0, a1, b0, b1)
        if not pairs:
            continue

        # gather contiguous blocks inside this LCS so we can drop very short runs
        blocks_len = 1
        for (prev_i, prev_j), (cur_i, cur_j) in zip(pairs, pairs[1:]):
            if cur_i == prev_i + 1 and cur_j == prev_j + 1:
                blocks_len += 1
            else:
                if blocks_len >= min_run:
                    total_points += blocks_len
                blocks_len = 1
        if blocks_len >= min_run:
            total_points += blocks_len

        # push sub‑windows between matched pairs onto the stack
        prev_ai, prev_bj = a0, b0
        for ia, jb in pairs:
            stack.append((prev_ai, ia, prev_bj, jb))
            prev_ai, prev_bj = ia + 1, jb + 1
        stack.append((prev_ai, a1, prev_bj, b1))  # right‑most tail

    return int(total_points)



if __name__ == "__main__":
    main()
