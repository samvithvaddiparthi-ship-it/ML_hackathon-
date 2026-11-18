"""Per-observation reference labels.

The .route file is an ordered list of arc_ids, NOT one arc per GPS point. To train/score an observation-level
classifier we need "which reference arc was the vehicle on at observation i".

Method: monotone alignment by dynamic programming (DTW-style). Let D[i,k] = distance (m) of GPS point i to the
k-th route arc. We look for a non-decreasing index sequence k_0 <= k_1 <= ... <= k_{n-1} minimising
sum_i min(D[i,k_i], CLIP). Monotonicity encodes "the vehicle drives the route in order", which resolves loops,
U-turns and parallel roads that a nearest-arc rule gets wrong. Skipping route arcs is free (GPS gaps / short arcs).

Points whose aligned distance exceeds `max_dist` get label -1 (= unlabeled). This removes truncated-route tails
(trips 33, 54), off-route detours and gross GPS errors, so we never train/score on labels we do not trust.
"""
import numpy as np


def align_route(gx, gy, net, route, clip: float = 80.0):
    """Returns (k, d): aligned route index and distance for every observation."""
    route = np.asarray(route)
    n, m = len(gx), len(route)
    ax, ay, dx, dy = net.ax[route], net.ay[route], net.dx[route], net.dy[route]
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    ar = np.arange(m)

    def row(i):
        t = np.clip(((gx[i] - ax) * dx + (gy[i] - ay) * dy) / L2, 0, 1)
        return np.hypot(gx[i] - ax - t * dx, gy[i] - ay - t * dy)

    back = np.empty((n, m), dtype=np.int32)
    cost = np.minimum(row(0), clip)
    back[0] = ar
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)                       # best predecessor j<=k
        arg = np.maximum.accumulate(np.where(cost <= pm, ar, 0))
        back[i] = arg
        cost = pm + np.minimum(row(i), clip)
    k = np.empty(n, dtype=np.int64)
    k[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        k[i - 1] = back[i, k[i]]
    d = np.array([row(i)[k[i]] for i in range(n)]) if n < 50 else _dist_at(gx, gy, ax, ay, dx, dy, L2, k)
    return k, d


def _dist_at(gx, gy, ax, ay, dx, dy, L2, k):
    t = np.clip(((gx - ax[k]) * dx[k] + (gy - ay[k]) * dy[k]) / L2[k], 0, 1)
    return np.hypot(gx - ax[k] - t * dx[k], gy - ay[k] - t * dy[k])


def observation_labels(gx, gy, net, route, max_dist: float = 30.0):
    """arc_id label per observation (-1 = unlabeled), plus aligned distance and route index."""
    k, d = align_route(gx, gy, net, route)
    lab = np.asarray(route)[k].astype(np.int64)
    lab[d > max_dist] = -1
    return lab, d, k
