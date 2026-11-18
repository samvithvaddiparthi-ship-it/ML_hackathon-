"""Candidate generation + feature engineering (one row per (observation, candidate arc)).

Problem framing: for every GPS observation we generate the road arcs within RADIUS metres (max MAX_K).
A classifier scores each (observation, candidate) pair: "is this the arc the vehicle is on?". The highest scored
candidate is the observation-level prediction (Top-1); the scores also become HMM emission probabilities.

All features are trip-independent (no absolute coordinates / ids), so a model trained on 80 trips around the
world transfers to unseen cities. Feature groups:
  geometric   : distance to arc, position along arc, arc length, rank / gap to the closest candidate
  direction   : cosine between GPS heading and arc heading (separates the two directions of a two-way road)
  temporal    : distance of neighbouring observations (i±1..±5) to the SAME arc -> "does the track follow this arc"
  kinematics  : speed, dt to neighbours, heading reliability, jump flag
  topology    : in/out degree, whether the arc continues from/into the nearest arc of the previous/next point
  context     : number of candidates, softmax-normalised Gaussian distance likelihood within the observation
"""
import numpy as np
import pandas as pd
from .geo import make_projector, RoadNet

RADIUS = 50.0          # m  (GPS→route p95 ≈ 14 m, max ≈ 36 m on labelled points; 50 m gives ~99.7 % recall)
MAX_K = 12
SIGMA = 5.0            # m, ≈ median GPS→route distance (Stage 1)
WIN_SHORT, WIN_LONG = 2, 5

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cand", "gauss_norm",
    "t_along", "arc_len", "cos_head", "abs_cos_head", "head_reliab",
    "win2_mean", "win2_max", "win5_mean", "win5_max", "win5_rank",
    "speed", "dt_prev", "dt_next", "jump",
    "out_deg_v", "in_deg_u", "cont_prev", "cont_next", "has_reverse_cand",
]


def track_kinematics(gx, gy, t):
    n = len(gx)
    i = np.arange(n)
    lo, hi = np.clip(i - 2, 0, n - 1), np.clip(i + 2, 0, n - 1)
    hx, hy = gx[hi] - gx[lo], gy[hi] - gy[lo]
    disp = np.hypot(hx, hy)
    heading = np.arctan2(hx, hy)
    p, nx_ = np.clip(i - 1, 0, n - 1), np.clip(i + 1, 0, n - 1)
    dt_span = np.maximum(t[nx_] - t[p], 1e-3)
    speed = np.hypot(gx[nx_] - gx[p], gy[nx_] - gy[p]) / dt_span
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    step = np.hypot(np.diff(gx), np.diff(gy)) / np.maximum(np.diff(t), 1e-3)
    jump = np.r_[0, (step > 60).astype(int)] | np.r_[(step > 60).astype(int), 0]
    reliab = np.clip(disp / 20.0, 0, 1)            # ~0 when stopped (heading meaningless), 1 when moving >= 20 m / 4 obs
    return heading, reliab, speed, dt_prev, dt_next, jump


def build_trip_features(trip, labels=None, radius=RADIUS, max_k=MAX_K):
    """Returns (DataFrame of candidate rows, RoadNet, projected gps). `labels` (per-obs arc ids, -1 = unknown)
    is optional and only used to create the target column - features never depend on it."""
    trk = trip.track
    proj = make_projector(trk.lon.mean(), trk.lat.mean())
    net = RoadNet(trip.nodes, trip.arcs, proj)
    gx, gy = proj.transform(trk.lon.to_numpy(), trk.lat.to_numpy())
    t = trk.t.to_numpy()
    n = len(gx)
    oi, arc, d, ta = net.candidates(gx, gy, radius=radius, max_k=max_k)

    heading, reliab, speed, dtp, dtn, jump = track_kinematics(gx, gy, t)
    f = pd.DataFrame({"obs": oi, "arc": arc, "dist": d, "t_along": ta})
    g = f.groupby("obs")["dist"]
    f["dist_rank"] = g.rank(method="first") - 1
    dmin = g.transform("min")
    f["dist_minus_min"] = f.dist - dmin
    f["dist_ratio_min"] = (f.dist + 1) / (dmin + 1)
    f["n_cand"] = g.transform("size")
    lik = np.exp(-0.5 * (f.dist / SIGMA) ** 2)
    f["gauss_norm"] = lik / np.maximum(lik.groupby(f.obs).transform("sum"), 1e-300)
    f["arc_len"] = net.length[arc]
    dh = heading[oi] - net.heading[arc]
    f["cos_head"] = np.cos(dh) * reliab[oi]
    f["abs_cos_head"] = np.abs(np.cos(dh)) * reliab[oi]
    f["head_reliab"] = reliab[oi]

    # temporal support: how close neighbouring observations are to the same arc
    offs = {}
    for o in range(-WIN_LONG, WIN_LONG + 1):
        j = np.clip(oi + o, 0, n - 1)
        offs[o] = np.minimum(net.project(gx[j], gy[j], arc)[0], 3 * radius)
    s = np.stack([offs[o] for o in range(-WIN_SHORT, WIN_SHORT + 1)])
    l = np.stack(list(offs.values()))
    f["win2_mean"], f["win2_max"] = s.mean(0), s.max(0)
    f["win5_mean"], f["win5_max"] = l.mean(0), l.max(0)
    f["win5_rank"] = f.groupby("obs")["win5_mean"].rank(method="first") - 1

    f["speed"], f["dt_prev"], f["dt_next"], f["jump"] = speed[oi], dtp[oi], dtn[oi], jump[oi]

    out_deg = np.bincount(net.u, minlength=len(net.x))
    in_deg = np.bincount(net.v, minlength=len(net.x))
    f["out_deg_v"], f["in_deg_u"] = out_deg[net.v[arc]], in_deg[net.u[arc]]

    # topology vs. the nearest candidate of the previous / next observation
    nearest = np.full(n, -1)
    first = np.r_[True, oi[1:] != oi[:-1]]
    nearest[oi[first]] = arc[first]
    pa = nearest[np.clip(oi - 1, 0, n - 1)]
    na = nearest[np.clip(oi + 1, 0, n - 1)]
    f["cont_prev"] = ((pa == arc) | (net.v[pa] == net.u[arc])).astype(int)
    f["cont_next"] = ((na == arc) | (net.u[na] == net.v[arc])).astype(int)

    rkey = set(zip(oi, net.u[arc], net.v[arc]))
    f["has_reverse_cand"] = np.fromiter(((o, b, a) in rkey for o, a, b in zip(oi, net.u[arc], net.v[arc])), int, len(oi))

    f.insert(0, "trip_id", trip.trip_id)
    if labels is not None:
        f["label_arc"] = labels[oi]
        f["y"] = (f.arc.to_numpy() == labels[oi]).astype(np.int8)
    return f, net, (gx, gy)
