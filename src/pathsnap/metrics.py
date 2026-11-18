"""Metrics on candidate tables (rows = (trip_id, obs, arc) with score p and binary target y)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score, confusion_matrix


def obs_boundaries(df: pd.DataFrame):
    """Start index of every observation block (candidate tables are stored contiguous per (trip, obs))."""
    key = df["trip_id"].astype(str).to_numpy()
    obs = df["obs"].to_numpy()
    new = np.r_[True, (obs[1:] != obs[:-1]) | (key[1:] != key[:-1])]
    return np.flatnonzero(new)


def top1_rows(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """Row index of the highest-scored candidate of every observation (ties -> first = closest)."""
    starts = obs_boundaries(df)
    sizes = np.diff(np.r_[starts, len(df)])
    obs_id = np.repeat(np.arange(len(starts)), sizes)
    order = np.lexsort((np.arange(len(p)), -p, obs_id))      # per obs: highest p, then original (distance) order
    first = np.r_[True, obs_id[order][1:] != obs_id[order][:-1]]
    return order[first]


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "accuracy": accuracy_score(y, yhat),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
    }


def top1_accuracy(df, p):
    r = top1_rows(df, p)
    return float(df["y"].to_numpy()[r].mean())


def route_continuity_from_top1(df, p, arcs_by_trip: dict) -> float:
    """Collapse each trip's top-1 arc sequence (consecutive duplicates removed) and return the share of
    consecutive arcs that are connected head-to-tail in the network (pooled over trips)."""
    r = top1_rows(df, p)
    sel = df.iloc[r][["trip_id", "obs", "arc"]]
    conn = tot = 0
    for tid, g in sel.groupby("trip_id", sort=False):
        a = g.sort_values("obs")["arc"].to_numpy()
        a = a[np.r_[True, a[1:] != a[:-1]]]
        if len(a) < 2:
            continue
        u, v = arcs_by_trip[tid]
        conn += int(np.sum(v[a[:-1]] == u[a[1:]])); tot += len(a) - 1
    return conn / max(tot, 1)


def binary_confusion(y, p, thr=0.5):
    return confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
