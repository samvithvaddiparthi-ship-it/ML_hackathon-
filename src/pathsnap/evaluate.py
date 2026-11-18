"""Server-side scoring against the reference route. Only the API / offline evaluation scripts may call
this - its output (numbers only) may go to the UI, the reference arcs themselves must not.

CURRENT STATE: route-level metrics. Observation-level Top-1 accuracy / macro-F1 are added once the
per-observation label alignment (labels.py, Stage 2/3) exists; keys are already reserved.
"""
from pathlib import Path
import numpy as np
from .io import load_trip


def route_continuity(route_arcs, arcs_u, arcs_v) -> float:
    """Share of consecutive predicted arcs that are head-to-tail connected in the network."""
    if len(route_arcs) < 2:
        return 1.0
    r = np.asarray(route_arcs)
    return float(np.mean(arcs_v[r[:-1]] == arcs_u[r[1:]]))


def score_trip(pred: dict, trip_dir: Path) -> dict:
    trip = load_trip(Path(trip_dir), with_route=True)
    ref = set(int(a) for a in trip.route)
    # undirected match too: a two-way road predicted in the wrong direction is still the right street
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    ref_und = {frozenset((u[a], v[a])) for a in ref}
    p = pred["route_arcs"]
    tp = sum(a in ref for a in set(p))
    prec = tp / max(len(set(p)), 1)
    rec = tp / max(len(ref), 1)
    return {
        "top1_accuracy": None,            # filled after Stage 3 (needs per-observation labels)
        "macro_f1": None,
        "route_precision": round(prec, 4),
        "route_recall": round(rec, 4),
        "route_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "route_recall_undirected": round(float(np.mean([frozenset((u[a], v[a])) in {frozenset((u[b], v[b])) for b in p} for a in ref])), 4) if p else 0.0,
        "route_continuity": round(route_continuity(p, u, v), 4),
        "n_obs": pred["n_obs"],
    }
