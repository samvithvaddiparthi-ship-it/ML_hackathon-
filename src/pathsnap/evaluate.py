"""Server-side scoring against the reference route. Only the API / offline evaluation scripts may call
this - its output (numbers only) may go to the UI; the reference arcs themselves must not.

Metric definitions (used identically everywhere in the project):
  top1_accuracy     observation-level: predicted arc == aligned reference arc, over labelled observations
                    (labels from pathsnap.labels: monotone alignment of GPS to the .route sequence, -1 = untrusted)
  macro_f1          observation-level macro-F1 over reference arc ids (each arc of the route is a class)
  route_*           set overlap between predicted route arcs and reference route arcs
  route_continuity  share of consecutive predicted route arcs that are head-to-tail connected in the network
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from .io import load_trip
from .geo import make_projector, RoadNet
from .labels import observation_labels


def route_continuity(route_arcs, arcs_u, arcs_v) -> float:
    if len(route_arcs) < 2:
        return 1.0
    r = np.asarray(route_arcs)
    return float(np.mean(arcs_v[r[:-1]] == arcs_u[r[1:]]))


def reference_labels(trip):
    proj = make_projector(trip.track.lon.mean(), trip.track.lat.mean())
    net = RoadNet(trip.nodes, trip.arcs, proj)
    gx, gy = proj.transform(trip.track.lon.to_numpy(), trip.track.lat.to_numpy())
    lab, _, _ = observation_labels(gx, gy, net, trip.route)
    return lab


def obs_metrics(pred_arc, lab):
    pred_arc, lab = np.asarray(pred_arc), np.asarray(lab)
    m = lab >= 0
    if not m.any():
        return None, None
    acc = float(np.mean(pred_arc[m] == lab[m]))
    f1 = float(f1_score(lab[m], pred_arc[m], labels=np.unique(lab[m]), average="macro", zero_division=0))
    return acc, f1


def score_trip(pred: dict, trip_dir: Path) -> dict:
    trip = load_trip(Path(trip_dir), with_route=True)
    lab = reference_labels(trip)
    acc, f1 = obs_metrics(pred["pred_arc"], lab)
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    ref = set(int(a) for a in trip.route)
    p = pred["route_arcs"]
    tp = sum(a in ref for a in set(p))
    prec = tp / max(len(set(p)), 1)
    rec = tp / max(len(ref), 1)
    pred_und = {frozenset((u[b], v[b])) for b in p}
    rec_und = float(np.mean([frozenset((u[a], v[a])) in pred_und for a in ref])) if p else 0.0
    extra = {}
    if "pred_arc_classifier_only" in pred:            # classifier-only baseline of the same request (numbers only)
        ca, cf = obs_metrics(pred["pred_arc_classifier_only"], lab)
        cr = pred.get("route_arcs_classifier_only", [])
        extra = {"top1_accuracy_classifier_only": None if ca is None else round(ca, 4),
                 "route_continuity_classifier_only": round(route_continuity(cr, u, v), 4) if cr else None}
    return {
        **extra,
        "top1_accuracy": None if acc is None else round(acc, 4),
        "macro_f1": None if f1 is None else round(f1, 4),
        "route_precision": round(prec, 4),
        "route_recall": round(rec, 4),
        "route_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "route_recall_undirected": round(rec_und, 4),
        "route_continuity": round(route_continuity(p, u, v), 4),
        "n_obs": pred["n_obs"],
        "n_scored_obs": int((lab >= 0).sum()),
    }
