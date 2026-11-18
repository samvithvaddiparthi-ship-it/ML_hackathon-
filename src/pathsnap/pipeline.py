"""Inference entry point used by the API.  CONTRACT (see docs/INTERFACE.md) - do not change the
return keys without updating the API/UI.

The prediction path NEVER reads the .route file (load_trip(..., with_route=False)), so reference
labels cannot leak into predictions.

CURRENT STATE: baseline = nearest candidate arc per observation (no ML yet). It will be replaced by
classifier + HMM/Viterbi once Stage 3-6 are done; the output format stays identical.
"""
import time
from pathlib import Path
import numpy as np
from .io import load_trip
from .geo import make_projector, RoadNet

MODEL_INFO = {"name": "baseline-nearest-arc", "version": "0.1"}


def collapse_route(arc_seq):
    """Consecutive duplicates removed -> ordered list of traversed arcs."""
    arc_seq = [int(a) for a in arc_seq if a >= 0]
    return [a for i, a in enumerate(arc_seq) if i == 0 or a != arc_seq[i - 1]]


def predict_trip(trip_dir: Path, use_hmm: bool = True) -> dict:
    t0 = time.perf_counter()
    trip = load_trip(Path(trip_dir), with_route=False)
    trk = trip.track
    proj = make_projector(trk.lon.mean(), trk.lat.mean())
    net = RoadNet(trip.nodes, trip.arcs, proj)
    gx, gy = proj.transform(trk.lon.to_numpy(), trk.lat.to_numpy())
    t1 = time.perf_counter()
    oi, ca, cd, _ = net.candidates(gx, gy)
    pred = np.full(len(trk), -1, dtype=int)
    prob = np.zeros(len(trk))
    first = np.r_[True, oi[1:] != oi[:-1]]            # candidates are sorted by distance per obs
    pred[oi[first]] = ca[first]
    prob[oi[first]] = 1.0
    t2 = time.perf_counter()
    route = collapse_route(pred)
    return {
        "trip_id": trip.trip_id,
        "n_obs": int(len(trk)),
        "gps": trk[["lat", "lon"]].to_numpy().round(7).tolist(),
        "t": trk.t.tolist(),
        "pred_arc": pred.tolist(),
        "pred_prob": prob.round(4).tolist(),
        "route_arcs": route,
        "route_coords": [net.arc_latlon(a) for a in route],
        "timing_ms": {"load_and_index": round((t1 - t0) * 1e3, 1), "predict": round((t2 - t1) * 1e3, 1),
                      "total": round((time.perf_counter() - t0) * 1e3, 1)},
        "model": MODEL_INFO,
    }
