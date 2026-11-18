"""Inference entry point used by the API.  CONTRACT: docs/INTERFACE.md - keys may be added, never removed.

Full prediction pipeline for one trip folder:
  raw .track/.nodes/.arcs  ->  per-trip AEQD projection + road-network index
  -> candidate arcs (<= 50 m, <= 12 per GPS observation) + 25 features      (pathsnap.features)
  -> XGBoost candidate probabilities                                       (artifacts/final/emission_model.joblib)
  -> HMM / Viterbi decoding over the directed road network                  (pathsnap.hmm, artifacts/final/hmm_config.json)
  -> one arc per observation + continuous route (shortest-path gap filling)

The .route file is NEVER read here (load_trip(..., with_route=False)), so reference labels cannot leak into
predictions; nothing is hard-coded per trip. Artifacts are produced by scripts/08_export_final.py.
"""
import json
import time
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from .io import load_trip
from .features import build_trip_features, FEATURES
from .hmm import viterbi, expand_route

ART = Path(__file__).resolve().parents[2] / "artifacts" / "final"


@lru_cache(maxsize=1)
def load_artifacts():
    meta = json.load(open(ART / "model_card.json"))
    return joblib.load(ART / "emission_model.joblib"), json.load(open(ART / "hmm_config.json")), meta


def _model_info():
    try:
        meta = load_artifacts()[2]
        return {"name": meta["name"], "version": meta["version"]}
    except FileNotFoundError:
        return {"name": "artifacts-missing (run scripts/08_export_final.py)", "version": "0"}


MODEL_INFO = _model_info()


def collapse_route(arc_seq):
    arc_seq = [int(a) for a in arc_seq if a >= 0]
    return [a for i, a in enumerate(arc_seq) if i == 0 or a != arc_seq[i - 1]]


def predict_trip(trip_dir: Path, use_hmm: bool = True) -> dict:
    model, hmm_cfg, _ = load_artifacts()
    t0 = time.perf_counter()
    trip = load_trip(Path(trip_dir), with_route=False)
    cands, net, (gx, gy) = build_trip_features(trip)
    t1 = time.perf_counter()
    prob = model.predict_proba(cands[FEATURES].to_numpy(np.float32))[:, 1]
    t2 = time.perf_counter()

    obs, arcs = cands["obs"].to_numpy(), cands["arc"].to_numpy()
    n = len(trip.track)
    starts = np.searchsorted(obs, np.arange(n + 1))
    best_rows = np.array([s + int(np.argmax(prob[s:e])) for s, e in zip(starts[:-1], starts[1:])])
    clf_arc = arcs[best_rows]

    hmm_info = {"used_hmm": bool(use_hmm)}
    if use_hmm:
        r = viterbi(cands, prob, trip.track.t.to_numpy(), gx, gy, net, hmm_cfg)
        rows, g = r["row"], r["graph"]
        route = expand_route(r["pred_arc"], g)
        hmm_info.update(n_gap_breaks=r["n_gap_breaks"], n_disconnect_breaks=r["n_disconnect_breaks"],
                        params={k: hmm_cfg[k] for k in ("beta", "w_e", "gap_s", "vmax")})
    else:
        rows, route = best_rows, collapse_route(clf_arc)
    t3 = time.perf_counter()
    pred_arc = arcs[rows]
    clf_route = collapse_route(clf_arc)
    return {
        "trip_id": trip.trip_id,
        "n_obs": int(n),
        "gps": trip.track[["lat", "lon"]].to_numpy().round(7).tolist(),
        "t": trip.track.t.tolist(),
        "pred_arc": pred_arc.astype(int).tolist(),
        "pred_prob": prob[rows].round(4).tolist(),
        "route_arcs": route,
        "route_coords": [net.arc_latlon(a) for a in route],
        "pred_arc_classifier_only": clf_arc.astype(int).tolist(),
        "route_arcs_classifier_only": clf_route,
        "route_coords_classifier_only": [net.arc_latlon(a) for a in clf_route],
        "hmm": hmm_info,
        "timing_ms": {"features": round((t1 - t0) * 1e3, 1), "classifier": round((t2 - t1) * 1e3, 1),
                      "hmm": round((t3 - t2) * 1e3, 1), "total": round((time.perf_counter() - t0) * 1e3, 1)},
        "model": _model_info(),
    }
