"""Smoke test for the running API: all trips with use_hmm true AND false, required keys, optional HMM keys
(validated when present), no reference-route leakage, bad ids rejected.

Usage (API must be running):  python api/test_api.py [http://localhost:8000]
Exit code 0 = all checks passed.
"""
import sys
import time

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
PRED_KEYS = {"trip_id", "n_obs", "gps", "t", "pred_arc", "pred_prob", "route_arcs", "route_coords",
             "timing_ms", "model", "metrics", "server_latency_ms"}
METRIC_KEYS = {"top1_accuracy", "macro_f1", "route_precision", "route_recall", "route_f1",
               "route_recall_undirected", "route_continuity", "n_obs"}
FORBIDDEN = {"route", "reference", "reference_arcs", "reference_coords", "ref_arcs", "ref_coords", "true_arcs"}

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("  FAIL", msg)


h = requests.get(f"{BASE}/health", timeout=10).json()
check(h.get("status") == "ok" and "name" in h.get("model", {}), f"/health malformed: {h}")
print("health:", h)

trips = requests.get(f"{BASE}/trips", timeout=30).json()["trips"]
print(f"{len(trips)} trips listed")
check(len(trips) > 0, "no trips listed")

OPTIONAL_HMM_KEYS = {"pred_arc_classifier_only", "route_coords_classifier_only", "hmm"}

for use_hmm in (True, False):
    print(f"\n--- use_hmm={use_hmm} ---")
    for tr in trips:
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/predict", json={"trip_id": tr["trip_id"], "use_hmm": use_hmm}, timeout=300)
        ms = (time.perf_counter() - t0) * 1e3
        tag = f"{tr['trip_id']} (use_hmm={use_hmm})"
        check(r.status_code == 200, f"{tag}: HTTP {r.status_code} {r.text[:200]}")
        if r.status_code != 200:
            continue
        p = r.json()
        check(PRED_KEYS <= p.keys(), f"{tag}: missing keys {PRED_KEYS - p.keys()}")
        check(not (FORBIDDEN & p.keys()), f"{tag}: reference key leaked {FORBIDDEN & p.keys()}")
        check(METRIC_KEYS <= p["metrics"].keys(), f"{tag}: missing metrics {METRIC_KEYS - p['metrics'].keys()}")
        check(all(isinstance(v, (int, float)) or v is None for v in p["metrics"].values()), f"{tag}: non-numeric metric")
        check(p["n_obs"] == tr["n_obs"] == len(p["gps"]) == len(p["pred_arc"]), f"{tag}: n_obs mismatch")
        check(len(p["route_coords"]) == len(p["route_arcs"]), f"{tag}: route_coords/route_arcs length")
        # optional HMM-era keys: validated only when present
        if "pred_arc_classifier_only" in p:
            check(len(p["pred_arc_classifier_only"]) == p["n_obs"], f"{tag}: pred_arc_classifier_only length")
        if "route_coords_classifier_only" in p:
            check(isinstance(p["route_coords_classifier_only"], list), f"{tag}: route_coords_classifier_only not a list")
        if "hmm" in p:
            check(isinstance(p["hmm"], dict), f"{tag}: hmm is not an object")
        check(isinstance(p.get("timing_ms", {}), dict), f"{tag}: timing_ms is not an object")
        m = p["metrics"]
        h = p.get("hmm") or {}
        extra = (f"  breaks gap/disc={h.get('n_gap_breaks')}/{h.get('n_disconnect_breaks')}" if h else "") + \
                (f"  [{', '.join(sorted(OPTIONAL_HMM_KEYS & p.keys()))}]" if OPTIONAL_HMM_KEYS & p.keys() else "")
        f1 = m["route_f1"]
        print(f"  {tr['trip_id']}  n_obs={p['n_obs']:5d}  top1={m['top1_accuracy'] if m['top1_accuracy'] is None else round(m['top1_accuracy'], 3)}  "
              f"route_f1={'n/a' if f1 is None else f'{f1:.3f}'}  server={p['server_latency_ms']:.0f} ms  roundtrip={ms:.0f} ms{extra}")

for bad, code in [("does_not_exist", 404), ("..%2F..%2Fetc", 400), ("../train", 400)]:
    r = requests.post(f"{BASE}/predict", json={"trip_id": bad}, timeout=10)
    check(r.status_code in (code, 404, 422), f"bad id {bad!r} returned {r.status_code}")

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} FAILURE(S)")
sys.exit(1 if failures else 0)
