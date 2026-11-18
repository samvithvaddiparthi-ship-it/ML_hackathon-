"""Repeated inference-time benchmark: untuned vs tuned XGBoost (Report §4).

A single timing (as in results/bayes_opt/tuned_vs_untuned.csv) is noisy - two runs of scripts/05_bayes_opt.py
measured 17.8 vs 12.1 and 14.1 vs 14.6 ms / 1k obs. This script scores the full student-test candidate table
`--repeats` times per model (after one warm-up call) and reports the median and minimum.

Run:  python analysis/benchmark_inference.py [--repeats 7]   ->  results/analysis/inference_benchmark.csv
Needs results/cache/student_test_cands.pkl (scripts/02_build_dataset.py).
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import joblib
import numpy as np
import pandas as pd

from config import RESULTS, ARTIFACTS
from pathsnap.features import FEATURES

MODELS = {"XGBoost (untuned)": "XGBoost.joblib", "XGBoost (tuned)": "XGBoost_tuned.joblib"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    a = ap.parse_args()
    cands = pd.read_pickle(RESULTS / "cache" / "student_test_cands.pkl")
    obs = pd.read_pickle(RESULTS / "cache" / "student_test_obs.pkl")
    X = cands[FEATURES].to_numpy(np.float32)
    n_obs = len(obs)
    rows = []
    for name, f in MODELS.items():
        m = joblib.load(ARTIFACTS / "models" / f)
        m.predict_proba(X[:2000])                                   # warm-up
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter(); m.predict_proba(X); ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1e6 / n_obs
        rows.append({"model": name, "median_ms_per_1k_obs": np.median(ts), "min_ms_per_1k_obs": ts.min(),
                     "max_ms_per_1k_obs": ts.max(), "repeats": a.repeats, "n_obs": n_obs, "n_candidates": len(X),
                     "cpu_count": os.cpu_count(), "machine": platform.platform()})
        print(f"{name:20s} median {np.median(ts):6.2f}  min {ts.min():6.2f}  max {ts.max():6.2f} ms / 1k obs")
    df = pd.DataFrame(rows)
    u, t = df.median_ms_per_1k_obs
    print(f"tuned is {100 * (1 - t / u):.0f} % faster (median)")
    out = RESULTS / "analysis" / "inference_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
