"""Stage A1 - labels, candidates and features for train (80) and student_test (10).
Outputs
  results/cache/{train,student_test}_cands.pkl   candidate-level feature tables (git-ignored, ~regenerable)
  results/cache/{train,student_test}_obs.pkl     per-observation label table
  results/dataset/trip_dataset_summary.csv       per-trip label coverage, candidate miss rate, nearest-arc baseline
  results/dataset/radius_study.csv               candidate recall vs search radius (train only)
"""
import sys, time
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS
from pathsnap.io import load_trip, list_trips
from pathsnap.geo import make_projector, RoadNet
from pathsnap.labels import observation_labels
from pathsnap.features import build_trip_features

CACHE, OUT = RESULTS / "cache", RESULTS / "dataset"
CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)


def process(trip_dir, partition):
    t0 = time.time()
    trip = load_trip(trip_dir)
    proj = make_projector(trip.track.lon.mean(), trip.track.lat.mean())
    net = RoadNet(trip.nodes, trip.arcs, proj)
    gx, gy = proj.transform(trip.track.lon.to_numpy(), trip.track.lat.to_numpy())
    lab, adist, _ = observation_labels(gx, gy, net, trip.route)
    cands, _, _ = build_trip_features(trip, labels=lab)
    obs = pd.DataFrame({"trip_id": trip.trip_id, "obs": np.arange(len(lab)), "label_arc": lab, "align_dist": adist})
    hit = cands.groupby("obs")["y"].max()
    obs["in_cands"] = obs.obs.map(hit).fillna(0).astype(int)
    top = cands[cands.dist_rank == 0].set_index("obs")["arc"]
    obs["nearest_arc"] = obs.obs.map(top).fillna(-1).astype(int)
    lb = obs.label_arc >= 0
    summ = {"partition": partition, "trip_id": trip.trip_id, "n_obs": len(obs), "n_labeled": int(lb.sum()),
            "labeled_frac": lb.mean(), "n_cand_rows": len(cands), "mean_cands": len(cands) / len(obs),
            "cand_miss_rate": 1 - obs.loc[lb, "in_cands"].mean(),
            "nearest_arc_top1": (obs.loc[lb, "nearest_arc"] == obs.loc[lb, "label_arc"]).mean(),
            "align_dist_median": float(np.median(adist[lb])), "secs": time.time() - t0}
    # radius study (recall of the label among ALL arcs within r, ignoring the top-k cap)
    if partition == "train":
        li = np.where(lb)[0]
        dl, _ = net.project(gx[li], gy[li], lab[li])
        for r in [15, 20, 30, 40, 50, 75, 100]:
            summ[f"recall_r{r}"] = float(np.mean(dl <= r))
    return cands, obs, summ


def main():
    for part, d in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        res = Parallel(n_jobs=-1, verbose=0)(delayed(process)(td, part) for td in list_trips(d))
        cands = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        summ = pd.DataFrame([r[2] for r in res])
        cands.to_pickle(CACHE / f"{part}_cands.pkl"); obs.to_pickle(CACHE / f"{part}_obs.pkl")
        summ.to_csv(OUT / f"trip_dataset_summary_{part}.csv", index=False)
        lb = obs.label_arc >= 0
        print(f"[{part}] trips={len(summ)} obs={len(obs)} labeled={lb.mean():.3f} cand_rows={len(cands)} "
              f"pos_rate={cands.y.mean():.3f} cand_miss={1 - obs.loc[lb, 'in_cands'].mean():.4f} "
              f"nearest_top1={(obs.loc[lb, 'nearest_arc'] == obs.loc[lb, 'label_arc']).mean():.4f}")
        if part == "train":
            rs = summ.filter(like="recall_r")
            w = summ.n_labeled / summ.n_labeled.sum()
            rad = pd.DataFrame({"radius_m": [int(c[8:]) for c in rs.columns], "label_recall": (rs.mul(w, axis=0)).sum().values})
            rad.to_csv(OUT / "radius_study.csv", index=False); print(rad.to_string(index=False))


if __name__ == "__main__":
    main()
