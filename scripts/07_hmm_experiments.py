"""Stage A4 - HMM/Viterbi decoder: parameter tuning + classifier-only vs classifier+HMM comparison.

Leakage-safe protocol
  * Emission probabilities for the 80 TRAIN trips are OUT-OF-FOLD: GroupKFold(5) (same folds as A2/A3), each trip
    is scored by a model that never saw it. Student-test probabilities come from the models refit on all 80 trips.
  * Decoder parameters (beta, w_e, gap_s) are chosen on the TRAIN out-of-fold decodes only (grid search),
    then evaluated once on the 10 STUDENT-TEST trips. The evaluator partition is never touched.
  * Emission model (untuned vs Optuna-tuned XGBoost) is chosen the same way (train OOF HMM Top-1).
Outputs: results/hmm/*.csv|json, plots in results/hmm/plots, artifacts/final/hmm_config.json
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS, ARTIFACTS, SEED
from pathsnap.io import load_trip
from pathsnap.geo import make_projector, RoadNet
from pathsnap.features import FEATURES
from pathsnap.hmm import viterbi, expand_route, Graph
from pathsnap.models import MODELS
from xgboost import XGBClassifier

OUT = RESULTS / "hmm"; OUT.mkdir(parents=True, exist_ok=True)
FINAL = ARTIFACTS / "final"; FINAL.mkdir(parents=True, exist_ok=True)
BEST_TUNED = json.load(open(RESULTS / "bayes_opt" / "best_params.json"))["params"]
EMISSION_MODELS = {
    "XGBoost_untuned": MODELS["XGBoost"]["make"],
    "XGBoost_tuned": lambda: XGBClassifier(**BEST_TUNED, tree_method="hist", eval_metric="logloss",
                                           random_state=SEED, n_jobs=-1),
}
# NOTE: Viterbi maximises sum(w_e*log p) - sum|d_route-d_gps|/beta, which is equivalent (x beta) to
# sum((w_e*beta)*log p) - sum|...|  -> only the product lambda = w_e*beta changes the decoded path.
# The grid therefore also extends lambda below the first grid's best corner (beta 3, w_e 0.5 -> lambda 1.5).
GRID = ([{"beta": b, "w_e": w, "gap_s": 60.0} for b in (3.0, 7.0, 15.0, 30.0) for w in (0.5, 1.0, 2.0)]
        + [{"beta": 3.0, "w_e": w, "gap_s": 60.0} for w in (0.0833, 0.1667, 0.25, 0.3333)])
GAP_GRID = (30.0, 60.0, 120.0)


def oof_scores(cands, factory):
    trips = cands.trip_id.to_numpy()
    X = cands[FEATURES].to_numpy(np.float32); y = cands.y.to_numpy(); lab = (cands.label_arc >= 0).to_numpy()
    p = np.zeros(len(cands))
    for tri, vai in GroupKFold(5).split(cands, groups=trips):
        tri = tri[lab[tri]]
        p[vai] = factory().fit(X[tri], y[tri]).predict_proba(X[vai])[:, 1]
    return p


def classifier_only(c, p):
    o = c.obs.to_numpy(); st = np.searchsorted(o, np.arange(o.max() + 2))
    arcs = c.arc.to_numpy()
    return np.array([arcs[s + np.argmax(p[s:e])] for s, e in zip(st[:-1], st[1:])])


def collapse(seq):
    seq = [int(a) for a in seq if a >= 0]
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def trip_metrics(pred_obs, route, lab, ref, u, v, L):
    ok = lab >= 0
    refset, ps = set(ref.tolist()), set(route)
    tp = len(ps & refset)
    r = np.asarray(route)
    cont = int(np.sum(v[r[:-1]] == u[r[1:]])) if len(r) > 1 else 0
    return {"n_lab": int(ok.sum()), "n_correct": int((pred_obs[ok] == lab[ok]).sum()),
            "tp": tp, "n_pred_arcs": len(ps), "n_ref_arcs": len(refset),
            "len_tp": float(L[list(ps & refset)].sum()), "len_ref": float(L[list(refset)].sum()),
            "len_pred": float(L[list(ps)].sum()), "cont_ok": cont, "cont_tot": max(len(r) - 1, 0),
            "macro_f1_arcs": float(f1_score(lab[ok], pred_obs[ok], labels=np.unique(lab[ok]), average="macro",
                                            zero_division=0)) if ok.any() else np.nan}


def run_trip(base, tid, c, probs_by_model, obs_lab, configs):
    tr = load_trip(base / tid)
    proj = make_projector(tr.track.lon.mean(), tr.track.lat.mean())
    net = RoadNet(tr.nodes, tr.arcs, proj)
    gx, gy = proj.transform(tr.track.lon.to_numpy(), tr.track.lat.to_numpy())
    t = tr.track.t.to_numpy()
    u, v, L = net.u, net.v, net.length
    g = Graph(net)
    rows = []
    for mname, p in probs_by_model.items():
        t0 = time.perf_counter(); clf = classifier_only(c, p); tc = time.perf_counter() - t0
        rows.append({"trip_id": tid, "model": mname, "decoder": "classifier_only", "secs": tc,
                     **trip_metrics(clf, collapse(clf), obs_lab, tr.route, u, v, L)})
        for cfg in configs:
            t0 = time.perf_counter()
            r = viterbi(c, p, t, gx, gy, net, cfg, graph=g)
            route = expand_route(r["pred_arc"], g)
            rows.append({"trip_id": tid, "model": mname, "decoder": "hmm", **cfg, "secs": time.perf_counter() - t0,
                         "gap_breaks": r["n_gap_breaks"], "disc_breaks": r["n_disconnect_breaks"],
                         **trip_metrics(r["pred_arc"], route, obs_lab, tr.route, u, v, L)})
    return rows


def aggregate(df, keys):
    g = df.groupby(keys, dropna=False)
    a = g.agg(n_lab=("n_lab", "sum"), n_correct=("n_correct", "sum"), tp=("tp", "sum"), n_pred=("n_pred_arcs", "sum"),
              n_ref=("n_ref_arcs", "sum"), len_tp=("len_tp", "sum"), len_ref=("len_ref", "sum"),
              len_pred=("len_pred", "sum"), cont_ok=("cont_ok", "sum"), cont_tot=("cont_tot", "sum"),
              macro_f1_arcs=("macro_f1_arcs", "mean"), secs=("secs", "sum"))
    a["top1_acc"] = a.n_correct / a.n_lab
    a["route_precision"] = a.tp / a.n_pred; a["route_recall"] = a.tp / a.n_ref
    a["route_f1"] = 2 * a.route_precision * a.route_recall / (a.route_precision + a.route_recall)
    a["length_precision"] = a.len_tp / a.len_pred; a["length_recall"] = a.len_tp / a.len_ref
    a["route_continuity"] = a.cont_ok / a.cont_tot
    return a.reset_index()


def per_trip_scores(df):
    d = df.copy()
    d["top1_acc"] = d.n_correct / d.n_lab.replace(0, np.nan)
    p = d.tp / d.n_pred_arcs; r = d.tp / d.n_ref_arcs
    d["route_f1"] = 2 * p * r / (p + r); d["route_continuity"] = d.cont_ok / d.cont_tot.replace(0, np.nan)
    return d


def main():
    tr = pd.read_pickle(RESULTS / "cache" / "train_cands.pkl")
    tr_obs = pd.read_pickle(RESULTS / "cache" / "train_obs.pkl")
    st = pd.read_pickle(RESULTS / "cache" / "student_test_cands.pkl")
    st_obs = pd.read_pickle(RESULTS / "cache" / "student_test_obs.pkl")

    # ---- emission probabilities
    oof_path = RESULTS / "cache" / "train_oof_scores.pkl"
    if oof_path.exists():
        oof = pd.read_pickle(oof_path)
    else:
        oof = {}
        for name, fac in EMISSION_MODELS.items():
            t0 = time.time(); oof[name] = oof_scores(tr, fac); print(f"OOF {name}: {time.time() - t0:.0f}s", flush=True)
        pd.to_pickle(oof, oof_path)
    st_scores = {"XGBoost_untuned": pd.read_pickle(RESULTS / "cache" / "student_test_scores.pkl")["scores"]["XGBoost"],
                 "XGBoost_tuned": pd.read_pickle(RESULTS / "cache" / "student_test_scores_tuned.pkl")["scores"]["XGBoost_tuned"]}

    def run_partition(cands, obs, scores, base, configs):
        jobs = []
        for tid, c in cands.groupby("trip_id", sort=False):
            idx = c.index.to_numpy()
            lab = obs[obs.trip_id == tid].sort_values("obs").label_arc.to_numpy()
            jobs.append(delayed(run_trip)(base, tid, c.reset_index(drop=True),
                                          {m: s[idx] for m, s in scores.items()}, lab, configs))
        res = Parallel(n_jobs=-1)(jobs)
        return pd.DataFrame([r for rr in res for r in rr])

    # ---- 1. grid search on TRAIN (out-of-fold emissions)
    t0 = time.time()
    tr_rows = run_partition(tr, tr_obs, oof, TRAIN_DIR, GRID)
    print(f"train grid: {time.time() - t0:.0f}s", flush=True)
    grid = aggregate(tr_rows, ["model", "decoder", "beta", "w_e", "gap_s"])
    hm = grid[grid.decoder == "hmm"].sort_values("top1_acc", ascending=False)
    best = hm.iloc[0]
    best_model = best.model
    print(grid.sort_values(["model", "decoder", "top1_acc"]).round(4).to_string())
    # ---- 1b. gap threshold sensitivity at the best (beta, w_e), best emission model
    gap_cfgs = [{"beta": best.beta, "w_e": best.w_e, "gap_s": gs} for gs in GAP_GRID if gs != best.gap_s]
    gap_rows = run_partition(tr, tr_obs, {best_model: oof[best_model]}, TRAIN_DIR, gap_cfgs)
    gap_rows = gap_rows[gap_rows.decoder == "hmm"]
    allgrid = pd.concat([tr_rows, gap_rows], ignore_index=True)
    allgrid.to_csv(OUT / "train_oof_grid_per_trip.csv", index=False)
    grid = aggregate(allgrid, ["model", "decoder", "beta", "w_e", "gap_s"])
    grid["lambda_we_x_beta"] = (grid.w_e * grid.beta).round(4)
    grid.to_csv(OUT / "train_oof_grid.csv", index=False)
    hm = grid[(grid.decoder == "hmm") & (grid.model == best_model)].sort_values(["top1_acc", "route_f1"], ascending=False)
    best = hm.iloc[0]
    cfg = {"beta": float(best.beta), "w_e": float(best.w_e), "gap_s": float(best.gap_s), "vmax": 50.0}
    print("selected decoder config:", cfg, "emission model:", best_model, flush=True)

    # ---- 2. student-test: classifier-only vs HMM, both emission models, selected config
    st_rows = run_partition(st, st_obs, st_scores, STUDENT_TEST_DIR, [cfg])
    st_rows.to_csv(OUT / "student_test_per_trip.csv", index=False)
    st_agg = aggregate(st_rows, ["model", "decoder"])
    st_agg.to_csv(OUT / "student_test_summary.csv", index=False)
    tr_sel = aggregate(allgrid[(allgrid.decoder == "classifier_only") |
                               ((allgrid.beta == cfg["beta"]) & (allgrid.w_e == cfg["w_e"]) & (allgrid.gap_s == cfg["gap_s"]))],
                       ["model", "decoder"])
    tr_sel.to_csv(OUT / "train_oof_summary.csv", index=False)
    cols = ["model", "decoder", "top1_acc", "macro_f1_arcs", "route_precision", "route_recall", "route_f1",
            "length_recall", "route_continuity", "secs"]
    print("\nTRAIN (out-of-fold):\n", tr_sel[cols].round(4).to_string())
    print("\nSTUDENT-TEST:\n", st_agg[cols].round(4).to_string())
    pt = per_trip_scores(st_rows)
    print(pt[["trip_id", "model", "decoder", "n_lab", "top1_acc", "route_f1", "route_continuity", "disc_breaks", "gap_breaks"]]
          .round(4).to_string())

    json.dump({**cfg, "emission_model": best_model,
               "selection": "grid search on TRAIN out-of-fold emissions, objective = pooled observation Top-1 (tie: route F1)",
               "train_oof_top1": float(best.top1_acc), "train_oof_route_f1": float(best.route_f1)},
              open(FINAL / "hmm_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
