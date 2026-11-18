"""Stage A2 - train & compare candidate classifiers.

Protocol (identical for every model):
  * Cross-validation: GroupKFold(5) over the 80 TRAIN trips (a trip never appears in train and validation fold).
    Fit on labelled observations of 4 folds; predict ALL candidate rows of the held-out trips.
    Classification metrics on labelled rows; route continuity on the full top-1 sequence.
  * Student-test: refit on all 80 train trips, evaluate on the 10 STUDENT_TEST trips.
  * Training curves: only for models with an iterative training history (MLP epochs, LightGBM/XGBoost rounds),
    recorded on fold-0 split (train = 4 folds, validation = held-out fold).
Outputs: results/models/*.csv|json, results/cache/student_test_scores.pkl, artifacts/models/*.joblib
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS, ARTIFACTS
from pathsnap.io import load_trip
from pathsnap.features import FEATURES
from pathsnap.models import MODELS, fit_model, SEED
from pathsnap.metrics import (candidate_metrics, top1_accuracy, top1_rows, route_continuity_from_top1,
                              binary_confusion)
from pathsnap.evaluate import obs_metrics

OUT = RESULTS / "models"; OUT.mkdir(parents=True, exist_ok=True)
MDIR = ARTIFACTS / "models"; MDIR.mkdir(parents=True, exist_ok=True)
N_FOLDS = 5


def arcs_lookup(cands, base):
    out = {}
    for tid in cands.trip_id.unique():
        a = load_trip(base / tid, with_route=False).arcs
        out[tid] = (a.u.to_numpy(), a.v.to_numpy())
    return out


def evaluate(df, p, arcs, obs_tab=None):
    lab = (df.label_arc >= 0).to_numpy()
    dl = df[lab]
    m = candidate_metrics(dl.y.to_numpy(), p[lab])
    m["top1_acc"] = top1_accuracy(dl, p[lab])
    m["route_continuity"] = route_continuity_from_top1(df, p, arcs)
    if obs_tab is not None:                                    # obs-level macro-F1 over arc classes, per trip pooled
        r = top1_rows(df, p)
        pred = df.iloc[r][["trip_id", "obs", "arc"]]
        mm = obs_tab.merge(pred, on=["trip_id", "obs"], how="left").fillna({"arc": -1})
        f1s = [obs_metrics(g.arc.to_numpy(), g.label_arc.to_numpy())[1] for _, g in mm.groupby("trip_id")]
        m["obs_macro_f1_arcs"] = float(np.nanmean([f for f in f1s if f is not None]))
    return m


def main(only=None):
    tr_all = pd.read_pickle(RESULTS / "cache" / "train_cands.pkl")
    st = pd.read_pickle(RESULTS / "cache" / "student_test_cands.pkl")
    st_obs = pd.read_pickle(RESULTS / "cache" / "student_test_obs.pkl")
    arcs_tr, arcs_st = arcs_lookup(tr_all, TRAIN_DIR), arcs_lookup(st, STUDENT_TEST_DIR)
    trips = tr_all.trip_id.to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(tr_all, groups=trips))
    json.dump({"folds": [sorted(set(trips[v])) for _, v in folds]}, open(OUT / "cv_folds_trips.json", "w"), indent=1)
    Xall = tr_all[FEATURES].to_numpy(np.float32); yall = tr_all.y.to_numpy(); lab_all = (tr_all.label_arc >= 0).to_numpy()
    Xst = st[FEATURES].to_numpy(np.float32)

    fold_rows, st_rows, scores, confs, hist = [], [], {}, {}, {}
    names = only or list(MODELS)
    for name in names:
        print(f"=== {name}", flush=True)
        for k, (tri, vai) in enumerate(folds):
            tri = tri[lab_all[tri]]
            m, ft = fit_model(name, Xall[tri], yall[tri], rng=np.random.default_rng(SEED + k))
            t0 = time.perf_counter(); p = m.predict_proba(Xall[vai])[:, 1]; it = time.perf_counter() - t0
            va = tr_all.iloc[vai]
            r = evaluate(va, p, arcs_tr)
            r.update(model=name, fold=k, fit_s=ft, predict_s=it)
            p_tr = m.predict_proba(Xall[tri])[:, 1]                 # training-fold score -> over/under-fitting check
            trm = candidate_metrics(yall[tri], p_tr)
            r.update(train_macro_f1=trm["macro_f1"], train_log_loss=trm["log_loss"],
                     train_top1=top1_accuracy(tr_all.iloc[tri], p_tr))
            fold_rows.append(r)
            print(f"  fold {k}: top1={r['top1_acc']:.4f} mF1={r['macro_f1']:.4f} ll={r['log_loss']:.4f} "
                  f"train_top1={r['train_top1']:.4f} ({ft:.1f}s)", flush=True)
            if k == 0 and name in ("MLP", "LightGBM", "XGBoost"):
                hist[name] = training_history(name, Xall[tri], yall[tri], Xall[vai][lab_all[vai]], yall[vai][lab_all[vai]])

        # refit on all 80 train trips -> student-test
        m, ft = fit_model(name, Xall[lab_all], yall[lab_all])
        t0 = time.perf_counter(); p = m.predict_proba(Xst)[:, 1]; it = time.perf_counter() - t0
        r = evaluate(st, p, arcs_st, st_obs[st_obs.label_arc >= 0])
        r.update(model=name, fit_s=ft, predict_s=it, infer_ms_per_1k_obs=it * 1e6 / st_obs.shape[0])
        st_rows.append(r)
        scores[name] = p
        lab = (st.label_arc >= 0).to_numpy()
        confs[name] = binary_confusion(st.y.to_numpy()[lab], p[lab]).tolist()
        joblib.dump(m, MDIR / f"{name}.joblib", compress=3)
        print(f"  student-test: top1={r['top1_acc']:.4f} mF1={r['macro_f1']:.4f} ll={r['log_loss']:.4f} "
              f"cont={r['route_continuity']:.4f} infer={it:.2f}s", flush=True)

    tag = "" if only is None else "_" + "_".join(names)
    fdf = pd.DataFrame(fold_rows); fdf.to_csv(OUT / f"cv_folds{tag}.csv", index=False)
    sdf = pd.DataFrame(st_rows); sdf.to_csv(OUT / f"student_test{tag}.csv", index=False)
    pd.to_pickle({"scores": scores, "index": st[["trip_id", "obs", "arc", "y", "label_arc"]]},
                 RESULTS / "cache" / f"student_test_scores{tag}.pkl")
    json.dump(confs, open(OUT / f"confusion_student_test{tag}.json", "w"), indent=1)
    json.dump(hist, open(OUT / f"training_histories{tag}.json", "w"))
    json.dump({n: {"family": MODELS[n]["family"], "config": MODELS[n]["config"]} for n in names},
              open(OUT / f"model_configs{tag}.json", "w"), indent=1)

    cv = fdf.groupby("model").agg(cv_macro_f1=("macro_f1", "mean"), cv_macro_f1_std=("macro_f1", "std"),
                                  cv_top1=("top1_acc", "mean"), cv_top1_std=("top1_acc", "std"),
                                  cv_log_loss=("log_loss", "mean"), cv_continuity=("route_continuity", "mean"),
                                  train_top1=("train_top1", "mean"), train_macro_f1=("train_macro_f1", "mean"),
                                  cv_fit_s=("fit_s", "mean"))
    s = sdf.set_index("model").add_prefix("st_")
    comp = cv.join(s).loc[names]
    comp.insert(0, "family", [MODELS[n]["family"] for n in comp.index])
    comp.insert(1, "config", [MODELS[n]["config"] for n in comp.index])
    comp.to_csv(OUT / f"comparison{tag}.csv")
    pd.set_option("display.width", 250)
    print(comp[["family", "cv_macro_f1", "cv_top1", "cv_log_loss", "st_top1_acc", "st_macro_f1", "st_log_loss",
                "st_route_continuity", "st_obs_macro_f1_arcs", "st_predict_s"]].round(4).to_string())


def training_history(name, Xtr, ytr, Xva, yva):
    """Real per-iteration losses; never synthesised."""
    if name == "MLP":
        m = MODELS[name]["make"](); m.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        return {"x_label": "epoch", "x": m.history_["epoch"], "train": m.history_["train_log_loss"],
                "val": m.history_["val_log_loss"]}
    m = MODELS[name]["make"]()
    if name == "LightGBM":
        m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_metric="binary_logloss")
        ev = m.evals_result_
        return {"x_label": "boosting round", "x": list(range(1, len(ev["training"]["binary_logloss"]) + 1)),
                "train": ev["training"]["binary_logloss"], "val": ev["valid_1"]["binary_logloss"]}
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    ev = m.evals_result()
    return {"x_label": "boosting round", "x": list(range(1, len(ev["validation_0"]["logloss"]) + 1)),
            "train": ev["validation_0"]["logloss"], "val": ev["validation_1"]["logloss"]}


if __name__ == "__main__":
    main(sys.argv[1:] or None)
