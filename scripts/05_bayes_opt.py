"""Stage A3 - Bayesian hyper-parameter optimisation of the selected model (XGBoost) with Optuna.

* Sampler: TPE (Tree-structured Parzen Estimator) = Bayesian optimisation; seed 42; 10 random start-up trials.
* Objective: MAXIMISE mean observation-level Top-1 accuracy over the SAME GroupKFold(5) split of the 80 train trips
  used in Stage A2 (so tuned and untuned numbers are directly comparable). CV macro-F1 and log loss are recorded
  as user attributes of every trial.
* Pruning: MedianPruner after each fold (trials clearly worse than the median of earlier trials stop early).
* The student-test and evaluator partitions are NOT used during the search.
* After the search: refit best params on all 80 trips -> evaluate on student-test -> compare with untuned.
Outputs: results/bayes_opt/* (study.db search record, trials.csv, best_params.json, tuned_vs_untuned.csv, plots),
         artifacts/models/XGBoost_tuned.joblib
"""
import sys, json, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS, ARTIFACTS, SEED
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, top1_accuracy

spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "03_train_models.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)

OUT = RESULTS / "bayes_opt"; OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

SEARCH_SPACE = {   # documented for the report; mirrored exactly in suggest()
    "n_estimators": "int [100, 800], log-uniform",
    "learning_rate": "float [0.01, 0.3], log-uniform",
    "max_depth": "int [3, 12]",
    "min_child_weight": "float [1, 200], log-uniform",
    "subsample": "float [0.5, 1.0]",
    "colsample_bytree": "float [0.4, 1.0]",
    "gamma": "float [0, 5]",
    "reg_lambda": "float [1e-3, 50], log-uniform",
    "reg_alpha": "float [1e-3, 10], log-uniform",
}


def suggest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }


def make_xgb(params):
    return XGBClassifier(**params, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)


def main():
    tr_all = pd.read_pickle(RESULTS / "cache" / "train_cands.pkl")
    trips = tr_all.trip_id.to_numpy()
    folds = list(GroupKFold(5).split(tr_all, groups=trips))            # identical to Stage A2
    X = tr_all[FEATURES].to_numpy(np.float32); y = tr_all.y.to_numpy(); lab = (tr_all.label_arc >= 0).to_numpy()

    def objective(trial):
        params = suggest(trial)
        accs, f1s, lls = [], [], []
        t0 = time.perf_counter()
        for k, (tri, vai) in enumerate(folds):
            tri = tri[lab[tri]]; vai = vai[lab[vai]]
            m = make_xgb(params).fit(X[tri], y[tri])
            p = m.predict_proba(X[vai])[:, 1]
            accs.append(top1_accuracy(tr_all.iloc[vai], p))
            cm = candidate_metrics(y[vai], p); f1s.append(cm["macro_f1"]); lls.append(cm["log_loss"])
            trial.report(float(np.mean(accs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(accs, ddof=1)))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(accs))

    storage = f"sqlite:///{OUT / 'study.db'}"
    study = optuna.create_study(study_name="xgboost_top1", direction="maximize", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    remaining = N_TRIALS - len([t for t in study.trials if t.state.is_finished()])
    t_search = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    search_s = time.perf_counter() - t_search

    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df.to_csv(OUT / "trials.csv", index=False)
    best = study.best_trial
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    json.dump({"best_trial": best.number, "best_cv_top1": best.value, "params": best.params,
               "user_attrs": best.user_attrs, "n_trials": len(study.trials), "n_complete": n_complete,
               "n_pruned": n_pruned, "search_space": SEARCH_SPACE, "objective": "maximise mean GroupKFold(5) Top-1",
               "sampler": "TPE seed 42, 10 start-up trials", "pruner": "MedianPruner(8 start-up, 1 warm-up fold)",
               "search_seconds_this_run": search_s},
              open(OUT / "best_params.json", "w"), indent=1)
    print(f"best trial #{best.number}: CV top1 {best.value:.4f}  {best.params}")

    # ---------- tuned vs untuned on identical folds / partitions ----------
    st = pd.read_pickle(RESULTS / "cache" / "student_test_cands.pkl")
    st_obs = pd.read_pickle(RESULTS / "cache" / "student_test_obs.pkl")
    arcs_tr, arcs_st = train.arcs_lookup(tr_all, TRAIN_DIR), train.arcs_lookup(st, STUDENT_TEST_DIR)
    Xst = st[FEATURES].to_numpy(np.float32)
    rows, folds_rows = [], []
    from pathsnap.models import MODELS
    for name, factory in [("XGBoost (untuned)", MODELS["XGBoost"]["make"]), ("XGBoost (tuned)", lambda: make_xgb(best.params))]:
        cv = []
        for k, (tri, vai) in enumerate(folds):
            tri = tri[lab[tri]]
            m = factory(); t0 = time.perf_counter(); m.fit(X[tri], y[tri]); ft = time.perf_counter() - t0
            p = m.predict_proba(X[vai])[:, 1]
            r = train.evaluate(tr_all.iloc[vai], p, arcs_tr)
            ptr = m.predict_proba(X[tri])[:, 1]
            r.update(model=name, fold=k, fit_s=ft, train_top1=top1_accuracy(tr_all.iloc[tri], ptr)); cv.append(r)
        folds_rows += cv
        cvd = pd.DataFrame(cv)
        m = factory(); t0 = time.perf_counter(); m.fit(X[lab], y[lab]); ft = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(Xst)[:, 1]; it = time.perf_counter() - t0
        s = train.evaluate(st, p, arcs_st, st_obs[st_obs.label_arc >= 0])
        rows.append({"model": name, "cv_macro_f1": cvd.macro_f1.mean(), "cv_top1": cvd.top1_acc.mean(),
                     "cv_top1_std": cvd.top1_acc.std(), "cv_log_loss": cvd.log_loss.mean(),
                     "cv_continuity": cvd.route_continuity.mean(), "train_top1": cvd.train_top1.mean(),
                     "st_top1": s["top1_acc"], "st_macro_f1": s["macro_f1"], "st_log_loss": s["log_loss"],
                     "st_precision": s["precision"], "st_recall": s["recall"], "st_continuity": s["route_continuity"],
                     "st_obs_macro_f1_arcs": s["obs_macro_f1_arcs"], "fit_s_all80": ft, "st_inference_s": it,
                     "infer_ms_per_1k_obs": it * 1e6 / len(st_obs)})
        if name == "XGBoost (tuned)":
            joblib.dump(m, ARTIFACTS / "models" / "XGBoost_tuned.joblib", compress=3)
            pd.to_pickle({"scores": {"XGBoost_tuned": p}, "index": st[["trip_id", "obs", "arc", "y", "label_arc"]]},
                         RESULTS / "cache" / "student_test_scores_tuned.pkl")
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    pd.DataFrame(folds_rows).to_csv(OUT / "tuned_vs_untuned_folds.csv", index=False)
    print(pd.DataFrame(rows).set_index("model").T.round(4).to_string())


if __name__ == "__main__":
    main()
