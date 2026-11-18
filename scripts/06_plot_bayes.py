"""Stage A3 plots from the recorded Optuna study (results/bayes_opt/study.db) and tuned_vs_untuned.csv."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS

IN = RESULTS / "bayes_opt"; OUT = IN / "plots"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b9b8b2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("wrote", OUT / name)


def main():
    study = optuna.load_study(study_name="xgboost_top1", storage=f"sqlite:///{IN / 'study.db'}")
    trials = study.trials
    comp = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    untuned = pd.read_csv(IN / "tuned_vs_untuned.csv").set_index("model")

    # 1 optimisation history
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter([t.number for t in comp], [t.value for t in comp], color=BLUE, s=28, label="completed trial (mean CV Top-1)", zorder=3)
    ax.scatter([t.number for t in pruned], [max(t.intermediate_values.values()) for t in pruned], marker="x", color=GREY,
               s=28, label="pruned trial (last reported fold mean)", zorder=3)
    xs = sorted(comp, key=lambda t: t.number); best = np.maximum.accumulate([t.value for t in xs])
    ax.step([t.number for t in xs], best, where="post", color=ORANGE, lw=2, label="best so far")
    ax.axhline(untuned.loc["XGBoost (untuned)", "cv_top1"], color=INK, ls="--", lw=1, label="untuned XGBoost (Stage A2 config)")
    ax.set_xlabel("trial"); ax.set_ylabel("mean GroupKFold(5) Top-1"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Bayesian optimisation (Optuna TPE) history - TRAIN partition, 80 trips", fontsize=10, loc="left")
    save(fig, "01_optimisation_history.png")

    # 2 hyper-parameter importance (fANOVA on completed trials)
    imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(s.index, s.values, color=BLUE, height=0.6); ax.grid(axis="y", visible=False)
    for i, v in enumerate(s.values):
        ax.text(v, i, f"  {v:.2f}", va="center", fontsize=8.5)
    ax.set_xlabel("fANOVA importance (share of objective variance)")
    ax.set_title("Which hyper-parameters mattered", fontsize=10, loc="left")
    save(fig, "02_param_importance.png")
    pd.Series(imp).to_csv(IN / "param_importance.csv", header=["importance"])

    # 3 slices for the two most important params
    top2 = list(pd.Series(imp).sort_values(ascending=False).index[:2])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, p in zip(axes, top2):
        ax.scatter([t.params[p] for t in comp], [t.value for t in comp], color=BLUE, s=25)
        ax.set_xlabel(p); ax.set_ylabel("mean CV Top-1")
        if p in ("learning_rate", "min_child_weight", "reg_lambda", "reg_alpha", "n_estimators"):
            ax.set_xscale("log")
    fig.suptitle("Objective vs the two most important hyper-parameters", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); save(fig, "03_param_slices.png")

    # 4 tuned vs untuned
    metrics = [("cv_top1", "CV Top-1"), ("cv_macro_f1", "CV macro-F1"), ("st_top1", "Student-test Top-1"),
               ("st_macro_f1", "Student-test macro-F1")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]; x = np.arange(len(metrics)); w = 0.38
    u = [untuned.loc["XGBoost (untuned)", m] for m, _ in metrics]; t = [untuned.loc["XGBoost (tuned)", m] for m, _ in metrics]
    ax.bar(x - w / 2 - 0.01, u, w, color=GREY, label="untuned"); ax.bar(x + w / 2 + 0.01, t, w, color=BLUE, label="tuned")
    for xi, a, b in zip(x, u, t):
        ax.text(xi - w / 2, a + 0.002, f"{a:.4f}", ha="center", fontsize=7.5); ax.text(xi + w / 2, b + 0.002, f"{b:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [l for _, l in metrics]); ax.set_ylim(0.85, 0.96); ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="x", visible=False)
    ax = axes[1]
    for i, (m, l) in enumerate([("cv_log_loss", "CV"), ("st_log_loss", "Student-test")]):
        ax.bar(i - 0.2, untuned.loc["XGBoost (untuned)", m], 0.38, color=GREY); ax.bar(i + 0.2, untuned.loc["XGBoost (tuned)", m], 0.38, color=BLUE)
    ax.set_xticks([0, 1], ["CV", "Student-test"]); ax.set_ylabel("log loss (lower better)"); ax.grid(axis="x", visible=False)
    fig.suptitle("Untuned vs tuned XGBoost (same folds, same partitions)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); save(fig, "04_tuned_vs_untuned.png")


if __name__ == "__main__":
    main()
