"""Stage A2 plots - every figure is drawn from the CSV/JSON files written by 03_train_models.py.
Training curves are drawn ONLY for models with a recorded per-iteration history (MLP, LightGBM, XGBoost)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS

IN, OUT = RESULTS / "models", RESULTS / "models" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("wrote", OUT / name)


def hbar(ax, labels, vals, err=None, best=None, fmt="{:.4f}", xlabel=""):
    y = np.arange(len(labels))
    cols = [BLUE if l == best else GREY for l in labels]
    ax.barh(y, vals, xerr=err, color=cols, height=0.62, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    for yi, v in zip(y, vals):
        ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=8.5, color=INK)


def main():
    comp = pd.read_csv(IN / "comparison.csv", index_col=0)
    folds = pd.read_csv(IN / "cv_folds.csv")
    best = comp.cv_top1.idxmax()

    # 1 CV macro-F1 with fold std
    c = comp.sort_values("cv_macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    hbar(ax, list(c.index), c.cv_macro_f1.values, c.cv_macro_f1_std.values, best, xlabel="Macro-F1 (mean of 5 GroupKFold folds, ± std)")
    ax.set_xlim(0.7, 1.0); ax.set_title("Cross-validation macro-F1 (candidate classification) - TRAIN partition, 80 trips", fontsize=10, loc="left")
    save(fig, "01_cv_macro_f1.png")

    # 2 student-test Top-1 and macro-F1
    c = comp.sort_values("st_top1_acc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(c)); w = 0.38
    ax.bar(x - w / 2 - 0.01, c.st_top1_acc, w, color=BLUE, label="Top-1 accuracy (observation level)")
    ax.bar(x + w / 2 + 0.01, c.st_macro_f1, w, color=ORANGE, label="Macro-F1 (candidate level)")
    for xi, a, b in zip(x, c.st_top1_acc, c.st_macro_f1):
        ax.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=7.5); ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, c.index, rotation=20, ha="right"); ax.set_ylim(0.6, 1.0); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_title("STUDENT-TEST partition (10 trips): Top-1 accuracy and macro-F1", fontsize=10, loc="left")
    save(fig, "02_student_test_top1_macro_f1.png")

    # 3 log loss, CV vs student-test
    c = comp.sort_values("cv_log_loss")
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(c))
    ax.bar(x - w / 2 - 0.01, c.cv_log_loss, w, color=BLUE, label="CV (train partition)")
    ax.bar(x + w / 2 + 0.01, c.st_log_loss, w, color=ORANGE, label="Student-test partition")
    ax.set_yscale("log"); ax.set_ylabel("Log loss (log scale, lower is better)")
    ax.set_xticks(x, c.index, rotation=20, ha="right"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8.5)
    for xi, a, b in zip(x, c.cv_log_loss, c.st_log_loss):
        ax.text(xi - w / 2, a * 1.05, f"{a:.3f}", ha="center", fontsize=7); ax.text(xi + w / 2, b * 1.05, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_title("Log loss of candidate probabilities", fontsize=10, loc="left")
    save(fig, "03_log_loss.png")

    # 4 fold stability (Top-1 per fold)
    order = comp.sort_values("cv_top1", ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, m in enumerate(order):
        v = folds[folds.model == m].top1_acc.values
        ax.plot([i] * len(v), v, "o", color=BLUE if m == best else MUTED, ms=6, alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right"); ax.set_ylabel("Top-1 accuracy on held-out fold")
    ax.grid(axis="x", visible=False)
    ax.set_title("Stability across the 5 GroupKFold folds (dots = folds, bar = mean) - TRAIN partition", fontsize=10, loc="left")
    save(fig, "04_cv_fold_stability.png")

    # 5 train vs CV (over/underfitting)
    c = comp.loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(c))
    ax.bar(x - w / 2 - 0.01, c.train_top1, w, color=GREY, label="Training folds (fit data)")
    ax.bar(x + w / 2 + 0.01, c.cv_top1, w, color=BLUE, label="Held-out folds (CV)")
    ax.set_ylim(0.6, 1.02); ax.set_xticks(x, c.index, rotation=20, ha="right"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); ax.set_ylabel("Top-1 accuracy")
    for xi, a, b in zip(x, c.train_top1, c.cv_top1):
        ax.text(xi, max(a, b) + 0.006, f"gap {a - b:+.3f}", ha="center", fontsize=7, color=MUTED)
    ax.set_title("Over/under-fitting check: training-fold vs held-out-fold Top-1", fontsize=10, loc="left")
    save(fig, "05_train_vs_cv_top1.png")

    # 6 inference time vs accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, r in comp.iterrows():
        ax.scatter(r.st_infer_ms_per_1k_obs, r.st_top1_acc, s=60, color=BLUE if m == best else MUTED, zorder=3,
                   edgecolor="white", linewidth=2)
        ax.annotate(m, (r.st_infer_ms_per_1k_obs, r.st_top1_acc), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Inference time, ms per 1,000 GPS observations (log scale)"); ax.set_ylabel("Student-test Top-1")
    ax.set_title("Accuracy vs inference cost (student-test partition)", fontsize=10, loc="left")
    save(fig, "06_accuracy_vs_inference_time.png")

    # 7 confusion matrices (row-normalised)
    conf = json.load(open(IN / "confusion_student_test.json"))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, m in zip(axes.flat, order):
        cm = np.array(conf[m]); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(m, fontsize=9); ax.grid(False)
    fig.suptitle("Confusion matrices, candidate level (threshold 0.5) - STUDENT-TEST partition", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); save(fig, "07_confusion_matrices.png")

    # 8 training curves (real histories only)
    hist = json.load(open(IN / "training_histories.json"))
    if hist:
        fig, axes = plt.subplots(1, len(hist), figsize=(4.2 * len(hist), 3.4))
        for ax, (m, h) in zip(np.atleast_1d(axes), hist.items()):
            ax.plot(h["x"], h["train"], color=BLUE, lw=2, label="training")
            ax.plot(h["x"], h["val"], color=ORANGE, lw=2, label="validation (held-out trips)")
            ax.set_xlabel(h["x_label"]); ax.set_ylabel("log loss"); ax.set_title(m, fontsize=9.5, loc="left")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Training vs validation loss (CV fold 0: 64 train trips / 16 validation trips)", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(); save(fig, "08_training_curves.png")


if __name__ == "__main__":
    main()
