"""Stage A4 plots from results/hmm/*.csv (real decodes only)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS, ARTIFACTS

IN = RESULTS / "hmm"; OUT = IN / "plots"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b9b8b2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("wrote", OUT / name)


def main():
    cfg = json.load(open(ARTIFACTS / "final" / "hmm_config.json")); em = cfg["emission_model"]
    grid = pd.read_csv(IN / "train_oof_grid.csv")

    # 1 decoder grid: only lambda = w_e * beta changes the Viterbi path, so plot Top-1 against lambda
    g = grid[grid.decoder == "hmm"].copy(); g["lam"] = (g.w_e * g.beta).round(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for m, col in [("XGBoost_untuned", BLUE), ("XGBoost_tuned", ORANGE)]:
        d = g[(g.model == m) & (g.gap_s == 60.0)].groupby("lam").top1_acc.max()
        ax.plot(d.index, d.values, "o-", color=col, lw=2, ms=5, label=f"{m} emissions (gap_s = 60 s)")
    clf = grid[(grid.decoder == "classifier_only") & (grid.model == em)].top1_acc.iloc[0]
    ax.axhline(clf, color=INK, ls="--", lw=1, label=f"classifier only ({em})")
    ax.set_xscale("log"); ax.set_xlabel("lambda = w_e x beta  (weight of classifier vs road-network transition)")
    ax.set_ylabel("pooled Top-1"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Decoder grid search on TRAIN out-of-fold emissions (80 trips)", fontsize=9.5, loc="left")
    ax = axes[1]
    d = g[(g.model == em) & (g.beta == cfg["beta"]) & (g.w_e == cfg["w_e"])].sort_values("gap_s")
    ax.bar([f"{x:g} s" for x in d.gap_s], d.top1_acc, color=[BLUE if x == cfg["gap_s"] else GREY for x in d.gap_s], width=0.6)
    for i, v in enumerate(d.top1_acc):
        ax.text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylim(d.top1_acc.min() - 0.004, d.top1_acc.max() + 0.002); ax.grid(axis="x", visible=False)
    ax.set_xlabel("gap_s (network constraint dropped above)"); ax.set_title(f"Gap threshold, lambda = {cfg['w_e'] * cfg['beta']:g}", fontsize=9.5, loc="left")
    fig.tight_layout(); save(fig, "01_decoder_grid.png")

    # 2 classifier-only vs HMM summary (train OOF and student-test)
    tr = pd.read_csv(IN / "train_oof_summary.csv"); st = pd.read_csv(IN / "student_test_summary.csv")
    mets = [("top1_acc", "Top-1"), ("macro_f1_arcs", "Obs macro-F1\n(arc classes)"), ("route_f1", "Route F1"),
            ("length_recall", "Route length\nrecall"), ("route_continuity", "Route\ncontinuity")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (df, title) in zip(axes, [(tr, "TRAIN partition (80 trips, out-of-fold)"), (st, "STUDENT-TEST partition (10 trips)")]):
        d = df[df.model == em].set_index("decoder")
        x = np.arange(len(mets)); w = 0.38
        a = [d.loc["classifier_only", m] for m, _ in mets]; b = [d.loc["hmm", m] for m, _ in mets]
        ax.bar(x - w / 2 - 0.01, a, w, color=ORANGE, label="classifier only (argmax per observation)")
        ax.bar(x + w / 2 + 0.01, b, w, color=BLUE, label="classifier + HMM/Viterbi")
        for xi, va, vb in zip(x, a, b):
            ax.text(xi - w / 2, va + 0.005, f"{va:.3f}", ha="center", fontsize=7.5); ax.text(xi + w / 2, vb + 0.005, f"{vb:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(x, [l for _, l in mets], fontsize=8.5); ax.set_ylim(0.5, 1.05); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Does HMM decoding help? ({em} emissions)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    save(fig, "02_classifier_vs_hmm.png")

    # 3 per-trip student-test Top-1 and route F1
    pt = pd.read_csv(IN / "student_test_per_trip.csv", dtype={"trip_id": str})
    pt = pt[pt.model == em].copy()
    pt["top1"] = pt.n_correct / pt.n_lab
    p = pt.tp / pt.n_pred_arcs; r = pt.tp / pt.n_ref_arcs; pt["route_f1"] = 2 * p * r / (p + r)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, m, lab in [(axes[0], "top1", "Top-1 accuracy"), (axes[1], "route_f1", "Route F1")]:
        piv = pt.pivot(index="trip_id", columns="decoder", values=m)
        x = np.arange(len(piv)); w = 0.38
        ax.bar(x - w / 2 - 0.01, piv["classifier_only"], w, color=ORANGE, label="classifier only")
        ax.bar(x + w / 2 + 0.01, piv["hmm"], w, color=BLUE, label="classifier + HMM")
        ax.set_xticks(x, [t[-2:] for t in piv.index]); ax.set_xlabel("student-test trip (last 2 digits of id)")
        ax.set_ylabel(lab); ax.set_ylim(0.5, 1.02); ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Per-trip effect of the HMM - STUDENT-TEST partition", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); save(fig, "03_per_trip_student_test.png")


if __name__ == "__main__":
    main()
