"""Stage B7 - error analysis of the selected classifier (XGBoost) on the STUDENT-TEST partition (10 trips).

Where does observation-level Top-1 fail, and is it the kind of error HMM/Viterbi decoding can fix?
Breakdowns: (a) number of candidates, (b) speed, (c) heading reliability, (d) position on the true arc /
junction, (e) GPS distance to the true arc, (f) per trip; plus an error taxonomy ("right street, wrong
direction" = predicted arc is the reverse of the true arc, etc.). Every bucket also shows the nearest-arc
baseline so the model's gain is visible.

Scores: results/cache/student_test_scores.pkl (key "XGBoost", written by scripts/03_train_models.py) if present;
otherwise the saved model artifacts/models/XGBoost.joblib is applied to results/cache/student_test_cands.pkl
(build it with scripts/02_build_dataset.py). Only labelled observations (label_arc >= 0) are scored, exactly as
in Stage A2. Training data is not touched.

Run:  python analysis/error_analysis.py      ->  results/analysis/*.png, *.csv, error_analysis.md
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import STUDENT_TEST_DIR, RESULTS, ARTIFACTS
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.metrics import top1_rows

OUT = RESULTS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = RESULTS / "cache"
MODEL = "XGBoost"
DPI = 150
C_MODEL, C_BASE = "#1f6fd1", "#b8bfcc"
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def load_scores():
    cands = pd.read_pickle(CACHE / "student_test_cands.pkl")
    obs = pd.read_pickle(CACHE / "student_test_obs.pkl")
    sp = CACHE / "student_test_scores.pkl"
    if sp.exists():
        d = pd.read_pickle(sp)
        idx = d["index"]
        if len(idx) == len(cands) and (idx[["trip_id", "obs", "arc"]].to_numpy() == cands[["trip_id", "obs", "arc"]].to_numpy()).all():
            return cands, obs, np.asarray(d["scores"][MODEL]), "results/cache/student_test_scores.pkl (Stage A2 run)"
    import joblib
    m = joblib.load(ARTIFACTS / "models" / f"{MODEL}.joblib")
    p = m.predict_proba(cands[FEATURES].to_numpy(np.float32))[:, 1]
    return cands, obs, p, f"artifacts/models/{MODEL}.joblib applied to regenerated candidates"


def per_obs_table(cands, obs, p):
    """One row per observation: predicted arc + features of the predicted and of the true candidate."""
    r = top1_rows(cands, p)
    pred = cands.iloc[r][["trip_id", "obs", "arc", "n_cand", "speed", "head_reliab"]].rename(columns={"arc": "pred_arc"})
    pred["pred_p"] = p[r]
    t = obs.merge(pred, on=["trip_id", "obs"], how="left")
    true = cands[cands.y == 1][["trip_id", "obs", "dist", "t_along", "out_deg_v", "in_deg_u", "arc_len"]]
    t = t.merge(true.rename(columns={"dist": "true_dist", "t_along": "true_t", "out_deg_v": "true_out_deg",
                                     "in_deg_u": "true_in_deg", "arc_len": "true_arc_len"}),
                on=["trip_id", "obs"], how="left")
    t = t[t.label_arc >= 0].copy()                       # scored observations only (as in Stage A2)
    t["correct"] = t.pred_arc == t.label_arc
    t["base_correct"] = t.nearest_arc == t.label_arc

    # error taxonomy needs the trip's arc endpoints
    kind = np.full(len(t), "correct", dtype=object)
    for tid, g in t.groupby("trip_id"):
        a = load_trip(STUDENT_TEST_DIR / tid, with_route=False).arcs
        u, v = a.u.to_numpy(), a.v.to_numpy()
        pa, la = g.pred_arc.to_numpy().astype(int), g.label_arc.to_numpy().astype(int)
        k = np.where(pa == la, "correct",
            np.where((u[pa] == v[la]) & (v[pa] == u[la]), "reverse direction (same street)",
            np.where((v[pa] == u[la]) | (u[pa] == v[la]) | (u[pa] == u[la]) | (v[pa] == v[la]),
                     "adjacent arc (shares a node)", "other arc")))
        k = np.where((k != "correct") & ~g.in_cands.to_numpy(), "true arc not a candidate", k)
        kind[np.flatnonzero(t.trip_id.to_numpy() == tid)] = k
    t["error_kind"] = kind
    return t


def bucket_table(t, col, bins, labels, name):
    b = pd.cut(t[col], bins=bins, labels=labels, right=False)
    g = t.groupby(b, observed=False).agg(n_obs=("correct", "size"), top1=("correct", "mean"),
                                         baseline_top1=("base_correct", "mean"))
    g["share_obs"] = g.n_obs / g.n_obs.sum()
    g["share_errors"] = t.assign(b=b)[~t.correct].groupby("b", observed=False).size() / (~t.correct).sum()
    g.index.name = name
    return g


def bar(ax, g, title):
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.baseline_top1, 0.4, color=C_BASE, label="nearest-arc baseline")
    ax.bar(x + 0.2, g.top1, 0.4, color=C_MODEL, label=MODEL)
    for i, (n, v) in enumerate(zip(g.n_obs, g.top1)):
        if n:
            ax.text(i + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
            ax.text(i, 0.02, f"n={n:,}", ha="center", fontsize=6.5, color="#333")
    ax.set_xticks(x, [str(i) for i in g.index], fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Top-1 accuracy"); ax.set_title(title, fontsize=9)


def main():
    cands, obs, p, src = load_scores()
    t = per_obs_table(cands, obs, p)
    acc = t.correct.mean()
    print(f"scores from {src}; scored obs={len(t):,}; top-1={acc:.4f}; baseline={t.base_correct.mean():.4f}")

    tabs = {
        "n_candidates": bucket_table(t, "n_cand", [1, 4, 7, 10, 13], ["1-3", "4-6", "7-9", "10-12"], "candidates"),
        "speed": bucket_table(t, "speed", [0, 1, 8, np.inf], ["stopped <1 m/s", "slow 1-8 m/s", "fast >8 m/s"], "speed"),
        "heading_reliability": bucket_table(t, "head_reliab", [0, 0.25, 0.75, 1.01],
                                            ["low <0.25", "mid 0.25-0.75", "high >=0.75"], "heading reliability"),
        "position_on_arc": bucket_table(t.assign(edge=np.minimum(t.true_t, 1 - t.true_t)), "edge",
                                        [0, 0.1, 0.51], ["near a node (<10 % from end)", "mid-arc"], "position"),
        "junction": bucket_table(t, "true_out_deg", [0, 2, 3, 99], ["dead end / 1", "2 (plain road)", ">=3 (junction)"],
                                 "out-degree at end of true arc"),
        "distance_to_true_arc": bucket_table(t, "true_dist", [0, 5, 10, 20, 50.01], ["0-5 m", "5-10 m", "10-20 m", "20-50 m"],
                                             "GPS distance to true arc"),
    }
    for k, g in tabs.items():
        g.to_csv(OUT / f"by_{k}.csv")

    per_trip = t.groupby("trip_id").agg(n_obs=("correct", "size"), top1=("correct", "mean"),
                                        baseline_top1=("base_correct", "mean"),
                                        reverse_errors=("error_kind", lambda s: (s == "reverse direction (same street)").sum()))
    per_trip.to_csv(OUT / "by_trip.csv")
    kinds = t[~t.correct].error_kind.value_counts()
    kinds_df = pd.DataFrame({"n": kinds, "share_of_errors": kinds / kinds.sum()})
    kinds_df.to_csv(OUT / "error_kinds.csv")

    # ---- figures
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    titles = {"n_candidates": "(a) candidates per observation", "speed": "(b) speed",
              "heading_reliability": "(c) heading reliability", "position_on_arc": "(d1) position on true arc",
              "junction": "(d2) junction at end of true arc", "distance_to_true_arc": "(e) GPS distance to true arc"}
    for ax, (k, g) in zip(axes.ravel(), tabs.items()):
        bar(ax, g, titles[k])
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"{MODEL} Top-1 accuracy by condition – STUDENT-TEST partition (10 trips, {len(t):,} labelled obs)")
    fig.tight_layout(); fig.savefig(OUT / "fig_b7_breakdowns.png", dpi=DPI); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    s = per_trip.sort_values("top1")
    x = np.arange(len(s))
    axes[0].bar(x - 0.2, s.baseline_top1, 0.4, color=C_BASE, label="nearest-arc baseline")
    axes[0].bar(x + 0.2, s.top1, 0.4, color=C_MODEL, label=MODEL)
    axes[0].set_xticks(x, s.index.str[-2:]); axes[0].set_xlabel("trip id (last 2 digits)")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Top-1 accuracy"); axes[0].legend(fontsize=7, loc="lower right")
    axes[0].set_title("(f) per trip – STUDENT-TEST", fontsize=9)
    axes[1].barh(kinds_df.index[::-1], kinds_df.share_of_errors[::-1] * 100, color="#d62728")
    for i, (n, v) in enumerate(zip(kinds_df.n[::-1], kinds_df.share_of_errors[::-1])):
        axes[1].text(v * 100 + 0.5, i, f"{v*100:.1f} % ({n:,})", va="center", fontsize=8)
    axes[1].set_xlabel("% of Top-1 errors"); axes[1].set_xlim(0, kinds_df.share_of_errors.max() * 125)
    axes[1].set_title(f"Error taxonomy – {int((~t.correct).sum()):,} errors, STUDENT-TEST", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_b7_trips_and_error_types.png", dpi=DPI); plt.close(fig)

    # ---- markdown
    def tbl(g):
        d = g.copy()
        for c in ["top1", "baseline_top1", "share_obs", "share_errors"]:
            d[c] = (d[c] * 100).map(lambda v: "–" if pd.isna(v) else f"{v:.1f} %")
        return d[["n_obs", "share_obs", "baseline_top1", "top1", "share_errors"]].to_markdown()

    rev = kinds_df.loc["reverse direction (same street)"] if "reverse direction (same street)" in kinds_df.index else None
    worst = per_trip.top1.idxmin()
    lines = [
        "# B7 – Error analysis of the selected classifier (XGBoost)",
        "",
        f"Partition: **student-test** (10 trips), {len(t):,} labelled observations. Scores: {src}.",
        f"Top-1 = {acc*100:.2f} % (nearest-arc baseline {t.base_correct.mean()*100:.2f} %); "
        f"{int((~t.correct).sum()):,} wrong observations. Produced by `analysis/error_analysis.py`.",
        "",
        "Columns: share of observations in the bucket, baseline and model Top-1, share of all model errors in the bucket.",
        "",
        *sum([[f"## {titles[k]}", tbl(g), ""] for k, g in tabs.items()], []),
        "## (f) per trip",
        per_trip.assign(top1=lambda d: (d.top1 * 100).round(1), baseline_top1=lambda d: (d.baseline_top1 * 100).round(1)).to_markdown(),
        "",
        "## Error taxonomy",
        kinds_df.assign(share_of_errors=lambda d: (d.share_of_errors * 100).round(1)).to_markdown(),
        "",
        "## Findings",
        "FINDINGS_PLACEHOLDER",
        "",
        "Figures: `fig_b7_breakdowns.png`, `fig_b7_trips_and_error_types.png`. CSVs: `by_*.csv`, `error_kinds.csv`.",
    ]
    md = "\n".join(lines)
    findings = make_findings(t, tabs, per_trip, kinds_df)
    (OUT / "error_analysis.md").write_text(md.replace("FINDINGS_PLACEHOLDER", findings), encoding="utf-8")
    print(findings)


def flip_runs(t, kind):
    """Lengths of consecutive runs (within a trip, in time order) of observations with the given error kind."""
    L = []
    for _, g in t.sort_values(["trip_id", "obs"]).groupby("trip_id"):
        e = (g.error_kind == kind).to_numpy().astype(int)
        d = np.diff(np.r_[0, e, 0])
        L += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return np.asarray(L)


def make_findings(t, tabs, per_trip, kinds):
    pct = lambda v: f"{v*100:.1f} %"
    sp, hr, nc = tabs["speed"], tabs["heading_reliability"], tabs["n_candidates"]
    pa, jn, dd = tabs["position_on_arc"], tabs["junction"], tabs["distance_to_true_arc"]
    err = t[~t.correct]
    trip_err = err.trip_id.value_counts() / len(err)
    w = trip_err.index[0]
    out = [
        f"* **Speed / heading is the main driver.** Above 8 m/s Top-1 is {pct(sp.iloc[2].top1)}; at 1–8 m/s it drops to "
        f"{pct(sp.iloc[1].top1)}, and these slow observations ({pct(sp.iloc[1].share_obs)} of all) hold "
        f"{pct(sp.iloc[1].share_errors)} of the errors. Fully stopped points are worst ({pct(sp.iloc[0].top1)}) but rare "
        f"({pct(sp.iloc[0].share_obs)}). By heading reliability: {pct(hr.iloc[0].top1)} (low) / {pct(hr.iloc[1].top1)} (mid) / "
        f"{pct(hr.iloc[2].top1)} (high) – when the vehicle moves too little, the heading features cannot tell the two "
        "directions of a street apart.",
    ]
    if "reverse direction (same street)" in kinds.index:
        r = kinds.loc["reverse direction (same street)"]
        L = flip_runs(t, "reverse direction (same street)")
        out.append(f"* **{pct(r.share_of_errors)} of all errors ({int(r.n):,}) are 'right street, wrong direction'** – the "
                   f"predicted arc is the reverse arc of the true one. They are short flips, not long wrong stretches: "
                   f"{len(L)} runs, median length {np.median(L):.0f} observation(s), longest {L.max()}.")
    if "adjacent arc (shares a node)" in kinds.index:
        a = kinds.loc["adjacent arc (shares a node)"]
        out.append(f"* {pct(a.share_of_errors)} of errors choose an arc that shares a node with the true arc (previous/next "
                   f"segment or a side road). Consistently, Top-1 is {pct(pa.iloc[0].top1)} within 10 % of an arc end vs "
                   f"{pct(pa.iloc[1].top1)} mid-arc, and {pct(jn.iloc[2].top1)} when the true arc ends in a junction (out-degree ≥ 3).")
    if "other arc" in kinds.index:
        out.append(f"* Only {pct(kinds.loc['other arc'].share_of_errors)} of errors are unrelated arcs, and "
                   f"{int(kinds.loc['true arc not a candidate'].n) if 'true arc not a candidate' in kinds.index else 0} errors "
                   "are candidate-generation misses (the true arc was not within 50 m / top-12).")
    out.append(f"* **One trip dominates**: trip {w} holds {pct(trip_err.iloc[0])} of all errors (Top-1 {pct(per_trip.top1.min())}; "
               f"{int(per_trip.loc[w,'reverse_errors'])} reverse-direction errors). It is a slow trip "
               f"(median speed {t[t.trip_id==w].speed.median():.1f} m/s). The other nine trips range "
               f"{pct(per_trip.drop(w).top1.min())}–{pct(per_trip.top1.max())}.")
    out.append(f"* Dense networks are harder: {pct(nc.iloc[0].top1)} with 1–3 candidates vs {pct(nc.iloc[3].top1)} with 10–12 "
               f"(the latter are {pct(nc.iloc[3].share_obs)} of observations and {pct(nc.iloc[3].share_errors)} of errors).")
    out.append(f"* GPS distance matters less: {pct(dd.iloc[0].top1)} at 0–5 m, {pct(dd.iloc[2].top1)} at 10–20 m; only the 20–50 m "
               f"bucket ({pct(dd.iloc[3].share_obs)} of observations) falls to {pct(dd.iloc[3].top1)}.")
    out.append(f"* The classifier gains most where the nearest-arc baseline fails: overall {pct(t.base_correct.mean())} → "
               f"{pct(t.correct.mean())}; at junctions {pct(jn.iloc[2].baseline_top1)} → {pct(jn.iloc[2].top1)}.")
    out.append("* **Implication for the HMM (§5):** most remaining errors are short direction flips or jumps to a "
               "neighbouring arc while the street itself is right. A transition model that only allows connected, "
               "consistently directed arc sequences should remove many of them, especially on slow trips like "
               f"{w}; it cannot help with the few candidate-generation misses.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
