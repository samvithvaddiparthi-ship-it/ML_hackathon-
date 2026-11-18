# Brief for Account B – Part 2 (paste into Aditya's Claude session)

B1–B5 are done – great work. Account A is now on Stage A3 (Optuna tuning of XGBoost), then A4 (HMM + Viterbi) and
A5 (final pipeline export). To balance the load, B takes over most of the **report writing, error analysis,
HMM-map tooling and demo preparation**. Same rules as before: only edit your folders (`eda/`, `viz/`, `api/`, `ui/`,
`deploy/`, `report/`, `analysis/` (new), `results/eda/`, `results/maps/`, `results/analysis/` (new), `README.md`),
work on `app-eda`, stop for the user's OK after each stage, update `docs/STATUS_B.md`.

## Setup
`git fetch origin && git merge origin/ml-core` into `app-eda` (folders are disjoint → no conflicts). You now have
A's Stage A1–A2 code and results: `results/models/comparison.csv`, `cv_folds.csv`, `student_test.csv`,
`confusion_student_test.json`, `training_histories.json`, `results/models/plots/*.png`, `docs/STATUS_A.md`
(model table + selection reasoning), `src/pathsnap/features.py` (25 features documented in the docstring).
The cached candidate tables are git-ignored – regenerate with
`.venv/bin/python scripts/02_build_dataset.py` (~10 s) and, if you need per-candidate scores,
`.venv/bin/python scripts/03_train_models.py` (~12 min, writes `results/cache/student_test_scores.pkl`).

## Stage B6 – Report §3 Model experimentation (biggest writing chunk)
In `report/report.md`:
* §3.1 consolidated table (all 8 models) from `comparison.csv`: family, key config, CV macro-F1, CV/ST Top-1,
  route continuity, ST macro-F1, log loss, inference time (`st_infer_ms_per_1k_obs`). Note Macro-F1 = candidate-level
  binary macro-F1; Top-1 = observation level; KNN fit on 150k-row subsample (say so).
* §3.2 one subsection per model (8) answering EVERY compulsory question: A1 why suitable, A2 strengths/limits,
  B5 hyper-parameters & initial values (`model_configs.json` / `src/pathsnap/models.py`), B6 loss/objective,
  C7 student-test performance, C8 Top-1 / macro-F1 / log loss, C9 over/underfitting with evidence
  (`train_top1` vs `cv_top1`, fold std, the loss curves for MLP/LightGBM/XGBoost). Tie every claim to an actual
  number – no textbook-only text. Observations already visible in the results: GaussianNB unstable (fold std 0.068,
  fold 4 = 0.66) because features are strongly correlated; KNN slowest inference (~800 ms / 1k obs);
  boosting ~5-pt train/CV gap and validation loss flattening ~round 120; MLP val loss plateaus after ~6 epochs.
* §3.3 embed all 8 plots from `results/models/plots/` with captions stating the partition.
* Primary-metric paragraph: CV Top-1 (route is built from per-observation Top-1), log loss as tie-break because
  probabilities become HMM emissions.

## Stage B7 – Error analysis (new, strengthens §3 and §5) → `analysis/error_analysis.py`, `results/analysis/`
Using the selected XGBoost scores on student-test (`results/cache/student_test_scores.pkl`, key `XGBoost`, joined
with `student_test_cands.pkl` / `student_test_obs.pkl`): Top-1 accuracy broken down by
(a) number of candidates, (b) speed bucket (stopped / slow / fast), (c) heading reliability, (d) junction vs mid-arc
(`out_deg_v`), (e) distance to true arc, (f) per trip; plus share of errors that are "right street, wrong
direction" (predicted arc is the reverse of the label). Plots + a short markdown of findings. These failure modes
are exactly what the HMM should fix, so A will use this in §5.

## Stage B8 – HMM comparison map tooling → `viz/make_route_maps.py`
CLI: `python viz/make_route_maps.py --trip-dir <dir> --out results/maps/<id>_routes.html` calling
`predict_trip(..., use_hmm=True)`; draw GPS, nearby network, reference route (report only – student-test trips),
classifier-only route (`route_coords_classifier_only`, orange dashed) and HMM route (red), legend + LayerControl,
auto PNG screenshot. Must work NOW with the baseline (classifier-only key missing → just skip that layer) so it is
ready the moment A4 lands. Also a `--all-student-test` flag that makes one map per student-test trip.

## Stage B9 – Report §2.2, §6, §7 polish + build
Finish §2.2 from your EDA (preprocessing answer, patterns, why the 25 features – use the features docstring and
`results/stage1/quality_issues.md`), write §6 around B8 maps, §7 architecture (diagram of Laptop 1 API ↔ Laptop 2 UI,
endpoints, logging, leakage safeguards: `.route` never read in `predict_trip`, UI never receives reference arcs).
Title page: team names/SRNs, Section H, AI-use disclosure. Make `report/build_report.py` produce the PDF cleanly.

## Stage B10 – Demo & viva preparation
* Real two-laptop rehearsal on the same hotspot: run API on Laptop 1, UI on Laptop 2, all 10 student-test trips;
  take the §7 screenshots (UI + Laptop 1 console logs). Write `deploy/DEMO_CHECKLIST.md` (exact steps, IPs,
  firewall, where the evaluator trips folder goes, fallback if Wi-Fi fails).
* `docs/VIVA_QA.md`: ~40 likely evaluator questions with short answers grounded in our files (dataset structure,
  label alignment, AEQD projection, candidate radius, each feature group, why XGBoost, GroupKFold, Optuna/TPE,
  HMM states/emissions/transitions/Viterbi, gaps, leakage prevention, API/UI). Every team member should read it.

Hand-back after each stage: commit, push `app-eda`, update `docs/STATUS_B.md`.
