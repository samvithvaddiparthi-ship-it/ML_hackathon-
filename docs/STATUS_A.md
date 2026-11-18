# Status – Account A (ML core, branch `ml-core`)

## Stage A1 – labels, candidates, features ✅
Run: `.venv/bin/python scripts/02_build_dataset.py` (~10 s on 8 cores) → `results/cache/*.pkl` (git-ignored) + `results/dataset/*.csv`.

* **Per-observation labels** (`src/pathsnap/labels.py`): the `.route` file is an arc sequence, not one arc per GPS point,
  so GPS points are aligned to the route by a monotone DP (DTW-style). Points > 30 m from their aligned arc → label -1
  (excluded from training/scoring). Train: 87.5 % of observations labelled (trips 54 / 33 only 10 % / 29 % –
  truncated reference routes), student-test: 99.2 %.
* **Candidates** (`geo.RoadNet.candidates`): arcs within 50 m, max 12 per observation, ~6.6 candidates/obs.
  Candidate miss rate (true arc not in the set): **0.19 % train, 0.04 % student-test**.
* **Features** (`src/pathsnap/features.py`, 25 features): distance/rank/gaussian likelihood, heading cosine (direction
  of two-way roads), windowed distances of neighbouring points to the same arc, speed/dt/jump, degrees, continuity
  with neighbours' nearest arc. Train 1.38 M candidate rows (13 % positive), student-test 147 k rows.
* **Baseline**: nearest arc = **56.2 % / 54.0 %** top-1 (train / student-test) – low mainly because both directions
  of a two-way road are equally near. A quick untuned LightGBM sanity run already gives 89.5 % student-test top-1.
* `evaluate.score_trip` now returns real `top1_accuracy` and `macro_f1` (keys unchanged; added `n_scored_obs`).

## Next: Stage A2 – train & compare ≥ 5 models (GroupKFold over trips, student-test comparison, plots).

## Stage A2 – model comparison ✅
Run: `.venv/bin/python scripts/03_train_models.py` (~12 min) then `scripts/04_plot_models.py`.
8 models, GroupKFold(5) on 80 train trips + refit → 10 student-test trips. Results: `results/models/comparison.csv`,
`cv_folds.csv`, `student_test.csv`, `confusion_student_test.json`, `training_histories.json`, plots in `results/models/plots/`.

| Model | CV top-1 (mean of 5 folds ± std) | CV macro-F1 (mean ± std) | CV log loss (mean) | ST top-1 | ST macro-F1 | ST log loss | ST continuity |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.8958 ± 0.0177** | 0.9366 ± 0.0094 | 0.0759 | 0.9015 | 0.9408 | 0.0642 | 0.836 |
| LightGBM | 0.8953 ± 0.0183 | 0.9366 ± 0.0095 | 0.0762 | 0.9034 | 0.9407 | 0.0640 | 0.832 |
| RandomForest | 0.8945 ± 0.0162 | 0.9340 ± 0.0095 | 0.0798 | 0.8946 | 0.9354 | 0.0687 | 0.826 |
| DecisionTree | 0.8891 ± 0.0180 | 0.9312 ± 0.0089 | 0.0931 | 0.8941 | 0.9329 | 0.0768 | 0.818 |
| MLP | 0.8889 ± 0.0169 | 0.9295 ± 0.0091 | 0.0849 | 0.8941 | 0.9328 | 0.0723 | 0.816 |
| LogisticRegression | 0.8749 ± 0.0167 | 0.9128 ± 0.0075 | 0.1114 | 0.8631 | 0.9111 | 0.1004 | 0.782 |
| KNN | 0.8734 ± 0.0177 | 0.9161 ± 0.0100 | 0.1317 | 0.8664 | 0.9152 | 0.1194 | 0.807 |
| GaussianNB | 0.7770 ± 0.0682 | 0.7803 ± 0.0135 | 1.3767 | 0.7537 | 0.7827 | 1.2477 | 0.800 |

CV columns = mean over the 5 GroupKFold folds (per-fold values: `results/models/cv_folds.csv`); ST = student-test (10 trips).

Selected for Bayesian optimisation: **XGBoost** (best CV top-1 = primary metric, best CV log loss → best HMM emissions,
4× faster inference than LightGBM). Boosting models show a ~5-point train/CV gap → mild overfitting to tune away.
Next: Stage A3 – Optuna Bayesian optimisation of XGBoost.

## Stage A3 – Bayesian hyper-parameter optimisation (XGBoost) ✅
Run: `.venv/bin/python scripts/05_bayes_opt.py 40` (~33 min wall clock) then `scripts/06_plot_bayes.py`.
Optuna TPE (seed 42, 10 random start-up trials) + MedianPruner, objective = mean GroupKFold(5) Top-1 on the 80 train
trips (same folds as A2). 40 trials: 25 completed, 15 pruned. Search record: `results/bayes_opt/study.db`, `trials.csv`.
Best (trial 31): n_estimators 410, learning_rate 0.0238, max_depth 6, min_child_weight 9.88, subsample 0.758,
colsample_bytree 0.409, gamma 4.85, reg_lambda 0.177, reg_alpha 0.0035. Most important (fANOVA): subsample 0.62.

| | CV top-1 | CV top-1 std | CV macro-F1 | CV log loss | train top-1 | ST top-1 | ST macro-F1 | ST log loss | ms / 1k obs |
|---|---|---|---|---|---|---|---|---|---|
| untuned | 0.8958 | 0.0177 | 0.9366 | 0.0759 | 0.9480 | 0.9015 | 0.9408 | 0.0642 | 17.8 |
| tuned | **0.8974** | **0.0157** | **0.9369** | 0.0759 | 0.9074 | 0.8947 | 0.9371 | 0.0672 | **12.1** |

Tuning gave a small CV gain (+0.16 pt top-1, inside the fold std), a lower fold variance, a much smaller
train/CV gap (5.2 → 1.0 pt: the overfitting is gone) and 32 % faster inference, but −0.7 pt on the 10 student-test
trips. Both are carried into the HMM stage; the final deployed model is chosen there on CV + student-test route results.

## Stage A4 ✅ – HMM/Viterbi route reconstruction + final model (pipeline is FINAL)
Run: `scripts/07_hmm_experiments.py` (OOF emissions + decoder grid, ~5 min) → `scripts/08_export_final.py` (exports
`artifacts/final/` + smoke test) → `scripts/09_plot_hmm.py`. Code: `src/pathsnap/hmm.py` (states / emission /
transition / gaps / restarts / route expansion documented in the docstring), `src/pathsnap/pipeline.py`.

* Decoder tuned on TRAIN out-of-fold emissions only. Only λ = w_e·β matters (proved in the docstring of 07); best λ = 0.75
  (β = 3 m, w_e = 0.25), gap_s = 120 s. Emission model chosen inside the HMM: **untuned XGBoost** (train-OOF HMM Top-1
  0.9523 vs 0.9490 for the Optuna-tuned one) → deployed model = "XGBoost untuned + HMM/Viterbi" v1.0.
* Student-test (10 trips): Top-1 0.9015 → **0.9466**, obs macro-F1 0.9476 → 0.9662, route F1 0.8994 → **0.9715**,
  route length recall 0.981 → 0.999, continuity 0.836 → **0.9995**. Train OOF: Top-1 0.8954 → 0.9523.
* HMM did not help: student-test trip 65 (−0.3 pt, the only trip with a gap > gap_s); 7 of 80 train trips (worst 22, 89: −1.9 pt).
* API end-to-end on student_test: `api/test_api.py` → ALL CHECKS PASSED (50 ms – 1.1 s per trip); `use_hmm=false` works.
* New response keys (additive): `pred_arc_classifier_only`, `route_arcs_classifier_only`, `route_coords_classifier_only`,
  `hmm` {used_hmm, n_gap_breaks, n_disconnect_breaks, params}, `timing_ms` {features, classifier, hmm, total};
  `metrics` adds `top1_accuracy_classifier_only`, `route_continuity_classifier_only`.
* **For Aditya**: §5 results table + Q6 ready to paste at the `<!-- A: results table + Q6 -->` marker:
  `docs/REPORT_SEC5_RESULTS_FROM_A.md` (Figures 27–29, Table 3). My §4 draft with real numbers: `docs/REPORT_SEC4_DRAFT_FROM_A.md`.
  Route maps / screenshots / PDF can now be generated (C4).
