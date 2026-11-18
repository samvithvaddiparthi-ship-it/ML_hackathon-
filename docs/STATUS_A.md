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
