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
