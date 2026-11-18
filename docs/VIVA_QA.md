# Viva preparation – likely questions with short answers

Every team member should be able to give these answers in their own words, and know where the evidence lives.
Numbers come from files in this repo (named in brackets). Items marked **[A: confirm]** depend on Stages A3–A5
and must be checked against Samvith's final implementation before the viva.

## Data and preprocessing
1. **What is in a trip folder?** Four headerless files: `.track` (lon, lat, t, one row per GPS fix), `.nodes`
   (lon, lat; the row number is the node id), `.arcs` (from-node, to-node; the row number is the arc id) and `.route`
   (reference arc ids in driving order). Each trip has its own network, so ids are local to the trip. [`pathsnap/io.py`]
2. **How big is the data?** Train: 80 trips, 208,417 GPS fixes. Student-test: 10 trips, 19,160 fixes. Networks
   have 649 to 307,392 arcs. [`results/eda/eda_summary.md`]
3. **Were there missing values?** None: 0 NaN, 0 out-of-range or (0, 0) coordinates, and timestamps are
   strictly increasing everywhere. The real problems are gaps, jumps and imperfect routes. [fig03, `missing_values.csv`]
4. **What data-quality issues did you find?**
   * 1,216 gaps above 5 s in train.
   * 555 implausible jumps above 60 m/s, 365 of them in trip 32.
   * Reference routes cut short in trips 02, 33 and 54.
   * 14 non-contiguous route steps.
   [`results/stage1/quality_issues.md`]
5. **How noisy is the GPS?** The median distance to the reference route is about 5 m, and the 95th percentile
   about 17 m (excluding the three truncated-route trips). [fig07]
6. **Why the AEQD projection?** The trips span several continents, so one UTM zone would distort distances. An
   azimuthal-equidistant projection centred on each trip gives metric coordinates whose error is far below the
   GPS noise. [`pathsnap/geo.py`]
7. **Why not delete jumps and outliers?** The API must return an arc for every observation. We keep the points
   and flag them as features (`jump`, `dt_prev`, `dt_next`), so the model learns to trust them less.
8. **How do you get one label per GPS point when `.route` is an arc list?** A monotone DP alignment, similar to
   DTW: the route index may only stay or advance, minimising the summed distance. Points more than 30 m from
   their aligned arc get no label (−1) and are excluded from training and scoring. 87.5 % of train and 99.2 % of
   student-test points are labelled. [`pathsnap/labels.py`]
9. **Why monotone?** The vehicle drives the route in order. Monotonicity resolves loops, U-turns and parallel
   roads, which a nearest-arc rule gets wrong.
10. **Which partition did you use for what?** Train for all fitting and cross-validation. Student-test only for
    final evaluation, never for model choice or tuning. Evaluator data only at the demo.

## Candidates and features
11. **What is a candidate?** A road arc within 50 m of a GPS point, at most 12 per point. The classifier scores
    every (observation, candidate) pair. [`RoadNet.candidates`]
12. **Why 50 m?** On train, 30 m already contains the true arc for 100 % of labelled points
    (`results/dataset/radius_study.csv`); 50 m adds a margin. The true arc is missing for only 0.19 % (train) and
    0.04 % (student-test) of points.
13. **Name the feature groups and why each exists.**
    * Geometric (distance, rank, Gaussian likelihood with σ = 5 m): handles noise.
    * Direction (heading cosine): about 95 % of roads are two-way, and only the heading separates the two arcs.
    * Temporal windows (distance of points i ± 2 and i ± 5 to the same arc): shows whether the track follows the arc.
    * Kinematics (speed, Δt, jump): tells the model when the other signals are unreliable.
    * Topology (degrees, continuity with the neighbours' nearest arc).
    [`pathsnap/features.py`]
14. **Why no absolute coordinates as features?** Every trip is a different city. Trip-independent features
    transfer to unseen trips, while coordinates would just memorise places.
15. **What is `head_reliab`?** The displacement over 4 observations divided by 20 m, clipped to [0, 1]. It is
    about 0 when the vehicle is stopped, where the heading is meaningless; it scales the heading features.
16. **Baseline?** Nearest arc: 56.2 % (train) and 54.0 % (student-test) Top-1. It is low because both directions of a
    two-way road are equally near.

## Models and evaluation
17. **How was cross-validation done, and why GroupKFold?** Five folds grouped by trip, so all points of a trip
    sit in the same fold. Neighbouring GPS points are almost identical, so a random split would leak and
    inflate the scores.
18. **Which models did you train?** Eight: logistic regression, Gaussian NB, KNN, decision tree, random forest,
    LightGBM, XGBoost and an MLP. [`pathsnap/models.py`, `results/models/comparison.csv`]
19. **Which metric picks the model, and why?** CV Top-1, because the route is built from per-observation Top-1.
    Log loss is the tie-break, because the probabilities become HMM emissions.
20. **Why XGBoost over LightGBM, when LightGBM has higher student-test Top-1?** XGBoost has the best CV Top-1
    (0.8958 vs 0.8953) and CV log loss (0.0759 vs 0.0762), and it is 4× faster at inference (12 vs 46 ms per 1k
    obs). The 0.2-point student-test difference is far below the fold spread (std ≈ 0.018), and choosing on the
    test set would be leakage.
21. **What does "macro-F1" mean in your tables?** In the model tables it is the candidate-level binary macro-F1
    at threshold 0.5. The API's `score_trip` reports observation-level macro-F1 over arc ids (each route arc is a
    class). Both are in Table 1 of the report.
22. **Why did Gaussian NB fail?** It assumes independent features, but ours are strongly correlated
    (dist/win2_mean |r| = 0.93). It double-counts evidence and becomes over-confident: log loss 1.25, precision
    0.49, and fold 4 Top-1 of only 0.66.
23. **Which models overfit, and what is the evidence?** Boosting: the train-vs-CV Top-1 gap is 0.052 for XGBoost
    and 0.048 for LightGBM. XGBoost's validation loss bottoms at round 120 and then rises while training loss
    keeps falling. Logistic regression shows no gap at all (0.001), i.e. it underfits. [fig17, fig20]
24. **Why are training curves shown only for three models?** Only the MLP (epochs) and LightGBM/XGBoost
    (boosting rounds) have a real iterative history. The others have none, and we did not fake curves.
25. **Why was KNN subsampled?** Its prediction cost grows with training size. Even on 150k rows it is the
    slowest model (798 ms per 1k obs).
26. **What are the errors of the selected model?** On student-test:
    * 50.5 % are "right street, wrong direction" flips, usually one observation long.
    * 33.6 % pick an arc next to the true one.
    * Slow driving (1–8 m/s) holds 72.8 % of the errors.
    * Trip 77 alone holds 49 % of the errors.
    [`results/analysis/error_analysis.md`]

## Bayesian optimisation (Stage A3 – final numbers from `results/bayes_opt/`)
27. **What is Bayesian optimisation and which sampler did you use?** It fits a probabilistic model of
    "hyper-parameters → score" from past trials and uses it to pick promising new trials, instead of a grid or random
    search. We used Optuna's TPE (Tree-structured Parzen Estimator): it splits the trials into good and bad ones,
    models each group's density, and samples where good/bad is highest. Seed 42, with the first 10 trials random.
28. **What did you search?** Nine XGBoost hyper-parameters:
    * `n_estimators` 100–800 and `learning_rate` 0.01–0.3, both log-uniform;
    * `max_depth` 3–12;
    * `min_child_weight` 1–200, log-uniform;
    * `subsample` 0.5–1 and `colsample_bytree` 0.4–1;
    * `gamma` 0–5;
    * `reg_lambda` 1e-3–50 and `reg_alpha` 1e-3–10, both log-uniform.
    [`best_params.json`]
29. **Objective?** Maximise the mean GroupKFold(5) Top-1 on the 80 train trips, using the same folds as the model
    comparison, so the numbers are directly comparable. Student-test was not used during the search.
30. **How many trials, and what is pruning?** 40 trials: 25 completed and 15 pruned. The MedianPruner stops a trial
    after a fold if its running mean is below the median of earlier trials at that fold, which saves time.
31. **Best values?** Trial 31: CV Top-1 0.8974. Its settings: 410 trees, lr 0.024, depth 6, min_child_weight 9.9,
    subsample 0.76, colsample 0.41, gamma 4.85, reg_lambda 0.18, reg_alpha 0.0035. These are much more
    regularised than the untuned model (depth 8, colsample 0.8, gamma 0).
32. **Which hyper-parameter mattered most?** `subsample`, with 62 % of the variance by fANOVA. Values below about
    0.65 cost 0.2–0.4 points of CV Top-1.
33. **Did tuning help? Be honest.**
    * CV Top-1 rose by only +0.16 points (0.8958 → 0.8974), inside the fold std, and the tuned model wins on 3 of 5 folds.
    * The real gain is that overfitting is gone: the train/CV gap went from 5.2 to 1.0 points and the fold std
      from 0.0177 to 0.0157.
    * Inference is about 39 % faster, by a repeated benchmark.
    * But student-test Top-1 dropped 0.7 points (0.9015 → 0.8947) on just 10 trips.
    * So we carried both models into the HMM stage and chose the final one there. [A: final choice]
34. **Why not tune on student-test?** That would turn the test set into a validation set, and its score would no
    longer be an unbiased estimate of performance on new trips.

## HMM / Viterbi [A: confirm every answer against the final code]
35. **Hidden states?** The candidate arcs of each observation.
36. **Emissions?** Derived from the classifier's probability for each candidate. [A: confirm the exact form]
37. **Transitions?** They favour arcs that are connected in the network and consistent with the distance
    driven; disconnected pairs are penalised or forbidden. [A: confirm]
38. **What does Viterbi do?** Dynamic programming that finds the single most probable sequence of states
    (arcs), in O(T · K²) for T observations and K candidates.
39. **Large gaps or missing candidates?** [A: confirm the gap rule, e.g. relaxed transitions above 30 s.] Every
    observation has at least two candidates by construction (`min_k`).
40. **Did the HMM help?** [A: confirm, §5 table: classifier-only vs HMM Top-1, route F1, continuity.] Expected
    from §3.4: it should remove the one-observation direction flips.

## System, API and UI
41. **How do the two laptops talk?** Over HTTP on the same hotspot. The UI on Laptop 2 calls `GET /trips` and
    `POST /predict` on `http://<Laptop-1-IP>:8000`; CORS is open. [`api/main.py`, `ui/app.js`]
42. **How do you guarantee no ground-truth leakage to the UI?**
    * `predict_trip` loads the trip with `with_route=False`.
    * Only `score_trip` reads `.route`, server-side, and it returns numbers only.
    * The API strips any reference-like key.
    * `api/test_api.py` asserts that no such key is returned.
    * The UI has no code that draws a reference route.
43. **What is logged?** One coloured `[PathSnap]` line per request on the Laptop 1 console (client IP, trip,
    n_obs, Top-1, route F1, latency), plus full metrics in `api/logs/api.log`.
44. **What if the evaluator sends a bad trip id?** 400 if it does not match `^[0-9A-Za-z_-]+$`, which blocks
    path traversal; 404 if the folder does not exist. The server never crashes.
45. **Why does the UI work without internet?** Leaflet is bundled in `ui/vendor/`. Only the OpenStreetMap
    background tiles need internet.
46. **Are the metrics computed on the UI side?** No, everything is computed on Laptop 1 against the reference
    route. The UI only displays the numbers it receives.
47. **How would you reproduce any number in the report?** Every figure and table is produced by a script in
    `scripts/`, `eda/`, `analysis/` or `viz/`. The README lists the commands per stage.
