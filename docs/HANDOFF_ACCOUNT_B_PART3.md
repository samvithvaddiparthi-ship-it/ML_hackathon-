# Brief for Account B – part 3 (paste into Aditya's Claude session)

Account A is now doing **only** the ML-critical path: HMM decoder tuning (`scripts/07_hmm.py`), classifier-only vs
HMM comparison, final model choice, wiring `src/pathsnap/pipeline.py` to XGBoost + HMM, and **report §5**.
Do NOT edit `src/`, `scripts/`, `artifacts/`, `results/hmm/` or report §5. Everything below is yours.
Pull the latest `ml-core` first (Stage A3 = Bayesian optimisation is pushed there).

## C1 – Report §4 Bayesian optimisation (start now, data is final)
Source files (all real, do not retype numbers by hand – read them with a small script):
`results/bayes_opt/best_params.json`, `trials.csv`, `tuned_vs_untuned.csv`, `param_importance.csv`,
plots `results/bayes_opt/plots/01..04_*.png`, code `scripts/05_bayes_opt.py`, summary in `docs/STATUS_A.md`.
Answer the 6 required questions: params searched + ranges (`search_space` in best_params.json), objective
(max mean GroupKFold(5) Top-1 on 80 train trips, same folds as §3), trials (40: 25 complete, 15 pruned, TPE seed 42,
MedianPruner), best values (trial 31), tuned vs untuned table (CV macro-F1, CV Top-1, ST Top-1, ST macro-F1,
log loss, inference ms/1k obs), and an honest verdict: small CV gain inside fold std, overfitting gap 5.2 → 1.0 pt,
32 % faster, but −0.7 pt on student-test; final choice is made in §5 after the HMM. Insert the 4 plots.

## C2 – UI / API polish for the HMM output
The pipeline will soon ADD keys: `pred_arc_classifier_only`, `route_coords_classifier_only`,
`hmm` (dict: `n_gap_breaks`, `n_disconnect_breaks`, params) and `timing_ms` entries `features`, `classifier`, `hmm`.
* UI: draw the classifier-only route (orange dashed, toggleable) and show a small "HMM: breaks / timings" card;
  must tolerate the keys being absent. Add a "use HMM" checkbox that sends `use_hmm`.
* API console line: add `hmm_breaks=` when present. Keep FORBIDDEN_KEYS filtering.
* Extend `api/test_api.py` to run each trip with `use_hmm` true and false.

## C3 – Report §8, README, viva prep
* §8 team contributions: ask the team (Samvith, Vikas, Haneesh, you) what each actually did – do not invent.
* README: final stage table incl. A3/A4 commands, final model, how to reproduce every number.
* `docs/VIVA_QA.md`: add Optuna answers from C1; leave HMM answers marked [A] – A fills them.

## C4 – When A says the pipeline is final (watch `docs/STATUS_A.md` for "Stage A4 ✅")
1. `python viz/make_route_maps.py --all-student-test` → put 2 screenshots in §6 (trip 00000077 + one clean trip).
2. Run `api/test_api.py` against student_test; two-laptop rehearsal; take the §7 UI + Laptop-1 console screenshots.
3. `python report/build_report.py` → final PDF; check every TODO marker is gone.

## C5 – Draft report §5 METHOD text now (A only adds the result tables later)
Read `src/pathsnap/hmm.py` (its docstring is the spec) and write in `report/report.md` §5 the answers to Q1–Q5:
hidden states (candidate arc + along-arc position per observation, ≤12), emission (w_e·log p_XGBoost), transition
(−|d_route − d_gps|/β over the directed network, speed-based cut-off), large gaps (dt > gap_s → network constraint
dropped), disconnected candidates (chain restarted, counted), route expansion with shortest-path arcs.
Leave a marker `<!-- A: results table + Q6 -->` where the classifier-only vs HMM results go. Do not invent numbers.

Report your progress in `docs/STATUS_B.md` as before.
