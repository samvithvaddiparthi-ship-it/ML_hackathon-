# B7 – Error analysis of the selected classifier (XGBoost)

Partition: **student-test** (10 trips), 19,012 labelled observations. Scores: artifacts/models/XGBoost.joblib applied to regenerated candidates.
Top-1 = 90.03 % (nearest-arc baseline 53.59 %); 1,896 wrong observations. Produced by `analysis/error_analysis.py`.

Columns: share of observations in the bucket, baseline and model Top-1, share of all model errors in the bucket.

## (a) candidates per observation
| candidates   |   n_obs | share_obs   | baseline_top1   | top1   | share_errors   |
|:-------------|--------:|:------------|:----------------|:-------|:---------------|
| 1-3          |    4454 | 23.4 %      | 55.7 %          | 98.8 % | 2.7 %          |
| 4-6          |    3868 | 20.3 %      | 60.4 %          | 95.1 % | 10.0 %         |
| 7-9          |    1931 | 10.2 %      | 58.9 %          | 91.2 % | 9.0 %          |
| 10-12        |    8759 | 46.1 %      | 48.3 %          | 83.0 % | 78.3 %         |

## (b) speed
| speed          |   n_obs | share_obs   | baseline_top1   | top1   | share_errors   |
|:---------------|--------:|:------------|:----------------|:-------|:---------------|
| stopped <1 m/s |     247 | 1.3 %       | 38.1 %          | 65.6 % | 4.5 %          |
| slow 1-8 m/s   |    5723 | 30.1 %      | 42.4 %          | 75.9 % | 72.8 %         |
| fast >8 m/s    |   13042 | 68.6 %      | 58.8 %          | 96.7 % | 22.7 %         |

## (c) heading reliability
| heading reliability   |   n_obs | share_obs   | baseline_top1   | top1   | share_errors   |
|:----------------------|--------:|:------------|:----------------|:-------|:---------------|
| low <0.25             |     732 | 3.9 %       | 38.0 %          | 60.1 % | 15.4 %         |
| mid 0.25-0.75         |    2361 | 12.4 %      | 39.0 %          | 62.6 % | 46.6 %         |
| high >=0.75           |   15919 | 83.7 %      | 56.5 %          | 95.5 % | 38.0 %         |

## (d1) position on true arc
| position                     |   n_obs | share_obs   | baseline_top1   | top1   | share_errors   |
|:-----------------------------|--------:|:------------|:----------------|:-------|:---------------|
| near a node (<10 % from end) |    3938 | 20.7 %      | 47.2 %          | 84.6 % | 32.0 %         |
| mid-arc                      |   15066 | 79.3 %      | 55.3 %          | 91.5 % | 67.6 %         |

## (d2) junction at end of true arc
| out-degree at end of true arc   |   n_obs | share_obs   | baseline_top1   | top1   | share_errors   |
|:--------------------------------|--------:|:------------|:----------------|:-------|:---------------|
| dead end / 1                    |    3504 | 18.4 %      | 83.6 %          | 92.8 % | 13.2 %         |
| 2 (plain road)                  |    9904 | 52.1 %      | 49.3 %          | 91.6 % | 43.8 %         |
| >=3 (junction)                  |    5596 | 29.4 %      | 42.5 %          | 85.6 % | 42.6 %         |

## (e) GPS distance to true arc
| GPS distance to true arc   |   n_obs | share_obs   | baseline_top1   | top1   | share_errors   |
|:---------------------------|--------:|:------------|:----------------|:-------|:---------------|
| 0-5 m                      |    9172 | 48.3 %      | 58.7 %          | 92.6 % | 35.7 %         |
| 5-10 m                     |    6049 | 31.8 %      | 53.3 %          | 90.2 % | 31.3 %         |
| 10-20 m                    |    3382 | 17.8 %      | 44.2 %          | 85.3 % | 26.2 %         |
| 20-50 m                    |     401 | 2.1 %       | 20.4 %          | 69.8 % | 6.4 %          |

## (f) per trip
|   trip_id |   n_obs |   top1 |   baseline_top1 |   reverse_errors |
|----------:|--------:|-------:|----------------:|-----------------:|
|  00000000 |    2501 |   95.5 |            54.3 |               16 |
|  00000007 |    1327 |   98   |            45.9 |               23 |
|  00000009 |    1031 |   89.8 |            42.3 |               27 |
|  00000040 |    2946 |   95.7 |            53.2 |               50 |
|  00000052 |     262 |   97.3 |            77.9 |                0 |
|  00000056 |    3920 |   92.4 |            68.8 |               33 |
|  00000065 |     768 |   86.7 |            68.9 |               54 |
|  00000076 |    1485 |   93.3 |            40.7 |                0 |
|  00000077 |    3898 |   76   |            42.7 |              739 |
|  00000091 |     874 |   90.7 |            59.8 |               16 |

## Error taxonomy
| error_kind                      |   n |   share_of_errors |
|:--------------------------------|----:|------------------:|
| reverse direction (same street) | 958 |              50.5 |
| adjacent arc (shares a node)    | 637 |              33.6 |
| other arc                       | 293 |              15.5 |
| true arc not a candidate        |   8 |               0.4 |

## Findings
* **Speed / heading is the main driver.** Above 8 m/s Top-1 is 96.7 %; at 1–8 m/s it drops to 75.9 %, and these slow observations (30.1 % of all) hold 72.8 % of the errors. Fully stopped points are worst (65.6 %) but rare (1.3 %). By heading reliability: 60.1 % (low) / 62.6 % (mid) / 95.5 % (high) – when the vehicle moves too little, the heading features cannot tell the two directions of a street apart.
* **50.5 % of all errors (958) are 'right street, wrong direction'** – the predicted arc is the reverse arc of the true one. They are short flips, not long wrong stretches: 465 runs, median length 1 observation(s), longest 23.
* 33.6 % of errors choose an arc that shares a node with the true arc (previous/next segment or a side road). Consistently, Top-1 is 84.6 % within 10 % of an arc end vs 91.5 % mid-arc, and 85.6 % when the true arc ends in a junction (out-degree ≥ 3).
* Only 15.5 % of errors are unrelated arcs, and 8 errors are candidate-generation misses (the true arc was not within 50 m / top-12).
* **One trip dominates**: trip 00000077 holds 49.4 % of all errors (Top-1 76.0 %; 739 reverse-direction errors). It is a slow trip (median speed 5.5 m/s). The other nine trips range 86.7 %–98.0 %.
* Dense networks are harder: 98.8 % with 1–3 candidates vs 83.0 % with 10–12 (the latter are 46.1 % of observations and 78.3 % of errors).
* GPS distance matters less: 92.6 % at 0–5 m, 85.3 % at 10–20 m; only the 20–50 m bucket (2.1 % of observations) falls to 69.8 %.
* The classifier gains most where the nearest-arc baseline fails: overall 53.6 % → 90.0 %; at junctions 42.5 % → 85.6 %.
* **Implication for the HMM (§5):** most remaining errors are short direction flips or jumps to a neighbouring arc while the street itself is right. A transition model that only allows connected, consistently directed arc sequences should remove many of them, especially on slow trips like 00000077; it cannot help with the few candidate-generation misses.

Figures: `fig_b7_breakdowns.png`, `fig_b7_trips_and_error_types.png`. CSVs: `by_*.csv`, `error_kinds.csv`.