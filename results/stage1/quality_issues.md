# Stage 1 – Dataset inventory & data-quality record

Partitions analysed: **train (80 trips)** and **student_test (10 trips)**. The evaluator-test partition was not available and has not been touched.
Produced by `scripts/01_inventory.py` → `trip_inventory.csv` (one row per trip, 60+ checks) and `partition_summary.csv`.

## Structure (verified on the actual files)
| File | Content | Verified |
|---|---|---|
| `.track` | lon, lat, t (s) – 3 cols, no header | 227,577 rows, all 3 numeric columns, 0 NaN, 0 out-of-range coords |
| `.nodes` | lon, lat – row index = node_id | 2,024,174 rows, all numeric |
| `.arcs` | from_node, to_node – row index = arc_id | 4,061,945 rows, 0 ids out of range |
| `.route` | chronological arc_id sequence | 34,813 rows, 0 ids out of range, 0 consecutive repeats |

* Every one of the 90 trip folders holds exactly the 4 files. Each trip has its **own** road network (node/arc ids are local to a trip).
* Column order really is (lon, lat): e.g. trip 34 centres at lon 44.3 / lat 40.4 (Armenia). Trips are spread worldwide (Europe-heavy, plus USA, Mexico, Vietnam, Japan, Argentina…).
* Train: 208,417 GPS obs (361 – 19,313 per trip, median 1,529). Student test: 19,160 obs (262 – 3,958, median 1,420).
* Networks: 299 – 155,003 nodes, 649 – 307,392 arcs per trip. ~95 % of arcs have a reverse arc (two-way roads stored as 2 directed arcs).

## Data-quality issues found
1. **Timestamps** – always strictly increasing (0 duplicates / reversals). Median Δt = 1 s, but only 93.8 % (train) / 98.2 % (student test) of steps are exactly 1 s.
   1,216 gaps > 5 s and 133 gaps > 30 s in train. Extreme gaps: trip 90 (153,064 s ≈ 42 h), trip 78 (52,810 s), trip 69 (2,658 s), trip 21 (2,161 s).
   Some trips are heavily down-sampled (trip 22: 53 % of steps are 1 s; trips 4, 39, 46 < 65 %).
2. **Non-integer timestamps** in trips 02 (7,072 rows), 71 (848), 97 (1,739) – sub-second jitter; harmless once Δt is used as a float.
3. **GPS jumps** – steps implying > 60 m/s (216 km/h): trip 32 (365), trip 13 (103), trip 01 (52) and a few others → outliers / multipath.
4. **Reference routes are not perfect ground truth**
   * 14 "breaks" (consecutive route arcs not head-to-tail connected) in 10 trips (e.g. trip 19 has 3).
   * **Truncated routes**: in trips 33 and 54 the route covers only the first part of the drive – after that the GPS keeps going but the route has already ended (median GPS→route distance 122 m and 139 m). Trip 02 also leaves the route at its end (548 m).
   * Hence per-observation labels must be derived carefully and points that cannot be aligned to the route must be **excluded from training/scoring**, not forced onto a wrong arc.
5. **GPS noise level** – median distance GPS→reference route ≈ 4.5 m, 95th percentile ≈ 14 m (train) / 16 m (student test). This sets the candidate search radius and the emission σ.
6. **Network connectivity** – no isolated nodes, but networks have 1 – 421 weakly connected components; the largest weak component holds 69 – 100 % of nodes. Some duplicate arcs (≤ 322) and self-loops (≤ 8), 0-length arcs (≤ 14). The HMM must treat unreachable candidate pairs as invalid transitions.
7. GPS track length is often longer than route length (ratio median 0.85) because 1 Hz noise accumulates as zig-zag distance when the vehicle is slow/stopped.

## Decisions taken (to be justified in the report)
* Distances are computed in a **per-trip azimuthal-equidistant projection (metres)** centred on the trip – trips are worldwide, so one fixed UTM zone would be wrong; AEQD is accurate to well under 0.1 % over a 50 km trip.
* Δt is used as a float; gaps > 30 s are flagged and treated specially by the HMM (looser transition model).
* Points with implausible speed are kept but get a "jump" feature; they are not deleted, because the API must still return a segment for every observation.
* Per-observation labels will be built by aligning each GPS point to the route arc sequence monotonically (Stage 2), with a distance cut-off to drop unalignable points.
