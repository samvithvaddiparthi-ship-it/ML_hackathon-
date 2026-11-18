# Brief for Account B (paste this whole file into the second Claude Code session)

You are joining the **PathSnap** project (PES University UE24CS352A Machine Learning hackathon: map-matching noisy
GPS to a road network with ML classifiers + HMM/Viterbi, served via FastAPI to a UI on a second laptop).
Team (Section H): Samvith Vaddiparthi PES1UG24AM468, Vikas MS PES1UG24AM469, Aditya A Kirankumar PES1UG24AM470,
Haneesh E PES1UG24AM471. AI assistance is used and will be disclosed in the README/report.

Another Claude session (**Account A**, Samvith, Laptop 1) owns the ML core. **You (Account B, Aditya, Laptop 2) own EDA, maps, API, UI, deployment and the
report skeleton.** Read `docs/INTERFACE.md` first – it is the contract. Only edit the folders it assigns to you,
work on branch `app-eda`, and open a PR into `main` at the end of each stage. Work in stages and stop for the
user's OK after each one.

## Setup
1. `git clone https://github.com/samvithvaddiparthi-ship-it/ML_hackathon-.git && cd ML_hackathon- && git checkout -b app-eda`
2. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (macOS: `brew install libomp` for LightGBM/XGBoost)
3. Dataset: folder with `train/` (80 trips) and `student_test/` (10 trips). Set `export PATHSNAP_DATA="/path/to/Student folder"`
   (default is `~/Downloads/Student folder`). Each trip folder `<id>/` has `<id>.track` (lon lat t),
   `.nodes` (lon lat, row = node_id), `.arcs` (u v, row = arc_id), `.route` (ground-truth arc_ids in order).
4. Read `results/stage1/quality_issues.md` and `results/stage1/trip_inventory.csv` (Stage 1 already done by A).
5. Rules that matter for grading: never use evaluator-test data for development; the UI must never receive
   reference-route arcs/geometry; no hard-coded predictions; every figure must come from real code runs.

## Stage B1 – EDA (Report §2.2)  → `eda/02_eda.py`, outputs in `results/eda/` and `results/maps/`
State the partition used in every figure title/caption. Produce (PNG, 150 dpi) + a `results/eda/eda_summary.md`:
* dataset inventory table (trips / obs per partition), obs-per-trip histogram, file sizes;
* missing-value analysis (NaN/invalid counts per column per partition – they are zero; show it), Δt distribution
  (log-y histogram), gaps > 5 s / > 30 s per trip, % of 1 s steps per trip;
* speed distribution and implausible-jump counts; GPS→reference-route distance histogram (noise level);
* GPS coordinate distribution / world scatter of trip centres; trajectory plots (lon/lat in projected metres) for ~4 trips;
* road-network visualisation (matplotlib, arcs as thin grey lines + GPS overlay) for 2–3 trips; network size and
  connectivity charts (#nodes, #arcs, #weak components, largest-component fraction);
* at least 2 Folium maps: one train trip and one student-test trip with GPS points, nearby network, reference route,
  layer control and legend → `results/maps/*.html` + a PNG screenshot for the report.
Reuse `pathsnap.io.load_trip`, `pathsnap.geo.make_projector`, `RoadNet` – do not reimplement loaders.

## Stage B2 – reusable Folium module → `viz/folium_maps.py`
`make_trip_map(gps_latlon, network_segments=None, reference_coords=None, predicted_coords=None,
classifier_only_coords=None, title="", out_html=None) -> folium.Map` – distinct colours (GPS blue dots, network grey,
reference green, classifier-only orange dashed, HMM/predicted red), `LayerControl`, an HTML legend, fit-bounds.
Plus `network_segments_near(trip, gps, radius_m=150)` that returns lat/lon segments of arcs near the track (keep maps light).
A will call this for the HMM comparison maps (§5/§6), so keep the signature.

## Stage B3 – FastAPI backend → `api/main.py` (+ `api/README.md`)
Exactly as in INTERFACE.md §2. Load trips from `PATHSNAP_EVAL_DATA`; call `predict_trip` then `score_trip`;
the colourful one-line console log per request; Python `logging` to `api/logs/api.log` as well; Swagger at `/docs`.
Run: `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Test against `student_test/` with the current baseline pipeline.
Add a small `api/test_api.py` (requests to all 10 trips, asserts keys & that no `route`/reference key is returned).

## Stage B4 – Evaluator UI → `ui/` (runs on Laptop 2)
Static `index.html` + `app.js` + `style.css` with Leaflet from a CDN (no build step; served by
`python -m http.server 5500` from `ui/`). INTERFACE.md §3. Clean, presentable: header with project/team, trip list
(cards or dropdown), "Run" button, spinner, map, metrics cards, error banner if API unreachable, API-URL setting.

## Stage B5 – Deployment + docs
* `deploy/`: `run_api.sh` / `run_api.bat`, `run_ui.sh` / `run_ui.bat`, instructions for the two-laptop demo
  (same Wi-Fi/hotspot, find Laptop 1 IP, open firewall port 8000, set API URL in UI, where eval trips go).
* `README.md`: overview, folder map, install, how to reproduce every stage (A will add the ML commands), deployment,
  AI-use disclosure, team.
* `report/`: report skeleton (Markdown → PDF via a script, or HTML→PDF) with the exact section structure of the
  "Deliverables" PDF: §2.2 EDA (fill it from B1), §3 model comparison (placeholders – A fills), §4 Bayesian opt, §5 HMM,
  §6 Folium, §7 API/UI/deployment (screenshots), team & AI-use disclosure. Title page with team names/SRNs, Section H.

## Hand-back
When a stage is done: commit, push `app-eda`, open/refresh the PR, and write a 5-line summary in `docs/STATUS_B.md`
(what exists, how to run it, open issues). Account A reads that file.
