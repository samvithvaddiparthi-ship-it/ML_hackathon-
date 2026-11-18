# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus), Section H.

| Name | SRN |
|---|---|
| Samvith Vaddiparthi | PES1UG24AM468 |
| Vikas MS | PES1UG24AM469 |
| Aditya A Kirankumar | PES1UG24AM470 |
| Haneesh E | PES1UG24AM471 |

PathSnap matches noisy GPS observations to the arcs of a road network: a per-observation ML classifier scores
candidate arcs, HMM/Viterbi decoding turns them into a connected route, a FastAPI server on Laptop 1 runs the
model, and a browser UI on Laptop 2 lets the evaluator pick a trip and see the predicted route and metrics.
Module contract between the two work streams: [`docs/INTERFACE.md`](docs/INTERFACE.md).

## Repository layout
| Folder | Content |
|---|---|
| `src/pathsnap/` | core library – raw readers (`io.py`), projection & road-network index (`geo.py`), inference (`pipeline.py`), scoring (`evaluate.py`), features/labels/models/HMM |
| `scripts/` | numbered ML pipeline scripts (inventory, labels, training, Bayesian optimisation, HMM evaluation) |
| `config.py` | data paths (env `PATHSNAP_DATA`, `PATHSNAP_EVAL_DATA`) and seed |
| `eda/` | `02_eda.py` – exploratory data analysis for Report §2.2 |
| `viz/` | `folium_maps.py` – reusable Folium map builder; `make_route_maps.py` – classifier-only vs HMM route maps |
| `analysis/` | `error_analysis.py` – where the selected classifier fails (student_test) |
| `api/` | FastAPI backend (`main.py`), smoke test (`test_api.py`), [API README](api/README.md) |
| `ui/` | evaluator UI – static `index.html` / `app.js` / `style.css`, Leaflet bundled in `ui/vendor/` |
| `deploy/` | start scripts for both laptops and the [two-laptop guide](deploy/DEPLOYMENT.md) |
| `report/` | report source (`report.md`) and PDF builder (`build_report.py`) |
| `results/` | generated tables, figures and maps (`stage1/`, `eda/`, `maps/`, `models/`, `hmm/`) |
| `artifacts/` | saved model weights and configuration used by the API |
| `docs/` | interface contract, hand-over notes, status files, viva Q&A (`VIVA_QA.md`) |

## Installation
Python 3.11+ recommended.
```bat
:: Windows (cmd)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\playwright install chromium        :: only for map screenshots / report PDF
```
```bash
# macOS / Linux   (macOS: brew install libomp  for LightGBM/XGBoost)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

## Data
The dataset is not committed. Point `PATHSNAP_DATA` at the folder containing `train/` (80 trips) and
`student_test/` (10 trips); the default is `~/Downloads/Student folder`. Each trip folder `<id>/` holds
`<id>.track` (lon lat t), `.nodes` (lon lat), `.arcs` (u v) and `.route` (reference arc ids).
```bat
set "PATHSNAP_DATA=C:\Users\<you>\Downloads\Student folder"
```
The evaluator-test trips are never used during development; at demo time they are placed on Laptop 1 and
selected with `PATHSNAP_EVAL_DATA`.

## Reproducing every stage
Commands use Windows paths (`.venv\Scripts\python`); on macOS/Linux use `.venv/bin/python`.

| Stage | Command | Outputs |
|---|---|---|
| 1 – dataset inventory & quality | `.venv\Scripts\python scripts\01_inventory.py` | `results/stage1/` |
| EDA (Report §2.2) | `.venv\Scripts\python eda\02_eda.py` | `results/eda/`, `results/maps/eda_*.html/.png` |
| A1 – labels, candidates, features | `.venv\Scripts\python scripts\02_build_dataset.py` | `results/cache/` (git-ignored), `results/dataset/` |
| A2 – train & compare 8 models | `.venv\Scripts\python scripts\03_train_models.py` then `scripts\04_plot_models.py` | `results/models/`, `artifacts/models/` |
| A3 – Bayesian optimisation (Optuna TPE, 40 trials, ~33 min) | `.venv\Scripts\python scripts\05_bayes_opt.py 40` then `scripts\06_plot_bayes.py` | `results/bayes_opt/`, `artifacts/models/XGBoost_tuned.joblib` |
| A3 – repeated inference benchmark (untuned vs tuned) | `.venv\Scripts\python analysis\benchmark_inference.py` | `results/analysis/inference_benchmark.csv` |
| A4–A5 – HMM tuning, final pipeline | `.venv\Scripts\python scripts\07_hmm.py` *(ML-core stream – command/outputs confirmed when A4 lands)* | `results/hmm/`, `artifacts/` |
| B7 – error analysis | `.venv\Scripts\python analysis\error_analysis.py` | `results/analysis/` |
| B8 – route-comparison maps | `.venv\Scripts\python viz\make_route_maps.py --all-student-test` | `results/maps/<id>_routes.html/.png` |
| API smoke test | start API, then `.venv\Scripts\python api\test_api.py` | console report |
| Report PDF | `.venv\Scripts\python report\build_report.py` | `report/PathSnap_Report.pdf` |

## Final model
Classifier: XGBoost (selected in Stage A2 on CV Top-1; Bayesian-optimised in A3; untuned vs tuned decided after the
HMM stage – see report §4/§5 and `docs/STATUS_A.md`). Decoder: HMM + Viterbi over the directed road network
(`src/pathsnap/hmm.py`). Served by `pathsnap.pipeline.predict_trip`, which the API calls unchanged.

## Deployment (two laptops)
Full guide: [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md).
1. **Laptop 1 (API):** `deploy\run_api.bat "C:\path\to\eval_trips"` – note the `http://<IP>:8000` it prints.
   Run `deploy\open_firewall.bat` once as Administrator.
2. **Laptop 2 (UI):** `deploy\run_ui.bat http://<laptop1-IP>:8000` – opens `http://localhost:5500`.
3. Same Wi-Fi / phone hotspot for both. Pick a trip → *Run map-matching*; Laptop 1's console logs each request.

## AI-assistance disclosure
AI coding assistants were used during development (code scaffolding, debugging and drafting documentation).
All experiments were actually executed by the team; every reported number is produced by the scripts in this
repository, and every team member can explain the code and results.
