# Two-laptop demo – step by step

| | Laptop 1 – model + API | Laptop 2 – evaluator UI |
|---|---|---|
| Runs | `api/` (FastAPI, port 8000), the trained model, road networks, scoring | `ui/` (static page, port 5500) |
| Needs | repo + `.venv` + the evaluation trip folders | repo (only `ui/` is used) + any Python 3 + a browser |

## 0. Before the evaluation (once)
1. Both laptops: `git clone` the repo and `git checkout main` (after the PRs are merged).
2. Laptop 1: create the environment – `python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`
   (macOS: `brew install libomp` first). `deploy\run_api.bat` does this automatically the first time.
3. Laptop 1 (Windows): run `deploy\open_firewall.bat` **as Administrator** once (opens TCP 8000).
   macOS: allow "python" when the firewall prompt appears.

## 1. Network
* Put both laptops on the **same Wi-Fi or the same phone hotspot**. Campus Wi-Fi often blocks laptop-to-laptop
  traffic (client isolation) – a phone hotspot is the safe choice.
* Leaflet is bundled in `ui/vendor/`, so the UI works without internet; only the OpenStreetMap background
  tiles need internet (without it the GPS points, route and metrics still show on a grey background).

## 2. Laptop 1 – start the API
1. Copy the evaluator's trip folders (each `<id>/` with `.track .nodes .arcs .route`) into one folder,
   e.g. `C:\PathSnap\eval_trips\`.
2. `deploy\run_api.bat "C:\PathSnap\eval_trips"`  (macOS/Linux: `deploy/run_api.sh "/path/eval_trips"`).
3. The console prints the trips found and a line like `UI API URL : http://192.168.43.17:8000` – that is the
   address Laptop 2 needs. (Or run `ipconfig` and take the IPv4 address of the Wi-Fi adapter.)
4. Check locally: open `http://localhost:8000/docs` → `GET /trips` → *Try it out*.

## 3. Laptop 2 – start the UI
1. `deploy\run_ui.bat http://192.168.43.17:8000` (use Laptop 1's IP). The browser opens `http://localhost:5500`.
2. If the status dot is red: check the IP, that both laptops are on the same network, and that
   `http://<laptop1-ip>:8000/health` opens in Laptop 2's browser (if not → firewall on Laptop 1).
3. Pick a trip → **Run map-matching**. Laptop 1's console prints one coloured `[PathSnap] ...` line per request.

## 4. Screenshots for Report §7
* Laptop 2: the UI after running a trip (map + metrics).
* Laptop 1: the API console showing the `[PathSnap]` request lines (the "CMD of the receiving laptop").

## Troubleshooting
| Symptom | Fix |
|---|---|
| `cannot reach http://…:8000` in the UI | wrong IP / different networks / firewall; test `/health` in Laptop 2's browser |
| UI loads but trip list is empty | `PATHSNAP_EVAL_DATA` points at the wrong folder – the API banner shows the folder and trip count |
| 404 for a trip | the folder name must equal the file stem (`00000052/00000052.track`) |
| Port 8000 busy | close the other uvicorn window, or change `--port` in `run_api.*` and the UI URL |
| Map is grey | no internet for map tiles – everything else still works |

## Helper scripts (Windows, double-click or run from any folder)
| Script | Does |
|---|---|
| `deploy\setup_env.bat` | create `.venv`, install requirements + Chromium (once) |
| `deploy\run_eda.bat` | re-run the EDA → `results\eda`, `results\maps` |
| `deploy\run_api.bat ["trips folder"]` | start the API (Laptop 1) |
| `deploy\run_ui.bat [api-url]` | serve and open the UI (Laptop 2) |
| `deploy\test_api.bat [api-url]` | smoke-test a running API |
| `deploy\test_map.bat [trip_id]` | build and open one Folium map from student_test |
| `deploy\run_error_analysis.bat` | B7 error analysis → `results\analysis` |
| `deploy\make_route_maps.bat [trip_id]` | classifier-only vs HMM route maps → `results\maps\<id>_routes.html/.png` |
| `deploy\benchmark_inference.bat` | repeated untuned vs tuned XGBoost inference timing |
| `deploy\build_report.bat` | build and open `report\PathSnap_Report.pdf` |
| `deploy\open_firewall.bat` | open TCP 8000 (run as Administrator, Laptop 1) |
| `deploy\push.bat "message"` | commit all changes on `app-eda` and push |
