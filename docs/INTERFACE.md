# PathSnap – interface contract between the two work-streams

Two people/agents work in parallel on **disjoint folders**. This file is the only coupling between them.
Change it only by agreement (and commit the change on its own).

| Stream | Owner | Folders it may edit |
|---|---|---|
| **A – ML core** (Laptop 1) | Account A | `src/pathsnap/`, `scripts/`, `artifacts/`, `results/stage1/`, `results/models/`, `results/hmm/`, `config.py` |
| **B – EDA, maps, API, UI, docs** (Laptop 2) | Account B | `eda/`, `viz/`, `api/`, `ui/`, `deploy/`, `report/`, `results/eda/`, `results/maps/`, `README.md` |

Shared, append-only: `requirements.txt` (add a line, never remove), `docs/`.

Git: A works on branch `ml-core`, B on branch `app-eda`; both open PRs into `main`. Because folders are disjoint, merges never conflict.

---

## 1. Python functions (owned by A, called by B)

```python
from pathsnap.pipeline import predict_trip     # src/pathsnap/pipeline.py
from pathsnap.evaluate import score_trip       # src/pathsnap/evaluate.py
from pathsnap.io import load_trip, list_trips  # raw readers
from pathsnap.geo import make_projector, RoadNet
```

### `predict_trip(trip_dir: Path, use_hmm: bool = True) -> dict`
Never reads `.route`. Returns (JSON-serialisable):

| key | type | meaning |
|---|---|---|
| `trip_id` | str | folder name |
| `n_obs` | int | number of GPS observations |
| `gps` | `[[lat, lon], ...]` | raw observations, in time order |
| `t` | `[float]` | timestamps (s) |
| `pred_arc` | `[int]` | predicted arc_id **per observation** (-1 = none) |
| `pred_prob` | `[float]` | model confidence of that arc |
| `route_arcs` | `[int]` | predicted route = `pred_arc` with consecutive duplicates removed |
| `route_coords` | `[[[lat,lon],[lat,lon]], ...]` | one 2-point polyline per entry of `route_arcs` |
| `timing_ms` | dict | stage timings, always contains `total` |
| `model` | dict | `{"name", "version"}` |

Later versions may ADD keys (e.g. `pred_arc_classifier_only`, `route_coords_classifier_only`) – consumers must ignore unknown keys.

### `score_trip(pred: dict, trip_dir: Path) -> dict`
Reads the reference route **server-side**. Returns numbers only:
`top1_accuracy, macro_f1` (None until Stage 3), `route_precision, route_recall, route_f1, route_recall_undirected, route_continuity, n_obs`.

**The UI must never receive reference arcs or reference geometry – only these numbers.**

Right now the pipeline is a *baseline* (nearest arc). It already returns the final format, so the API/UI can be built and tested against it today; A swaps the internals later without changing keys.

---

## 2. HTTP API (built by B in `api/`, runs on Laptop 1)

Base URL `http://<laptop1-ip>:8000`, host `0.0.0.0`, CORS open (`*`).
Trip folder root = env `PATHSNAP_EVAL_DATA` (during development point it at `student_test/`).

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | – | `{"status":"ok","model":{...}}` |
| GET | `/trips` | – | `{"trips":[{"trip_id":"00000000","n_obs":2503}, ...]}` (ids from folder names, n_obs = lines in `.track`) |
| POST | `/predict` | `{"trip_id":"00000000"}` | `predict_trip(...)` output **plus** `"metrics": score_trip(...)` and `"server_latency_ms"` |

* 404 for an unknown trip_id; trip_id must match `^[0-9A-Za-z_-]+$` (no path traversal).
* Every `/predict` call prints one clear log line on the API console, e.g.
  `[PathSnap] 2026-11-18 14:02:11 | client=192.168.1.7 | trip=00000052 | n_obs=262 | top1=0.913 | route_f1=0.88 | 412 ms`
  (the evaluator must see these logs on Laptop 1 – they go in the report as screenshots).

## 3. UI (built by B in `ui/`, runs on Laptop 2)
* Field for API base URL (default `http://localhost:8000`, remembered in localStorage).
* Lists the 10 trips from `GET /trips`; evaluator picks one → `POST /predict`.
* Leaflet/Folium map: raw GPS points (blue), predicted route (red), layer control + legend, auto-fit bounds.
* Metrics panel: Top-1 accuracy, macro F1, route precision/recall/F1, continuity, n_obs, latency, model name.
* Must work for all 10 trips; no reference route on the map.
