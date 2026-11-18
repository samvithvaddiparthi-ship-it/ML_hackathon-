# PathSnap API (Laptop 1)

FastAPI backend implementing `docs/INTERFACE.md` §2. It loads a trip folder, runs
`pathsnap.pipeline.predict_trip` (classifier + HMM/Viterbi once Stage A is final) and scores it server-side
with `pathsnap.evaluate.score_trip`. The reference route never leaves the server – only metric numbers do.

## Run
```bat
:: Windows cmd, from the repo root
set "PATHSNAP_EVAL_DATA=C:\path\to\trips"      :: folder holding the 10 trip sub-folders
.venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000
```
```bash
# macOS / Linux
PATHSNAP_EVAL_DATA="/path/to/trips" .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Or just `deploy\run_api.bat` / `deploy/run_api.sh`. On start-up the console prints the trip folder, the number
of trips found and the `http://<LAN-IP>:8000` address to type into the UI. Swagger UI: `http://localhost:8000/docs`.

## Endpoints
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{"status":"ok","model":{"name","version"}}` |
| GET | `/trips` | – | `{"trips":[{"trip_id","n_obs"}]}` – sub-folders of `PATHSNAP_EVAL_DATA` |
| POST | `/predict` | `{"trip_id":"00000052","use_hmm":true}` | `predict_trip` output + `metrics` + `server_latency_ms`; passes through the optional HMM keys `pred_arc_classifier_only`, `route_coords_classifier_only`, `hmm`, extra `timing_ms` entries |

Errors: 400 for an id that is not `^[0-9A-Za-z_-]+$` (blocks path traversal), 404 for an unknown trip,
500 with the exception text if the pipeline fails (the server keeps running).

## Logs
* Console: one coloured line per request, e.g.
  `[PathSnap] 2026-11-18 14:02:11 | client=192.168.1.7 | trip=00000052 | n_obs=262 | top1=n/a | route_f1=0.71 | 28 ms`
  (route_f1 green ≥ 0.8, yellow ≥ 0.5, red below). When the pipeline reports HMM chain breaks the line adds
  `hmm_breaks=<gap+disconnect>`; requests with `use_hmm=false` show `hmm=off`. Screenshot this window for Report §7.
* File: `api/logs/api.log` (rotating, plain text, full metrics per request). Not committed.

## Test
With the API running: `python api/test_api.py http://localhost:8000` – calls `/predict` for every listed trip with `use_hmm` true and false,
checks all contract keys, that no reference-route key is returned, n_obs consistency, and that bad ids are rejected.
