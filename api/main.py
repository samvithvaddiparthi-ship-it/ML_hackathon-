"""PathSnap FastAPI backend (runs on Laptop 1). Contract: docs/INTERFACE.md section 2.

    GET  /health   -> {"status": "ok", "model": {...}}
    GET  /trips    -> {"trips": [{"trip_id", "n_obs"}, ...]}
    POST /predict  {"trip_id": "..."} -> predict_trip(...) + "metrics" (score_trip) + "server_latency_ms"

Trip folders are read from env PATHSNAP_EVAL_DATA (during development: the student_test folder).
The reference route is only read server-side by score_trip; the response carries numbers only.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000      (Swagger UI at /docs)
"""
from __future__ import annotations

import logging
import os
import re
import socket
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware           # noqa: E402
from pydantic import BaseModel, Field                        # noqa: E402

from config import EVAL_DIR                                  # noqa: E402
from pathsnap.pipeline import predict_trip, MODEL_INFO       # noqa: E402
from pathsnap.evaluate import score_trip                     # noqa: E402

TRIP_ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")
# keys that could carry reference-route information must never leave the server
FORBIDDEN_KEYS = {"route", "reference", "reference_arcs", "reference_coords", "ref_arcs", "ref_coords", "true_arcs"}

if os.name == "nt":
    os.system("")                     # enables ANSI colours in the Windows console (cmd / PowerShell)
C = {"reset": "\033[0m", "bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
     "red": "\033[91m", "grey": "\033[90m", "magenta": "\033[95m"}

# ------------------------------------------------------------------ logging (file, plain text)
LOG_DIR = ROOT / "api" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    fh = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
log.propagate = False


def data_dir() -> Path:
    return Path(os.environ.get("PATHSNAP_EVAL_DATA", EVAL_DIR))


def trip_path(trip_id: str) -> Path:
    if not TRIP_ID_RE.match(trip_id):
        raise HTTPException(status_code=400, detail="trip_id may only contain letters, digits, '_' and '-'")
    p = data_dir() / trip_id
    if not (p / f"{trip_id}.track").is_file():
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{trip_id}'")
    return p


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))            # no packet is sent; picks the outgoing interface
        ips.add(s.getsockname()[0]); s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def hmm_breaks(pred):
    """Total HMM chain breaks (gap + disconnect) if the pipeline reports them, else None."""
    h = pred.get("hmm") if isinstance(pred, dict) else None
    if not isinstance(h, dict):
        return None
    import numbers
    vals = [h.get(k) for k in ("n_gap_breaks", "n_disconnect_breaks")
            if isinstance(h.get(k), numbers.Real) and not isinstance(h.get(k), bool)]
    return int(sum(vals)) if vals else None


def to_jsonable(x):
    """numpy scalars/arrays -> plain Python, recursively (the HMM params dict may carry numpy values)."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        v = x.item()
        return None if isinstance(v, float) and v != v else v
    if isinstance(x, float) and x != x:          # NaN is not valid JSON
        return None
    return x


def console_line(client, trip_id, n_obs, metrics, ms, status="ok", breaks=None, use_hmm=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if status != "ok":
        print(f"{C['bold']}{C['magenta']}[PathSnap]{C['reset']} {C['grey']}{ts}{C['reset']} | client={client} | "
              f"trip={trip_id} | {C['red']}{status}{C['reset']}", flush=True)
        return
    f1 = metrics.get("route_f1") or 0
    col = C["green"] if f1 >= 0.8 else C["yellow"] if f1 >= 0.5 else C["red"]
    print(f"{C['bold']}{C['magenta']}[PathSnap]{C['reset']} {C['grey']}{ts}{C['reset']} | "
          f"client={C['cyan']}{client}{C['reset']} | trip={C['bold']}{trip_id}{C['reset']} | n_obs={n_obs} | "
          f"top1={fmt(metrics.get('top1_accuracy'))} | {col}route_f1={fmt(metrics.get('route_f1'), 2)}{C['reset']} | "
          + ("" if use_hmm else f"{C['yellow']}hmm=off{C['reset']} | ")
          + ("" if breaks is None else f"hmm_breaks={breaks} | ")
          + f"{ms:.0f} ms", flush=True)


# ------------------------------------------------------------------ app
from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(_app):
    banner()
    yield


app = FastAPI(lifespan=lifespan, title="PathSnap API", version="1.0",
              description="Map-matching of noisy GPS trips to the road network (UE24CS352A hackathon, Section H).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000052"])
    use_hmm: bool = True


def banner():
    d = data_dir()
    n = len([p for p in d.iterdir() if p.is_dir()]) if d.is_dir() else 0
    print(f"\n{C['bold']}{C['magenta']}PathSnap API{C['reset']}  model={MODEL_INFO['name']} v{MODEL_INFO['version']}")
    print(f"  trips folder : {d}  ({n} trips){'' if n else C['red'] + '  <-- not found / empty, set PATHSNAP_EVAL_DATA' + C['reset']}")
    for ip in lan_ips():
        print(f"  UI API URL   : {C['cyan']}http://{ip}:8000{C['reset']}")
    print(f"  Swagger docs : http://localhost:8000/docs\n", flush=True)
    log.info("startup data_dir=%s trips=%d model=%s", d, n, MODEL_INFO)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_INFO}


@app.get("/trips")
def trips():
    d = data_dir()
    if not d.is_dir():
        raise HTTPException(status_code=500, detail=f"trip folder not found: {d}")
    out = []
    for p in sorted(x for x in d.iterdir() if x.is_dir() and TRIP_ID_RE.match(x.name)):
        trk = p / f"{p.name}.track"
        if trk.is_file():
            out.append({"trip_id": p.name, "n_obs": count_lines(trk)})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    try:
        p = trip_path(req.trip_id)
    except HTTPException as e:
        console_line(client, req.trip_id, "-", {}, 0, status=f"HTTP {e.status_code}: {e.detail}")
        log.warning("client=%s trip=%s status=%s", client, req.trip_id, e.status_code)
        raise
    try:
        pred = predict_trip(p, use_hmm=req.use_hmm)
        metrics = score_trip(pred, p)
    except Exception as e:                                   # keep the server alive, report the failure
        log.exception("predict failed trip=%s", req.trip_id)
        console_line(client, req.trip_id, "-", {}, 0, status=f"ERROR {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}")
    for k in FORBIDDEN_KEYS & pred.keys():                   # defence in depth: never ship reference data
        pred.pop(k)
    ms = (time.perf_counter() - t0) * 1e3
    pred["metrics"] = metrics
    pred["server_latency_ms"] = round(ms, 1)
    brk = hmm_breaks(pred)
    console_line(client, req.trip_id, pred["n_obs"], metrics, ms, breaks=brk, use_hmm=req.use_hmm)
    log.info("client=%s trip=%s use_hmm=%s n_obs=%d metrics=%s hmm_breaks=%s timing_ms=%s latency_ms=%.1f model=%s",
             client, req.trip_id, req.use_hmm, pred["n_obs"], metrics, brk, pred.get("timing_ms"), ms, pred.get("model"))
    return to_jsonable(pred)
