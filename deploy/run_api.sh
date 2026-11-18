#!/usr/bin/env bash
# PathSnap API - Laptop 1 (macOS/Linux). Usage: deploy/run_api.sh ["/path/to/eval trips"]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATHSNAP_EVAL_DATA="${1:-${PATHSNAP_EVAL_DATA:-$HOME/Downloads/Student folder/student_test}}"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
fi
[ -d "$PATHSNAP_EVAL_DATA" ] || { echo "[PathSnap] Trip folder not found: $PATHSNAP_EVAL_DATA"; exit 1; }
echo "[PathSnap] Trips: $PATHSNAP_EVAL_DATA"
exec .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
