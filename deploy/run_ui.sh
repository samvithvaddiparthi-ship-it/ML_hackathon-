#!/usr/bin/env bash
# PathSnap evaluator UI - Laptop 2 (macOS/Linux). Optional arg: API URL to pre-fill.
cd "$(dirname "$0")/../ui"
URL="http://localhost:5500/${1:+?api=$1}"
( sleep 1; (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") || true ) &
exec python3 -m http.server 5500
