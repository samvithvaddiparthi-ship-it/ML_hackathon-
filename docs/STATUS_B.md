# Status – Account B (Aditya, Laptop 2)

**Stage B1 (EDA, Report §2.2) – done, awaiting OK.**
1. Exists: `eda/02_eda.py` → 11 figures + `eda_summary.md` + CSVs in `results/eda/`; Folium maps `results/maps/eda_train_00000075.html`, `eda_student_test_00000052.html`. Reusable map builder `viz/folium_maps.py` (`make_trip_map`, `network_segments_near` – B2 signature) is already in place.
2. Run: `PATHSNAP_DATA="/path/to/Student folder" .venv/bin/python eda/02_eda.py` (~25 s). Every figure title states its partition (train / student_test); evaluator data is never read.
3. Finding for A: pooled train GPS→route p95 is 140 m only because of truncated-route trips 02/33/54; excluding them median 4.9 m / p95 17.6 m; per-trip p95 median 14.2 m (train), 15.8 m (student_test) → use for candidate radius / emission σ.
4. Open: map PNG screenshots need a machine with internet (Leaflet CDN + OSM tiles) – rerun the script locally, it writes `results/maps/*.png` automatically via Playwright (`playwright install chromium` if no Chrome).
5. Network-connectivity chart (fig11) reads `results/stage1/trip_inventory.csv` (A's Stage 1 output) rather than recomputing.
