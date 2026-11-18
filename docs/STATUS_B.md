# Status – Account B (Aditya, Laptop 2)

**Stages B1–B5 done (branch `app-eda`).** B1 EDA → `eda/02_eda.py`, `results/eda/`, `results/maps/`.
1. Exists: B2 `viz/folium_maps.py` (`make_trip_map`, `network_segments_near` – contract signature); B3 `api/main.py` per INTERFACE §2 (+ `api/README.md`, `api/test_api.py`); B4 `ui/` (Leaflet bundled, no CDN); B5 `deploy/` (run_api/run_ui .bat/.sh, open_firewall.bat, DEPLOYMENT.md), full `README.md`, `report/report.md` + `report/build_report.py` (→ PDF).
2. Run: `deploy\run_api.bat "<trips folder>"` on Laptop 1, `deploy\run_ui.bat http://<ip>:8000` on Laptop 2; test `python api/test_api.py` – passes for all 10 student_test trips with the baseline (route F1 0.43–0.74, server 30–850 ms).
3. For A: the API calls `predict_trip(p, use_hmm=...)` + `score_trip` unchanged; UI shows Top-1/macro-F1 as "N/A" while they are None and draws `route_coords_classifier_only` (orange dashed) automatically if you add that key. Report §2.2.7, §3–§5 are TODO(A) placeholders in `report/report.md`; README "ML stages" row needs your commands.
4. Open: Report §7 screenshots (UI + Laptop 1 console) must be taken at the real two-laptop run; §6 HMM comparison maps after Stage A HMM lands.
5. Requirements appended: `playwright`, `markdown`.
