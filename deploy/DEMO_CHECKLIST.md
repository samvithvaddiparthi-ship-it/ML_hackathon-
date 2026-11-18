# Demo-day checklist (two laptops)

Laptop 1 = model + API (Samvith). Laptop 2 = evaluator UI (Aditya). Tick every box at the rehearsal and again
on the day. Detailed explanations: `deploy/DEPLOYMENT.md`.

## The day before (rehearsal)
- [ ] Both laptops on `main` with the final model: `git pull`, then on Laptop 1 `deploy\setup_env.bat` once.
- [ ] Laptop 1: `deploy\open_firewall.bat` → right-click → **Run as administrator** (one time).
- [ ] Phone hotspot ON (name/password noted below). Both laptops join it. Campus Wi-Fi often blocks
      laptop-to-laptop traffic – use the hotspot.
- [ ] Laptop 1: `deploy\run_api.bat "C:\Users\<you>\Downloads\Student folder\student_test"`
      → banner shows `(10 trips)` and `UI API URL : http://<IP>:8000`. Write the IP below.
- [ ] Laptop 2: open `http://<IP>:8000/health` in the browser → `{"status":"ok",...}`.
      If it does not load: firewall (step 2) or different networks.
- [ ] Laptop 2: `deploy\run_ui.bat http://<IP>:8000` → green status dot, 10 trips listed.
- [ ] Run **all 10** student-test trips in the UI; each prints one `[PathSnap]` line on Laptop 1.
- [ ] Laptop 2: `deploy\test_api.bat http://<IP>:8000` → `ALL CHECKS PASSED`.
- [ ] Screenshots for Report §7 → save into `report/figures/`:
  - [ ] `ui_screenshot.png` – UI after a run (map + metric cards visible), Laptop 2.
  - [ ] `api_console.png` – Laptop 1 console with several `[PathSnap]` lines.
- [ ] Note the slowest trip's latency (largest `server` ms in the UI): ________ ms (must be well under 120 s).

| Item | Value |
|---|---|
| Hotspot name / password | |
| Laptop 1 IP on hotspot | |
| Evaluator trips folder on Laptop 1 | `C:\PathSnap\evaluator_test` |

## On the day
1. Hotspot on → both laptops join → **charge both laptops**, disable sleep (Settings → System → Power & battery → Screen and sleep → Never).
2. Laptop 1: unzip the evaluator trips so the folder holds 10 sub-folders `<id>\<id>.track ...`
   (e.g. `C:\PathSnap\evaluator_test\00000015\00000015.track`).
3. Laptop 1: `deploy\run_api.bat "C:\PathSnap\evaluator_test"` → check banner says **10 trips** and note the IP.
4. Laptop 2: `deploy\run_ui.bat http://<IP>:8000` → status dot green, the 10 evaluator ids listed.
5. Let the evaluator pick trips. Keep the Laptop 1 console visible – it is part of the evaluation.
6. Do **not** run experiments on the evaluator trips beyond what the evaluator asks for.

## Fallbacks
| Problem | Fallback |
|---|---|
| Hotspot / Wi-Fi blocks the laptops | second phone's hotspot; or connect both laptops with an Ethernet cable (Windows assigns 169.254.x.x addresses – use Laptop 1's from `ipconfig`) |
| Laptop 2 cannot reach Laptop 1 at all | run the UI on Laptop 1 too: `deploy\run_ui.bat http://localhost:8000` and explain the network issue |
| Map background grey | no internet → tiles missing; GPS, route and metrics still work (Leaflet is bundled) |
| API window closed / crashed | re-run `deploy\run_api.bat ...`; the UI just needs **Connect** again |
| Port 8000 in use | close other uvicorn windows; or edit `--port` in `run_api.bat` and the UI URL |
| Laptop 1 dies | Laptop 2 has the repo: `deploy\setup_env.bat`, then run API + UI locally on Laptop 2 |
