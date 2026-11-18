@echo off
REM PathSnap evaluator UI - Laptop 2 (Windows). Serves ui\ on http://localhost:5500 and opens the browser.
REM Optional: deploy\run_ui.bat http://192.168.1.20:8000   (pre-fills the API URL)
cd /d "%~dp0..\ui"
set "URL=http://localhost:5500/"
if not "%~1"=="" set "URL=http://localhost:5500/?api=%~1"
start "" "%URL%"
python -m http.server 5500
