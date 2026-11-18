@echo off
REM One-time setup: create .venv and install everything. Double-click or run from any folder.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
".venv\Scripts\playwright.exe" install chromium
echo.
echo [PathSnap] Environment ready.
pause
exit /b 0
:err
echo [PathSnap] pip install failed - copy the error above to Claude.
pause
exit /b 1
