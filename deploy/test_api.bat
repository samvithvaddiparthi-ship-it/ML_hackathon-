@echo off
REM Smoke-test a RUNNING API (start deploy\run_api.bat first). Optional arg: API URL.
cd /d "%~dp0.."
set "URL=%~1"
if "%URL%"=="" set "URL=http://localhost:8000"
".venv\Scripts\python.exe" api\test_api.py %URL%
pause
