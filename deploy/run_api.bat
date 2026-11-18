@echo off
REM PathSnap API - Laptop 1 (Windows). Usage: deploy\run_api.bat ["C:\path\to\eval trips"]
setlocal
cd /d "%~dp0.."
if not "%~1"=="" set "PATHSNAP_EVAL_DATA=%~1"
if "%PATHSNAP_EVAL_DATA%"=="" set "PATHSNAP_EVAL_DATA=%USERPROFILE%\Downloads\Student folder\student_test"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv || goto :err
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
)
if not exist "%PATHSNAP_EVAL_DATA%" (
  echo [PathSnap] Trip folder not found: "%PATHSNAP_EVAL_DATA%"
  echo            Pass it as the first argument or set PATHSNAP_EVAL_DATA.
  goto :err
)
echo [PathSnap] Trips: %PATHSNAP_EVAL_DATA%
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
goto :eof
:err
pause
exit /b 1
