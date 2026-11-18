@echo off
REM Repeated inference benchmark: untuned vs tuned XGBoost -> results\analysis\inference_benchmark.csv
cd /d "%~dp0.."
if "%PATHSNAP_DATA%"=="" set "PATHSNAP_DATA=%USERPROFILE%\Downloads\Student folder"
if not exist "results\cache\student_test_cands.pkl" ".venv\Scripts\python.exe" scripts\02_build_dataset.py
".venv\Scripts\python.exe" analysis\benchmark_inference.py
pause
