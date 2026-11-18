@echo off
REM B7 error analysis (student_test) -> results\analysis. Builds the candidate cache first if it is missing.
cd /d "%~dp0.."
if "%PATHSNAP_DATA%"=="" set "PATHSNAP_DATA=%USERPROFILE%\Downloads\Student folder"
if not exist "results\cache\student_test_cands.pkl" ".venv\Scripts\python.exe" scripts\02_build_dataset.py || goto :end
".venv\Scripts\python.exe" analysis\error_analysis.py
start "" results\analysis\error_analysis.md
:end
pause
