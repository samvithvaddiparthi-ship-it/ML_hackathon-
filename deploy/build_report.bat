@echo off
REM Build report\PathSnap_Report.pdf from report\report.md and open it.
cd /d "%~dp0.."
".venv\Scripts\python.exe" report\build_report.py && start "" report\PathSnap_Report.pdf
pause
