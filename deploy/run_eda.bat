@echo off
REM Re-run the EDA (charts -> results\eda, Folium maps + screenshots -> results\maps).
cd /d "%~dp0.."
if "%PATHSNAP_DATA%"=="" set "PATHSNAP_DATA=%USERPROFILE%\Downloads\Student folder"
".venv\Scripts\python.exe" eda\02_eda.py
pause
