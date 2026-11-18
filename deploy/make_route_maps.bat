@echo off
REM B8 route-comparison maps (GPS, network, reference, classifier-only, HMM) for all 10 student_test trips.
REM Optional: deploy\make_route_maps.bat 00000077   -> one trip only
cd /d "%~dp0.."
if "%PATHSNAP_DATA%"=="" set "PATHSNAP_DATA=%USERPROFILE%\Downloads\Student folder"
if "%~1"=="" (
  ".venv\Scripts\python.exe" viz\make_route_maps.py --all-student-test
  start "" results\maps
) else (
  ".venv\Scripts\python.exe" viz\make_route_maps.py --trip-dir "%PATHSNAP_DATA%\student_test\%~1" --out "results\maps\%~1_routes.html"
  start "" "results\maps\%~1_routes.html"
)
pause
