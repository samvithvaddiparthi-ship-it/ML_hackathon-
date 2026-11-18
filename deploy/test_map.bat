@echo off
REM Make one Folium map (default trip 00000052 from student_test) and open it. Optional arg: trip id.
cd /d "%~dp0.."
set "TRIP=%~1"
if "%TRIP%"=="" set "TRIP=00000052"
if "%PATHSNAP_DATA%"=="" set "PATHSNAP_DATA=%USERPROFILE%\Downloads\Student folder"
".venv\Scripts\python.exe" -c "import sys,os; sys.path[:0]=['.','src']; from pathlib import Path; from pathsnap.io import load_trip; from viz.folium_maps import make_trip_map, network_segments_near; t=load_trip(Path(os.environ['PATHSNAP_DATA'])/'student_test'/'%TRIP%'); make_trip_map(t.track[['lat','lon']].to_numpy(), network_segments_near(t), title='Trip %TRIP% [student_test]', out_html=os.path.join(os.environ['TEMP'],'pathsnap_test_map.html')); print('map written')" || goto :end
start "" "%TEMP%\pathsnap_test_map.html"
:end
pause
