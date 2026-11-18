@echo off
REM Commit everything changed on app-eda and push. Usage: deploy\push.bat "commit message"
cd /d "%~dp0.."
set "MSG=%~1"
if "%MSG%"=="" set "MSG=Update from Laptop 2"
git checkout app-eda || goto :end
git add -A
git commit -m "%MSG%"
git push -u origin app-eda
:end
pause
