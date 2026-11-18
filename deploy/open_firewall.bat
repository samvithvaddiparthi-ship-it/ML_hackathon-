@echo off
REM Run ONCE on Laptop 1 as Administrator (right-click -> Run as administrator).
REM Allows the UI laptop to reach the API on TCP port 8000.
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
pause
