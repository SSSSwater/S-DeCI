@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0generate_sdeci_architecture_visio.ps1" %*
pause
