@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows_pipeline.ps1"
if errorlevel 1 pause
