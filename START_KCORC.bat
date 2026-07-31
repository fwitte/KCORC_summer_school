@echo off
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_kcorc.ps1"

if errorlevel 1 (
    echo.
    echo An error occurred. Please take a screenshot of this window.
    pause
)