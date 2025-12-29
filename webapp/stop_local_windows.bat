@echo off
echo ======================================
echo  Stopping PIXEL-T2I Web App (Windows)
echo ======================================

REM --- Stop backend (8000) ---
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
  taskkill /PID %%a /F >nul 2>&1
)

REM --- Stop frontend (5500) ---
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5500') do (
  taskkill /PID %%a /F >nul 2>&1
)

echo Done.
pause
