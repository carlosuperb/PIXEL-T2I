@echo off
setlocal

echo ======================================
echo  PIXEL-T2I Local Web App (Windows)
echo ======================================

REM ---- Backend ----
start "PIXEL-T2I Backend" cmd /k ^
  "cd /d webapp\backend && python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload"

REM ---- Frontend ----
start "PIXEL-T2I Frontend" cmd /k ^
  "cd /d webapp\frontend && python -m http.server 5500"

echo.
echo Frontend: http://127.0.0.1:5500
echo Backend : http://127.0.0.1:8000/docs
echo.
pause
