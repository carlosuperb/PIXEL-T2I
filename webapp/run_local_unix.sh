#!/usr/bin/env bash

echo "======================================"
echo " PIXEL-T2I Local Web App (macOS/Linux)"
echo "======================================"

# --- Backend ---
(
  cd webapp/backend || exit
  echo "[Backend] Starting FastAPI..."
  python3 -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
) &

# --- Frontend ---
(
  cd webapp/frontend || exit
  echo "[Frontend] Starting static server..."
  python3 -m http.server 5500
) &

echo
echo "Frontend: http://127.0.0.1:5500"
echo "Backend : http://127.0.0.1:8000/docs"
echo
echo "Press Ctrl+C to stop all services."
wait
