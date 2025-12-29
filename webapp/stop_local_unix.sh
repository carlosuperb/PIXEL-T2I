#!/usr/bin/env bash

echo "======================================"
echo " Stopping PIXEL-T2I Web App (Unix)"
echo "======================================"

# Kill backend (8000)
lsof -ti :8000 | xargs -r kill

# Kill frontend (5500)
lsof -ti :5500 | xargs -r kill

echo "Done."
