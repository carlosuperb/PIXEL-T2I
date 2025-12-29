# PIXEL-T2I Web Application (Local)

This directory contains the **local web interface** for the PIXEL-T2I project.
The web application provides an interactive frontend and a FastAPI backend
for running diffusion-based pixel sprite generation models.

Due to the computational cost of diffusion inference, this web application
is designed to run **locally**, preferably with **GPU acceleration**.

---

## Directory Structure

```
webapp/
├── frontend/ # Static web UI (HTML / CSS / JavaScript)
├── backend/ # FastAPI backend (model loading & inference)
├── web_cache/ # Generated image cache (runtime outputs)
├── README.md # This file
├── run_local_unix.sh # Start script (macOS / Linux)
├── run_local_windows.bat
├── stop_local_unix.sh # Stop script (macOS / Linux)
└── stop_local_windows.bat
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for reasonable inference speed)
- All required Python packages installed  
  (see project-level `requirements.txt` or environment setup)

---

## Local Deployment

The frontend and backend are served as separate services and are started
together using the provided scripts located in the `webapp/` directory.

### Start (Windows)

```bash
run_local_windows.bat
```

### Start (macOS / Linux)

```bash
chmod +x run_local_unix.sh
./run_local_unix.sh
```
---

## Stop Services

### Stop (Windows)

```
stop_local_windows.bat
```

### Stop (macOS/Linux)

```
chmod +x stop_local_unix.sh
./stop_local_unix.sh
```

---

## Manual Startup (Optional)

### Backend (FastAPI)

```bash
cd webapp/backend
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend (Static Server)

```bash
cd webapp/frontend
python -m http.server 5500
```

---

## Access

- **Frontend UI:**  
  http://127.0.0.1:5500

- **Backend API (Swagger docs):**  
  http://127.0.0.1:8000/docs

---

## Notes

- The backend loads multiple diffusion models during startup.  
  Initial launch time may be significant depending on hardware capabilities,
  particularly GPU availability and memory.

- All generated images are cached at runtime under  
  `webapp/web_cache/`.

- The frontend communicates with the backend exclusively via HTTP requests
  on port `8000`.

- This web application is intended for **local development, experimentation,
  and demonstration purposes**, and is not designed for public or production
  deployment.

---