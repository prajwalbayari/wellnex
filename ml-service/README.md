---
title: Wellnex ML Service
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Wellnex ML Service

This Space hosts the Wellnex FastAPI inference service for diabetes, heart disease, breast cancer, and unified report analysis.

## Required repository layout
The following files must be at the Space repository root:
- `Dockerfile`
- `main.py`
- `start_server.py`
- `requirements.txt`
- `models/`

## Endpoints
- `GET /health`
- `POST /predict/diabetes`
- `POST /predict/heart`
- `POST /predict/breast`
- `POST /predict/unified`

## Runtime
- Python 3.11
- Docker Space
- Port `7860` in Hugging Face Spaces

## Environment variables
- `MODEL_DIR=./models`
- `POSITIVE_THRESHOLD=0.50`
- `DIABETES_POSITIVE_THRESHOLD=0.40`
- `HEART_POSITIVE_THRESHOLD=0.20`
- `BREAST_CANCER_THRESHOLD=0.50`
- `BREAST_CALIBRATION_ENABLED=false`
- `CORS_ORIGINS=*`
