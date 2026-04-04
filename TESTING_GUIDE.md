# Wellnex Testing Guide

This guide covers the current production scope:
- Diabetes model
- Heart disease model
- Breast cancer model

## 1. Start Services

### ML Service
`c:/Users/Prajwal/Desktop/Wellnex/WellnexWeb/.venv/Scripts/python.exe ml-service/start_server.py`

### Backend
`cd backend && npm run dev`

### Frontend
`cd frontend && npm run dev`

## 2. Health Check
Call ML health endpoint:

`GET http://127.0.0.1:8001/health`

Expected `loaded_models`:
- `diabetes`
- `heart`
- `breast`

## 3. Breast Prediction API
`POST /predict/breast` with multipart `file`.

Expected:
- HTTP 200
- JSON with `prediction` and `probability`
- Threshold logic uses `BREAST_CANCER_THRESHOLD` from ML config

## 4. Unified Pipeline API
`POST /predict/unified` with multipart:
- `file`
- `supplemental_data` JSON (optional)

Optional manual text payload structure:
```json
{
  "manual_text": {
    "diabetes": "age 50 fasting glucose 130 hba1c 6.5",
    "heart": "age 50 male yes systolic bp 140 totChol 220"
  }
}
```

Expected behavior:
- Image uploads evaluate breast model.
- Tabular extraction + manual text supports diabetes/heart completion flow.
- Response includes `summary`, `missing_fields`, and `model_results`.

## 5. Frontend Build Check
From frontend folder:

`npm run build`

Expected: successful Vite build.

## 6. Backend Syntax Check
From backend folder:

`Get-ChildItem -Recurse -Filter *.js | ForEach-Object { node --check $_.FullName }`

Expected: no syntax errors.

## 7. ML Syntax Check
From ml-service folder:

`python -m py_compile main.py`

Expected: no syntax errors.
