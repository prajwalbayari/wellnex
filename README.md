# Wellnex - Multi-Disease Prediction Platform

Wellnex is a full-stack health prediction platform with three active models:
- Diabetes (tabular)
- Heart Disease (tabular)
- Breast Cancer (image)

## Stack
- Frontend: React + Vite + Tailwind
- Backend: Node.js + Express + MongoDB
- ML Service: FastAPI + scikit-learn + Keras/TensorFlow

## Active Model Inputs
- Diabetes: structured clinical values
- Heart Disease: structured cardiovascular values
- Breast Cancer: medical image (histopathology/mammography workflow)

## Unified Pipeline
A single upload flow is available through the dashboard:
1. User uploads a report/image file.
2. Optional manual textual notes for diabetes and heart can be added.
3. Backend forwards the file and supplemental data to ML `/predict/unified`.
4. ML service extracts text, infers fields, merges manual text, and evaluates relevant active models.

## Repository Structure
- frontend: user interface
- backend: API, auth, prediction persistence
- ml-service: model loading and inference
- TEST_CASES.txt: testing scenarios (kept intentionally)
- TESTING_GUIDE.md: testing steps (kept intentionally)

## Run
### 1) ML service
Use the project virtual environment Python:

`c:/Users/Prajwal/Desktop/Wellnex/WellnexWeb/.venv/Scripts/python.exe ml-service/start_server.py`

### 2) Backend
`cd backend && npm run dev`

### 3) Frontend
`cd frontend && npm run dev`

## Notes
- Only diabetes, heart, and breast are supported in production.
- The project is currently scoped to diabetes, heart disease, and breast cancer models.
