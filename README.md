# Wellnex - Multi-Disease Prediction Platform

![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB?logo=react&logoColor=white)
![Node](https://img.shields.io/badge/Backend-Node.js%20%2B%20Express-339933?logo=node.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/ML%20Service-FastAPI-009688?logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?logo=mongodb&logoColor=white)

Wellnex is a full-stack multi-disease prediction system that combines:
- A React web app for authentication, report upload, manual clinical input, and prediction history
- A Node.js/Express API for auth, orchestration, and persistence
- A FastAPI ML microservice for diabetes, heart disease, breast cancer, and unified report analysis

The current production scope includes three model domains:
- Diabetes (tabular)
- Heart disease (tabular)
- Breast cancer (medical image)

## Table of Contents
- [Overview](#overview)
- [Current Features](#current-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Local Installation and Setup](#local-installation-and-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment Notes](#deployment-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview
Wellnex supports both guided and direct prediction workflows:
- Unified report analysis: upload a report file once and run cross-model analysis
- Manual tabular predictions: submit detailed diabetes or heart form inputs
- Breast image prediction: submit a supported medical image file
- Prediction history: save, view, and delete previous predictions per authenticated user

The backend enforces JWT-based route protection for prediction endpoints and stores prediction records in MongoDB.

## Current Features
- User signup, login, JWT authentication, and protected profile endpoint
- Global frontend auth state with auto-logout handling on `401`
- Unified upload workflow (`/predictions/unified`) with:
	- report file upload support (PDF, TXT, CSV, JSON, XML, DOCX, JPG/PNG/WEBP)
	- optional manual text hints (`manual_text`) for diabetes and heart
	- missing-field detection and prefill guidance for tabular completion
- Tabular prediction endpoint for diabetes and heart
- Image prediction endpoint for breast cancer
- Prediction history list and delete per logged-in user
- Backend hardening:
	- CORS allowlist from environment
	- Helmet security headers
	- Rate limiting
	- Request body limits
	- Structured error handling including Multer upload errors
- ML service startup model loading with health endpoint and threshold tuning via environment variables

## Architecture
1. Frontend sends authenticated requests to backend API.
2. Backend validates auth and input, then forwards model inference requests to ML service.
3. ML service runs disease-specific or unified inference and returns prediction payloads.
4. Backend persists eligible prediction results to MongoDB and returns response to frontend.

High-level flow:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:5000/api`
- ML service: `http://localhost:8001`

## Tech Stack
### Frontend
- React 18
- React Router DOM 6
- Axios
- Tailwind CSS 3
- Vite 6
- react-hot-toast, react-icons

### Backend
- Node.js (ES modules)
- Express 4
- Mongoose 8
- JWT (`jsonwebtoken`)
- bcryptjs
- Axios + form-data (backend-to-ML forwarding)
- Multer (multipart uploads)
- Helmet
- express-rate-limit
- CORS

### ML Service
- FastAPI + Uvicorn
- TensorFlow/Keras
- scikit-learn
- LightGBM
- CatBoost (if installed in environment)
- NumPy, Pandas, Joblib
- Pillow
- pypdf

### Database
- MongoDB (Atlas or self-hosted)

## Project Structure
```text
WellnexWeb/
	backend/
		config/
			db.js
			cloudinary.js
		controllers/
			authController.js
			predictionController.js
		middleware/
			authMiddleware.js
			errorMiddleware.js
			upload.js
		models/
			User.js
			Prediction.js
		routes/
			authRoutes.js
			predictionRoutes.js
		server.js
		package.json
	frontend/
		src/
			components/
			context/
			pages/
			services/
			App.jsx
			main.jsx
		package.json
	ml-service/
		models/
		main.py
		start_server.py
		requirements.txt
	reports/
	README.md
```

## Environment Variables
Create `.env` files in each service directory.

### backend/.env
Required:
```env
NODE_ENV=development
PORT=5000
MONGO_URI=<your_mongodb_connection_string>
JWT_SECRET=<your_long_random_secret>
ML_SERVICE_URL=http://localhost:8001
```

Recommended:
```env
CORS_ORIGIN=http://localhost:5173
ML_SERVICE_TIMEOUT_MS=20000
RATE_LIMIT_MAX=300
```

Optional (only if Cloudinary uploads are used):
```env
CLOUDINARY_CLOUD_NAME=<cloud_name>
CLOUDINARY_API_KEY=<api_key>
CLOUDINARY_API_SECRET=<api_secret>
```

### frontend/.env
```env
VITE_API_URL=http://localhost:5000
VITE_API_TIMEOUT_MS=20000
```

### ml-service/.env
```env
MODEL_DIR=./models
POSITIVE_THRESHOLD=0.50
DIABETES_POSITIVE_THRESHOLD=0.40
HEART_POSITIVE_THRESHOLD=0.20
BREAST_CANCER_THRESHOLD=0.50
BREAST_CALIBRATION_ENABLED=false
CORS_ORIGINS=*
```

Security note:
- Never commit real secrets in `.env` files.
- Rotate any credentials that were ever exposed in git history.

## Local Installation and Setup
### Prerequisites
- Node.js 18+
- npm 9+
- Python 3.10+ (project currently uses local `.venv`)
- MongoDB database (Atlas or local)

### 1) Clone and open the project
```bash
git clone <your-repo-url>
cd WellnexWeb
```

### 2) Install frontend and backend dependencies
```bash
cd backend
npm install

cd ../frontend
npm install
```

### 3) Install ML dependencies
```bash
cd ../ml-service
pip install -r requirements.txt
```

### 4) Configure environment files
- Add `.env` in `backend/`, `frontend/`, and `ml-service/` using the variables above.

### 5) Start services (recommended order)
Start ML service:
```bash
cd ml-service
python start_server.py
```

Start backend API:
```bash
cd backend
npm run dev
```

Start frontend:
```bash
cd frontend
npm run dev
```

### 6) Verify health
- Backend: `GET http://localhost:5000/api/health`
- ML service: `GET http://localhost:8001/health`

## Usage
1. Open the frontend URL shown by Vite (typically `http://localhost:5173`).
2. Create an account or log in.
3. Go to dashboard and choose one of the modes:
	 - Upload report (unified analysis)
	 - Fill details manually (diabetes/heart)
4. Review prediction output and confidence.
5. Check the Prediction History table for saved records.
6. Delete history items if needed.

## API Reference
Base URL: `http://localhost:5000/api`

### Auth
- `POST /auth/signup` - Register a new user
- `POST /auth/login` - Login and receive JWT
- `GET /auth/profile` - Get profile of authenticated user

### Predictions (JWT required)
- `POST /predictions/tabular` - Predict diabetes or heart from JSON input
- `POST /predictions/image` - Predict breast cancer from uploaded image (`multipart/form-data`)
- `POST /predictions/unified` - Unified analysis from uploaded report file (`multipart/form-data`)
- `GET /predictions/history` - List prediction history for current user
- `DELETE /predictions/:id` - Delete a prediction owned by current user

### Health
- `GET /health` on ML service (`http://localhost:8001/health`)
- `GET /api/health` on backend (`http://localhost:5000/api/health`)

### ML Service Endpoints (direct)
Base URL: `http://localhost:8001`
- `POST /predict/diabetes`
- `POST /predict/heart`
- `POST /predict/breast`
- `POST /predict/unified`

## Deployment Notes
Recommended deployment split:
- Frontend: static hosting (Vercel/Netlify)
- Backend: Node runtime (Render/Railway/Fly.io)
- ML service: Python runtime (Render/Railway/Fly.io)
- Database: MongoDB Atlas

Production checklist:
1. Set all environment variables in each deployment target.
2. Set backend `CORS_ORIGIN` to your deployed frontend domain.
3. Set frontend `VITE_API_URL` to deployed backend URL.
4. Set backend `ML_SERVICE_URL` to deployed ML URL.
5. Ensure ML `MODEL_DIR` contains required model artifacts.
6. Confirm health endpoints before enabling public traffic.

## Troubleshooting
- `401 Not authorised`: verify JWT token in local storage and backend `JWT_SECRET`.
- `ML service unavailable`: verify ML server is running and backend `ML_SERVICE_URL` is correct.
- Upload rejected: check allowed file types and file size limits.
- CORS issues: confirm frontend origin is included in backend `CORS_ORIGIN`.
- Empty or partial unified result: provide supplemental fields/manual text and retry.

## Contributing
1. Create a feature branch.
2. Keep changes scoped and documented.
3. Run local health checks and smoke tests before opening PR.
4. Open a pull request with a concise description of changes and test evidence.
