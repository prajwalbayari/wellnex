# ─────────────────────────────────────────────────────────────
# Wellnex ML Microservice – FastAPI + TensorFlow / Keras / Sklearn
# ─────────────────────────────────────────────────────────────
import os
import io
import re
import json
import zipfile
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import keras

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ── Custom Keras Objects (needed for model loading) ───────────
@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """Focal Loss for imbalanced classification."""

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self._extra_config = {}
        for key in list(kwargs.keys()):
            if key not in ("reduction", "name", "dtype", "fn"):
                self._extra_config[key] = kwargs.pop(key)
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        config.update(self._extra_config)
        return config


@keras.saving.register_keras_serializable(package="CustomLayers")
class DenseNetPreprocessing(keras.layers.Layer):
    """Model-side preprocessing used by exported DenseNet pipelines."""

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        # Inference pipeline feeds [0,1] images; convert to DenseNet expected range.
        x = x * 255.0
        return keras.applications.densenet.preprocess_input(x)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "FocalLoss": FocalLoss,
    "DenseNetPreprocessing": DenseNetPreprocessing,
    "CustomLayers>DenseNetPreprocessing": DenseNetPreprocessing,
}

from typing import Any

# ── Configuration ─────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))

# Global registries – populated at startup
models: dict[str, Any] = {}
artifacts: dict[str, dict] = {}   # scalers, imputers, feature names, etc.

# Lung cancer class labels (4-class model)
LUNG_CLASSES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]


# ── Model loading helpers ─────────────────────────────────────

def _load_joblib(path: Path):
    return joblib.load(str(path))


def _load_keras(path: Path):
    return keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS)


def _load_disease(name: str):
    """Load model + any preprocessing artefacts for a given disease."""
    disease_dir = MODEL_DIR / name
    if not disease_dir.is_dir():
        print(f"⚠️  Model directory not found: {disease_dir}")
        return

    art: dict[str, object] = {}

    # ── model file ──
    model_path = None
    for ext in (".pkl", ".joblib", ".keras"):
        candidate = disease_dir / f"model{ext}"
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        print(f"⚠️  No model file in {disease_dir}")
        return

    if model_path.suffix == ".keras":
        art["model"] = _load_keras(model_path)
    else:
        art["model"] = _load_joblib(model_path)

    # ── optional artefacts ──
    for artifact_name in ("scaler", "imputer", "feature_names", "feature_stats"):
        for ext in (".pkl", ".joblib"):
            p = disease_dir / f"{artifact_name}{ext}"
            if p.exists():
                art[artifact_name] = _load_joblib(p)
                break

    models[name] = art["model"]
    artifacts[name] = art
    print(f"✅  Loaded {name}: model + {[k for k in art if k != 'model']}")


# ── Startup / Shutdown lifespan ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in ("diabetes", "heart", "lung", "breast"):
        _load_disease(name)
    yield
    models.clear()
    artifacts.clear()


app = FastAPI(title="Wellnex ML Service", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ─────────────────────────────────────────

class DiabetesInput(BaseModel):
    age: float
    gender: str                            # "Female", "Male", "Other"
    ethnicity: str = "White"               # "Asian", "Black", "Hispanic", "Other", "White"
    education_level: str = "Bachelor"      # "Bachelor", "Highschool", "No formal", "Postgraduate"
    income_level: str = "Middle"           # "High", "Low", "Lower-Middle", "Middle", "Upper-Middle"
    employment_status: str = "Employed"    # "Employed", "Retired", "Student", "Unemployed"
    smoking_status: str = "Never"          # "Current", "Former", "Never"
    alcohol_consumption_per_week: float = 0
    physical_activity_minutes_per_week: float = 150
    diet_score: float = 5.0
    sleep_hours_per_day: float = 7
    screen_time_hours_per_day: float = 4
    family_history_diabetes: float = 0     # 0 or 1
    hypertension_history: float = 0        # 0 or 1
    cardiovascular_history: float = 0      # 0 or 1
    bmi: float
    waist_to_hip_ratio: float = 0.85
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float = 75
    cholesterol_total: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    glucose_fasting: float
    glucose_postprandial: float
    insulin_level: float
    hba1c: float
    diabetes_risk_score: float = 0         # 0-10 scale


class HeartInput(BaseModel):
    male: float                # 0 or 1
    age: float
    education: float = 2       # 1-4 (education level)
    currentSmoker: float       # 0 or 1
    cigsPerDay: float = 0
    BPMeds: float = 0          # 0 or 1
    prevalentStroke: float = 0 # 0 or 1
    prevalentHyp: float = 0   # 0 or 1
    diabetes: float = 0        # 0 or 1
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float


class PredictionResult(BaseModel):
    prediction: str
    probability: float


# ── Helper Functions ──────────────────────────────────────────

DIABETES_DEFAULTS: dict[str, Any] = {
    "age": 40.0,
    "gender": "Male",
    "ethnicity": "White",
    "education_level": "Bachelor",
    "income_level": "Middle",
    "employment_status": "Employed",
    "smoking_status": "Never",
    "alcohol_consumption_per_week": 0.0,
    "physical_activity_minutes_per_week": 150.0,
    "diet_score": 5.0,
    "sleep_hours_per_day": 7.0,
    "screen_time_hours_per_day": 4.0,
    "family_history_diabetes": 0.0,
    "hypertension_history": 0.0,
    "cardiovascular_history": 0.0,
    "bmi": 25.0,
    "waist_to_hip_ratio": 0.85,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "heart_rate": 75.0,
    "cholesterol_total": 180.0,
    "hdl_cholesterol": 50.0,
    "ldl_cholesterol": 100.0,
    "triglycerides": 150.0,
    "glucose_fasting": 95.0,
    "glucose_postprandial": 120.0,
    "insulin_level": 10.0,
    "hba1c": 5.6,
    "diabetes_risk_score": 0.0,
}

HEART_DEFAULTS: dict[str, Any] = {
    "male": 1.0,
    "age": 50.0,
    "education": 2.0,
    "currentSmoker": 0.0,
    "cigsPerDay": 0.0,
    "BPMeds": 0.0,
    "prevalentStroke": 0.0,
    "prevalentHyp": 0.0,
    "diabetes": 0.0,
    "totChol": 200.0,
    "sysBP": 130.0,
    "diaBP": 85.0,
    "BMI": 26.0,
    "heartRate": 75.0,
    "glucose": 95.0,
}

DIABETES_REQUIRED_FIELDS = [
    "age",
    "gender",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
]

HEART_REQUIRED_FIELDS = [
    "male",
    "age",
    "currentSmoker",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

DIABETES_CATEGORICAL_FIELDS = {
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
}
DIABETES_BINARY_FIELDS = {
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
}
DIABETES_NUMERIC_FIELDS = set(DIABETES_DEFAULTS.keys()) - DIABETES_CATEGORICAL_FIELDS - DIABETES_BINARY_FIELDS

HEART_BINARY_FIELDS = {
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
}
HEART_NUMERIC_FIELDS = set(HEART_DEFAULTS.keys()) - HEART_BINARY_FIELDS

DIABETES_ALIASES: dict[str, list[str]] = {
    "age": ["age"],
    "bmi": ["bmi", "body mass index"],
    "waist_to_hip_ratio": ["waist to hip ratio", "waist hip ratio", "waist/hip ratio"],
    "systolic_bp": ["systolic bp", "systolic blood pressure", "sbp"],
    "diastolic_bp": ["diastolic bp", "diastolic blood pressure", "dbp"],
    "heart_rate": ["heart rate", "pulse"],
    "glucose_fasting": ["fasting glucose", "glucose fasting", "fbs", "fasting blood sugar"],
    "glucose_postprandial": ["postprandial glucose", "post meal glucose", "pp glucose"],
    "insulin_level": ["insulin", "insulin level"],
    "hba1c": ["hba1c", "hb a1c", "a1c"],
    "cholesterol_total": ["total cholesterol", "cholesterol total", "total chol"],
    "hdl_cholesterol": ["hdl", "hdl cholesterol"],
    "ldl_cholesterol": ["ldl", "ldl cholesterol"],
    "triglycerides": ["triglycerides", "tg"],
    "alcohol_consumption_per_week": ["alcohol", "drinks per week"],
    "physical_activity_minutes_per_week": ["physical activity", "exercise minutes", "activity minutes"],
    "diet_score": ["diet score"],
    "sleep_hours_per_day": ["sleep hours", "sleep per day"],
    "screen_time_hours_per_day": ["screen time", "screen hours"],
    "diabetes_risk_score": ["diabetes risk score", "risk score"],
    "family_history_diabetes": ["family history diabetes", "family history"],
    "hypertension_history": ["hypertension history", "history of hypertension"],
    "cardiovascular_history": ["cardiovascular history", "heart disease history"],
    "gender": ["gender", "sex"],
    "ethnicity": ["ethnicity", "race"],
    "education_level": ["education level", "education"],
    "income_level": ["income level", "income"],
    "employment_status": ["employment status", "employment"],
    "smoking_status": ["smoking status", "smoker"],
}

HEART_ALIASES: dict[str, list[str]] = {
    "male": ["male", "gender", "sex"],
    "age": ["age"],
    "education": ["education", "education level"],
    "currentSmoker": ["current smoker", "smoker"],
    "cigsPerDay": ["cigs per day", "cigarettes per day"],
    "BPMeds": ["bp meds", "blood pressure medication"],
    "prevalentStroke": ["prevalent stroke", "stroke history"],
    "prevalentHyp": ["prevalent hypertension", "hypertension"],
    "diabetes": ["diabetes"],
    "totChol": ["total cholesterol", "totchol"],
    "sysBP": ["systolic bp", "systolic blood pressure", "sbp"],
    "diaBP": ["diastolic bp", "diastolic blood pressure", "dbp"],
    "BMI": ["bmi", "body mass index"],
    "heartRate": ["heart rate", "pulse"],
    "glucose": ["glucose", "blood glucose"],
}

CHOICE_SYNONYMS: dict[str, dict[str, str]] = {
    "gender": {
        "female": "Female",
        "male": "Male",
        "other": "Other",
        "nonbinary": "Other",
        "non-binary": "Other",
    },
    "ethnicity": {
        "asian": "Asian",
        "black": "Black",
        "hispanic": "Hispanic",
        "other": "Other",
        "white": "White",
    },
    "education_level": {
        "bachelor": "Bachelor",
        "highschool": "Highschool",
        "high school": "Highschool",
        "no formal": "No formal",
        "postgraduate": "Postgraduate",
    },
    "income_level": {
        "high": "High",
        "low": "Low",
        "lower-middle": "Lower-Middle",
        "lower middle": "Lower-Middle",
        "middle": "Middle",
        "upper-middle": "Upper-Middle",
        "upper middle": "Upper-Middle",
    },
    "employment_status": {
        "employed": "Employed",
        "retired": "Retired",
        "student": "Student",
        "unemployed": "Unemployed",
    },
    "smoking_status": {
        "current": "Current",
        "former": "Former",
        "never": "Never",
    },
}

LUNG_HINTS = {"lung", "chest", "thorax", "ct", "xray", "x-ray"}
BREAST_HINTS = {"breast", "mammogram", "mammo", "histopath", "biopsy"}


def _alias_to_pattern(alias: str) -> str:
    parts = [re.escape(part) for part in alias.strip().split() if part.strip()]
    return r"\s*".join(parts)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_binary(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 0.5 else 0.0
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "positive", "present"}:
        return 1.0
    if token in {"0", "false", "no", "n", "negative", "absent"}:
        return 0.0
    return None


def _coerce_choice(field: str, value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    synonyms = CHOICE_SYNONYMS.get(field, {})
    if normalized in synonyms:
        return synonyms[normalized]
    for key, canonical in synonyms.items():
        if normalized == key.lower():
            return canonical
    return None


def _extract_number(text: str, aliases: list[str]) -> float | None:
    for alias in aliases:
        label = _alias_to_pattern(alias)
        match = re.search(rf"{label}[^\d\-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _extract_binary(text: str, aliases: list[str]) -> float | None:
    for alias in aliases:
        label = _alias_to_pattern(alias)
        match = re.search(
            rf"{label}[^a-zA-Z0-9]*(yes|no|true|false|positive|negative|present|absent|1|0)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return _coerce_binary(match.group(1))
    return None


def _extract_choice(text: str, field: str, aliases: list[str]) -> str | None:
    synonyms = CHOICE_SYNONYMS.get(field, {})
    for alias in aliases:
        label = _alias_to_pattern(alias)
        for key, canonical in synonyms.items():
            token = _alias_to_pattern(key)
            if re.search(rf"{label}.{{0,30}}\b{token}\b", text, flags=re.IGNORECASE):
                return canonical
    # fallback: global scan
    for key, canonical in synonyms.items():
        token = _alias_to_pattern(key)
        if re.search(rf"\b{token}\b", text, flags=re.IGNORECASE):
            return canonical
    return None


def _decode_text_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded = file_bytes.decode(encoding)
            if decoded.strip():
                return decoded
        except UnicodeDecodeError:
            continue
    return ""


def _extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            with zf.open("word/document.xml") as doc_xml:
                xml_text = doc_xml.read().decode("utf-8", errors="ignore")
        xml_text = re.sub(r"<w:tab\s*/>", " ", xml_text)
        xml_text = re.sub(r"</w:p>", "\n", xml_text)
        plain = re.sub(r"<[^>]+>", "", xml_text)
        return re.sub(r"\n{3,}", "\n\n", plain).strip()
    except Exception:
        return ""


def _extract_text_from_file(file_bytes: bytes, filename: str | None, content_type: str | None) -> str:
    safe_name = (filename or "").lower()
    safe_type = (content_type or "").lower()

    if ("pdf" in safe_type or safe_name.endswith(".pdf")) and PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [(page.extract_text() or "") for page in reader.pages[:20]]
            text = "\n".join(pages).strip()
            if text:
                return text
        except Exception:
            pass

    if safe_name.endswith(".docx"):
        docx_text = _extract_text_from_docx(file_bytes)
        if docx_text:
            return docx_text

    if safe_type.startswith("text/") or safe_name.endswith((".txt", ".csv", ".json", ".xml", ".md", ".yaml", ".yml", ".tsv")):
        return _decode_text_bytes(file_bytes)

    # Best-effort fallback for unknown file types.
    return _decode_text_bytes(file_bytes)


def _extract_inputs_from_text(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    diabetes_raw = {k: None for k in DIABETES_DEFAULTS.keys()}
    heart_raw = {k: None for k in HEART_DEFAULTS.keys()}

    for field in DIABETES_NUMERIC_FIELDS:
        diabetes_raw[field] = _extract_number(text, DIABETES_ALIASES.get(field, [field]))
    for field in DIABETES_BINARY_FIELDS:
        diabetes_raw[field] = _extract_binary(text, DIABETES_ALIASES.get(field, [field]))
    for field in DIABETES_CATEGORICAL_FIELDS:
        diabetes_raw[field] = _extract_choice(text, field, DIABETES_ALIASES.get(field, [field]))

    for field in HEART_NUMERIC_FIELDS:
        heart_raw[field] = _extract_number(text, HEART_ALIASES.get(field, [field]))
    for field in HEART_BINARY_FIELDS:
        heart_raw[field] = _extract_binary(text, HEART_ALIASES.get(field, [field]))

    # Cross-fill from shared values when possible.
    if heart_raw.get("age") is None and diabetes_raw.get("age") is not None:
        heart_raw["age"] = diabetes_raw["age"]
    if heart_raw.get("male") is None and diabetes_raw.get("gender") in {"Male", "Female"}:
        heart_raw["male"] = 1.0 if diabetes_raw["gender"] == "Male" else 0.0

    return diabetes_raw, heart_raw


def _count_populated(fields: dict[str, Any], required_fields: list[str]) -> int:
    return sum(1 for field in required_fields if fields.get(field) is not None)


def _infer_tabular_intent(diabetes_raw: dict[str, Any], heart_raw: dict[str, Any]) -> str:
    d_score = _count_populated(diabetes_raw, DIABETES_REQUIRED_FIELDS)
    h_score = _count_populated(heart_raw, HEART_REQUIRED_FIELDS)

    if d_score == 0 and h_score == 0:
        return "none"
    if d_score >= max(5, h_score + 2):
        return "diabetes"
    if h_score >= max(4, d_score + 2):
        return "heart"
    return "both"


def _apply_supplemental_inputs(
    diabetes_raw: dict[str, Any],
    heart_raw: dict[str, Any],
    supplemental_data: dict[str, Any],
) -> None:
    diabetes_sup = supplemental_data.get("diabetes", {}) if isinstance(supplemental_data, dict) else {}
    heart_sup = supplemental_data.get("heart", {}) if isinstance(supplemental_data, dict) else {}

    if isinstance(diabetes_sup, dict):
        for field, value in diabetes_sup.items():
            if field not in DIABETES_DEFAULTS:
                continue
            if field in DIABETES_NUMERIC_FIELDS:
                coerced = _coerce_number(value)
            elif field in DIABETES_BINARY_FIELDS:
                coerced = _coerce_binary(value)
            else:
                coerced = _coerce_choice(field, value)
            if coerced is not None:
                diabetes_raw[field] = coerced

    if isinstance(heart_sup, dict):
        for field, value in heart_sup.items():
            if field not in HEART_DEFAULTS:
                continue
            if field in HEART_NUMERIC_FIELDS:
                coerced = _coerce_number(value)
            else:
                coerced = _coerce_binary(value)
            if coerced is not None:
                heart_raw[field] = coerced


def _build_resolved_inputs(raw_data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(defaults)
    for key, value in raw_data.items():
        if value is not None:
            resolved[key] = value
    return resolved


def _render_text_to_image(text: str, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        clean_text = "No readable text extracted from uploaded report."

    wrap_width = max(18, width // 7)
    lines = textwrap.wrap(clean_text, width=wrap_width)[:40]
    y = 6
    for line in lines:
        if y >= height - 12:
            break
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 11

    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _is_real_image(file_bytes: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def _infer_cancer_intent(filename: str | None, extracted_text: str) -> str:
    haystack = f"{filename or ''} {extracted_text or ''}".lower()
    has_lung = any(token in haystack for token in LUNG_HINTS)
    has_breast = any(token in haystack for token in BREAST_HINTS)
    if has_lung and not has_breast:
        return "lung"
    if has_breast and not has_lung:
        return "breast"
    return "unknown"


def _prepare_image_array(
    file_bytes: bytes,
    extracted_text: str,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, str]:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0), "uploaded-image"
    except Exception:
        fallback_arr = _render_text_to_image(extracted_text, target_size)
        return fallback_arr, "generated-from-text"

def _ensure(name: str):
    if name not in models:
        raise HTTPException(503, f"Model '{name}' is not loaded.")
    return models[name], artifacts.get(name, {})


def _preprocess_image(file_bytes: bytes, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ────────────────────────────────────────────────────────
# DIABETES  preprocessing  (replicate training notebook)
# ────────────────────────────────────────────────────────

def _diabetes_preprocess(data: DiabetesInput, art: dict) -> np.ndarray:
    """Convert raw form input -> 47-feature scaled vector."""
    row = {
        "age": data.age,
        "gender": data.gender,
        "ethnicity": data.ethnicity,
        "education_level": data.education_level,
        "income_level": data.income_level,
        "employment_status": data.employment_status,
        "smoking_status": data.smoking_status,
        "alcohol_consumption_per_week": data.alcohol_consumption_per_week,
        "physical_activity_minutes_per_week": data.physical_activity_minutes_per_week,
        "diet_score": data.diet_score,
        "sleep_hours_per_day": data.sleep_hours_per_day,
        "screen_time_hours_per_day": data.screen_time_hours_per_day,
        "family_history_diabetes": data.family_history_diabetes,
        "hypertension_history": data.hypertension_history,
        "cardiovascular_history": data.cardiovascular_history,
        "bmi": data.bmi,
        "waist_to_hip_ratio": data.waist_to_hip_ratio,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "heart_rate": data.heart_rate,
        "cholesterol_total": data.cholesterol_total,
        "hdl_cholesterol": data.hdl_cholesterol,
        "ldl_cholesterol": data.ldl_cholesterol,
        "triglycerides": data.triglycerides,
        "glucose_fasting": data.glucose_fasting,
        "glucose_postprandial": data.glucose_postprandial,
        "insulin_level": data.insulin_level,
        "hba1c": data.hba1c,
        "diabetes_risk_score": data.diabetes_risk_score,
    }
    df = pd.DataFrame([row])

    # Separate numeric / categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ["gender", "ethnicity", "education_level",
                        "income_level", "employment_status", "smoking_status"]

    # Replace medical zeros with median (from training stats)
    feature_stats = art.get("feature_stats", {})
    medians = feature_stats.get("numeric_medians", {}) if isinstance(feature_stats, dict) else {}
    zero_not_accepted = [
        "glucose_fasting", "glucose_postprandial", "bmi",
        "insulin_level", "systolic_bp", "diastolic_bp", "heart_rate",
    ]
    for col in numeric_cols:
        if col in zero_not_accepted:
            df[col] = df[col].replace(0, np.nan)
        median_val = medians.get(col)
        if median_val is None:
            col_median = df[col].median()
            median_val = col_median if not np.isnan(col_median) else 0.0
        df[col] = df[col].fillna(median_val)

    # One-hot encode categoricals deterministically (drop_first reference
    # categories must match training: Female, Asian, Bachelor, High, Employed, Current)
    _cat_dummies = {
        "gender":            ["Male", "Other"],
        "ethnicity":         ["Black", "Hispanic", "Other", "White"],
        "education_level":   ["Highschool", "No formal", "Postgraduate"],
        "income_level":      ["Low", "Lower-Middle", "Middle", "Upper-Middle"],
        "employment_status": ["Retired", "Student", "Unemployed"],
        "smoking_status":    ["Former", "Never"],
    }
    for col, categories in _cat_dummies.items():
        val = df[col].iloc[0]
        for cat in categories:
            df[f"{col}_{cat}"] = int(val == cat)
        df.drop(columns=[col], inplace=True)

    # ── Feature engineering ──
    df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 100],
                             labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df["bp_risk"] = ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)
    df["high_glucose"] = (df["glucose_fasting"] >= 126).astype(int)
    df["prediabetic_glucose"] = ((df["glucose_fasting"] >= 100) &
                                  (df["glucose_fasting"] < 126)).astype(int)
    df["hba1c_risk"] = pd.cut(df["hba1c"], bins=[0, 5.7, 6.5, 100],
                              labels=[0, 1, 2]).astype(float).fillna(0)

    # Align to saved feature order
    feature_names = art.get("feature_names", [])
    if feature_names is not None and len(feature_names) > 0:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    # Scale
    scaler = art.get("scaler")
    if scaler is not None:
        return scaler.transform(df)
    return df.values.astype(np.float64)


# ────────────────────────────────────────────────────────
# HEART  preprocessing  (replicate FHS training notebook)
# ────────────────────────────────────────────────────────

def _heart_preprocess(data: HeartInput, art: dict) -> np.ndarray:
    """Convert raw form input -> 63-feature imputed + scaled vector."""
    row = {
        "male": data.male,
        "age": data.age,
        "education": data.education,
        "currentSmoker": data.currentSmoker,
        "cigsPerDay": data.cigsPerDay,
        "BPMeds": data.BPMeds,
        "prevalentStroke": data.prevalentStroke,
        "prevalentHyp": data.prevalentHyp,
        "diabetes": data.diabetes,
        "totChol": data.totChol,
        "sysBP": data.sysBP,
        "diaBP": data.diaBP,
        "BMI": data.BMI,
        "heartRate": data.heartRate,
        "glucose": data.glucose,
    }
    df = pd.DataFrame([row])

    # ── Feature engineering (must match training notebook) ──
    df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
    df["mean_arterial_pressure"] = (df["sysBP"] + 2 * df["diaBP"]) / 3
    df["bp_ratio"] = df["sysBP"] / (df["diaBP"] + 0.001)

    def _bp_stage(sbp, dbp):
        if sbp < 120 and dbp < 80:
            return 0
        elif sbp < 130 and dbp < 80:
            return 1
        elif sbp < 140 or dbp < 90:
            return 2
        else:
            return 3

    df["bp_stage"] = df.apply(lambda r: _bp_stage(r["sysBP"], r["diaBP"]), axis=1)

    def _bmi_cat(bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        elif bmi < 35:
            return 3
        else:
            return 4

    df["bmi_category"] = df["BMI"].apply(_bmi_cat)

    def _age_group(age):
        if age <= 40:
            return 0.0
        elif age <= 50:
            return 1.0
        elif age <= 60:
            return 2.0
        else:
            return 3.0

    df["age_group"] = df["age"].apply(_age_group)
    df["age_squared"] = (df["age"] ** 2) / 100

    # Smoking
    df["cigsPerDay"] = df["cigsPerDay"].fillna(0)
    df["smoking_intensity"] = df["currentSmoker"] * df["cigsPerDay"]
    df["heavy_smoker"] = ((df["cigsPerDay"] >= 20) & (df["currentSmoker"] == 1)).astype(float)

    # Cholesterol
    df["high_cholesterol"] = (df["totChol"] >= 240).astype(float)
    df["borderline_cholesterol"] = ((df["totChol"] >= 200) & (df["totChol"] < 240)).astype(float)

    # Glucose
    df["abnormal_glucose"] = (df["glucose"] >= 100).astype(float)
    df["diabetic_glucose"] = (df["glucose"] >= 126).astype(float)

    # Heart rate
    df["elevated_hr"] = (df["heartRate"] >= 80).astype(float)
    df["low_hr"] = (df["heartRate"] < 60).astype(float)

    # Composite risk scores
    df["risk_factor_count"] = (
        (df["male"] == 1).astype(float)
        + (df["age"] >= 55).astype(float)
        + (df["currentSmoker"] == 1).astype(float)
        + (df["prevalentHyp"] == 1).astype(float)
        + (df["diabetes"] == 1).astype(float)
        + df["high_cholesterol"]
    )
    df["metabolic_risk"] = df["bmi_category"] + df["diabetes"] + df["abnormal_glucose"]
    df["cv_risk_score"] = (
        df["bp_stage"] * 2
        + df["bmi_category"] * 1.5
        + df["risk_factor_count"] * 1
        + df["smoking_intensity"] / 10
    )

    # Interaction features
    df["age_bp_interaction"] = df["age"] * df["sysBP"] / 1000
    df["age_chol_interaction"] = df["age"] * df["totChol"] / 10000
    df["bmi_bp_interaction"] = df["BMI"] * df["sysBP"] / 1000
    df["smoke_age_interaction"] = df["smoking_intensity"] * df["age"] / 100
    df["male_smoker"] = (df["male"] * df["currentSmoker"]).astype(float)
    df["diabetes_hyp_combo"] = (df["diabetes"] * df["prevalentHyp"]).astype(float)

    # Binary flags
    df["is_hypertensive"] = ((df["sysBP"] >= 140) | (df["diaBP"] >= 90)).astype(float)
    df["is_obese"] = (df["BMI"] >= 30).astype(float)
    df["high_risk_age"] = (df["age"] >= 55).astype(float)
    df["high_risk_patient"] = (
        (df["is_hypertensive"] == 1)
        | (df["is_obese"] == 1)
        | (df["diabetes"] == 1)
        | (df["prevalentStroke"] == 1)
    ).astype(float)

    # Polynomial features
    df["sysBP_squared"] = (df["sysBP"] ** 2) / 10000
    df["diaBP_squared"] = (df["diaBP"] ** 2) / 10000
    df["BMI_squared"] = (df["BMI"] ** 2) / 100
    df["glucose_squared"] = (df["glucose"] ** 2) / 10000
    df["totChol_squared"] = (df["totChol"] ** 2) / 100000

    # Ratio features
    df["cholesterol_per_age"] = df["totChol"] / (df["age"] + 1)
    df["glucose_per_BMI"] = df["glucose"] / (df["BMI"] + 1)
    df["heartRate_per_age"] = df["heartRate"] / (df["age"] + 1)

    # Triple interactions
    df["age_bp_chol"] = (df["age"] * df["sysBP"] * df["totChol"]) / 1000000
    df["male_age_smoker"] = df["male"] * df["age"] * df["currentSmoker"]
    df["diabetes_hyp_age"] = df["diabetes"] * df["prevalentHyp"] * df["age"] / 10

    # Metabolic syndrome
    df["metabolic_syndrome_score"] = (
        (df["BMI"] >= 30).astype(float)
        + (df["sysBP"] >= 130).astype(float)
        + (df["diaBP"] >= 85).astype(float)
        + (df["glucose"] >= 100).astype(float)
        + (df["totChol"] >= 200).astype(float)
    ) / 5.0

    # Heart rate extras
    df["hr_deviation"] = np.abs(df["heartRate"] - 75)
    df["hr_risk_zone"] = ((df["heartRate"] < 60) | (df["heartRate"] > 100)).astype(float)

    # Critical thresholds
    df["critical_cholesterol"] = (df["totChol"] >= 260).astype(float)
    df["critical_glucose"] = (df["glucose"] >= 150).astype(float)
    df["critical_bp"] = ((df["sysBP"] >= 160) | (df["diaBP"] >= 100)).astype(float)

    # Age-adjusted scores
    df["age_adjusted_bp"] = (df["sysBP"] - df["age"]) / 10
    df["age_adjusted_chol"] = (df["totChol"] - df["age"] * 2) / 100

    # Lifestyle compound
    df["lifestyle_compound"] = (
        df["male"] * 2
        + df["smoking_intensity"] / 20
        + (1 - df["education"].fillna(2) / 4) * 0.5
    )

    # ── Align to saved feature order ──
    feature_names = art.get("feature_names")
    if feature_names is not None and len(feature_names) > 0:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    # ── Impute then scale ──
    imputer = art.get("imputer")
    if imputer is not None:
        arr = imputer.transform(df)
    else:
        arr = df.values

    scaler = art.get("scaler")
    if scaler is not None:
        arr = scaler.transform(arr)

    return arr.astype(np.float64)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "wellnex-ml",
        "loaded_models": list(models.keys()),
    }


@app.post("/predict/diabetes", response_model=PredictionResult)
async def predict_diabetes(data: DiabetesInput):
    model, art = _ensure("diabetes")
    X = _diabetes_preprocess(data, art)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        prob = float(model.predict(X)[0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return PredictionResult(prediction=label, probability=round(prob, 4))


@app.post("/predict/heart", response_model=PredictionResult)
async def predict_heart(data: HeartInput):
    model, art = _ensure("heart")
    X = _heart_preprocess(data, art)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        prob = float(model.predict(X)[0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return PredictionResult(prediction=label, probability=round(prob, 4))


@app.post("/predict/lung", response_model=PredictionResult)
async def predict_lung(file: UploadFile = File(...)):
    model, _ = _ensure("lung")
    contents = await file.read()
    img = _preprocess_image(contents, target_size=(448, 448))
    preds = model.predict(img, verbose=0)[0]  # shape (4,)
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = LUNG_CLASSES[class_idx]
    prediction = "Negative" if label == "Normal" else "Positive"
    return PredictionResult(
        prediction=f"{prediction} ({label})",
        probability=round(confidence, 4),
    )


@app.post("/predict/breast", response_model=PredictionResult)
async def predict_breast(file: UploadFile = File(...)):
    model, _ = _ensure("breast")
    contents = await file.read()
    img = _preprocess_image(contents, target_size=(160, 160))
    prob = float(model.predict(img, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return PredictionResult(prediction=label, probability=round(prob, 4))


@app.post("/predict/unified")
async def predict_unified(
    file: UploadFile = File(...),
    supplemental_data: str = Form("{}"),
):
    try:
        supplemental = json.loads(supplemental_data) if supplemental_data else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid supplemental_data JSON.")

    contents = await file.read()
    extracted_text = _extract_text_from_file(contents, file.filename, file.content_type)
    is_real_image = _is_real_image(contents)
    cancer_intent = _infer_cancer_intent(file.filename, extracted_text)

    diabetes_raw, heart_raw = _extract_inputs_from_text(extracted_text)
    _apply_supplemental_inputs(diabetes_raw, heart_raw, supplemental)
    tabular_intent = _infer_tabular_intent(diabetes_raw, heart_raw)

    if is_real_image:
        tabular_intent = "none"

    missing_diabetes = [field for field in DIABETES_REQUIRED_FIELDS if diabetes_raw.get(field) is None]
    missing_heart = [field for field in HEART_REQUIRED_FIELDS if heart_raw.get(field) is None]

    if is_real_image:
        missing_diabetes = []
        missing_heart = []
    elif tabular_intent == "diabetes":
        missing_heart = []
    elif tabular_intent == "heart":
        missing_diabetes = []

    missing_fields = {
        "diabetes": missing_diabetes,
        "heart": missing_heart,
    }

    diabetes_input = _build_resolved_inputs(diabetes_raw, DIABETES_DEFAULTS)
    heart_input = _build_resolved_inputs(heart_raw, HEART_DEFAULTS)

    model_results: dict[str, Any] = {}

    # Diabetes model
    if is_real_image:
        model_results["diabetes"] = {
            "status": "skipped",
            "reason": "Image upload: diabetes model expects tabular clinical data.",
        }
    elif tabular_intent == "heart":
        model_results["diabetes"] = {
            "status": "skipped",
            "reason": "Input report appears heart-focused.",
        }
    elif missing_fields["diabetes"]:
        model_results["diabetes"] = {
            "status": "needs_input",
            "missing_fields": missing_fields["diabetes"],
        }
    else:
        try:
            d_model, d_art = _ensure("diabetes")
            d_arr = _diabetes_preprocess(DiabetesInput(**diabetes_input), d_art)
            if hasattr(d_model, "predict_proba"):
                d_prob = float(d_model.predict_proba(d_arr)[0][1])
            else:
                d_prob = float(d_model.predict(d_arr)[0])
            d_label = "Positive" if d_prob >= 0.5 else "Negative"
            model_results["diabetes"] = {
                "status": "success",
                "prediction": d_label,
                "probability": round(d_prob, 4),
                "positive_probability": round(d_prob, 4),
            }
        except Exception as exc:
            model_results["diabetes"] = {
                "status": "error",
                "error": str(exc),
            }

    # Heart model
    if is_real_image:
        model_results["heart"] = {
            "status": "skipped",
            "reason": "Image upload: heart model expects tabular clinical data.",
        }
    elif tabular_intent == "diabetes":
        model_results["heart"] = {
            "status": "skipped",
            "reason": "Input report appears diabetes-focused.",
        }
    elif missing_fields["heart"]:
        model_results["heart"] = {
            "status": "needs_input",
            "missing_fields": missing_fields["heart"],
        }
    else:
        try:
            h_model, h_art = _ensure("heart")
            h_arr = _heart_preprocess(HeartInput(**heart_input), h_art)
            if hasattr(h_model, "predict_proba"):
                h_prob = float(h_model.predict_proba(h_arr)[0][1])
            else:
                h_prob = float(h_model.predict(h_arr)[0])
            h_label = "Positive" if h_prob >= 0.5 else "Negative"
            model_results["heart"] = {
                "status": "success",
                "prediction": h_label,
                "probability": round(h_prob, 4),
                "positive_probability": round(h_prob, 4),
            }
        except Exception as exc:
            model_results["heart"] = {
                "status": "error",
                "error": str(exc),
            }

    image_source = "uploaded-image" if is_real_image else "non-image"

    if not is_real_image:
        model_results["lung"] = {
            "status": "skipped",
            "reason": "Cancer models require a medical image (CT/mammogram/histopathology).",
        }
        model_results["breast"] = {
            "status": "skipped",
            "reason": "Cancer models require a medical image (CT/mammogram/histopathology).",
        }
    else:
        # For real medical images, evaluate both cancer models.
        try:
            lung_model, _ = _ensure("lung")
            lung_img, image_source = _prepare_image_array(contents, extracted_text, (448, 448))
            lung_preds = lung_model.predict(lung_img, verbose=0)[0]
            class_idx = int(np.argmax(lung_preds))
            class_name = LUNG_CLASSES[class_idx]
            class_conf = float(lung_preds[class_idx])
            normal_prob = float(lung_preds[LUNG_CLASSES.index("Normal")])
            lung_positive_prob = 1.0 - normal_prob
            lung_prediction = "Negative (Normal)" if class_name == "Normal" else f"Positive ({class_name})"
            lung_score = float(class_conf * lung_positive_prob)
            if cancer_intent == "lung":
                lung_score += 0.10
            model_results["lung"] = {
                "status": "success",
                "prediction": lung_prediction,
                "probability": round(class_conf, 4),
                "positive_probability": round(lung_positive_prob, 4),
                "selection_score": round(lung_score, 4),
            }
        except Exception as exc:
            model_results["lung"] = {
                "status": "error",
                "error": str(exc),
            }

        try:
            breast_model, _ = _ensure("breast")
            breast_img, image_source = _prepare_image_array(contents, extracted_text, (160, 160))
            breast_prob = float(breast_model.predict(breast_img, verbose=0)[0][0])
            breast_label = "Positive" if breast_prob >= 0.5 else "Negative"
            breast_score = float(breast_prob)
            if cancer_intent == "breast":
                breast_score += 0.10
            model_results["breast"] = {
                "status": "success",
                "prediction": breast_label,
                "probability": round(breast_prob, 4),
                "positive_probability": round(breast_prob, 4),
                "selection_score": round(breast_score, 4),
            }
        except Exception as exc:
            model_results["breast"] = {
                "status": "error",
                "error": str(exc),
            }

    scored_tabular = [
        (disease, float(res.get("selection_score", res["positive_probability"])))
        for disease, res in model_results.items()
        if disease in {"diabetes", "heart"}
        and res.get("status") == "success"
        and isinstance(res.get("positive_probability"), (float, int))
    ]

    cancer_threshold = 0.8 if cancer_intent == "unknown" else 0.5

    scored_cancer = [
        (disease, float(res.get("selection_score", res["positive_probability"])))
        for disease, res in model_results.items()
        if disease in {"lung", "breast"}
        and res.get("status") == "success"
        and isinstance(res.get("positive_probability"), (float, int))
        and float(res.get("selection_score", res.get("positive_probability", 0))) >= cancer_threshold
    ]

    if cancer_intent == "unknown" and len(scored_cancer) >= 2:
        sorted_c = sorted(scored_cancer, key=lambda item: item[1], reverse=True)
        # Avoid arbitrarily choosing between breast/lung when both are similarly high.
        if (sorted_c[0][1] - sorted_c[1][1]) < 0.15:
            scored_cancer = []

    # Strict modality routing.
    if is_real_image:
        scored = scored_cancer
    elif tabular_intent in {"diabetes", "heart", "both"} and scored_tabular:
        scored = scored_tabular
    else:
        scored = scored_tabular

    if scored:
        likely_disease, confidence = max(scored, key=lambda item: item[1])
        confidence = float(confidence)
    else:
        likely_disease, confidence = None, 0.0

    needs_user_input = bool(
        (model_results.get("diabetes", {}).get("status") == "needs_input")
        or (model_results.get("heart", {}).get("status") == "needs_input")
    )

    return {
        "needs_user_input": needs_user_input,
        "summary": {
            "likely_disease": likely_disease,
            "confidence": round(confidence, 4),
            "provisional": needs_user_input,
            "overall_prediction": (
                f"Most likely {likely_disease}" if likely_disease else "Unable to determine"
            ),
        },
        "missing_fields": missing_fields,
        "prefill_inputs": {
            "diabetes": diabetes_raw,
            "heart": heart_raw,
        },
        "resolved_inputs": {
            "diabetes": diabetes_input,
            "heart": heart_input,
        },
        "file_info": {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "cancer_image_source": image_source,
            "is_real_image": is_real_image,
            "cancer_intent": cancer_intent,
            "tabular_intent": tabular_intent,
            "text_extracted": bool(extracted_text.strip()),
        },
        "model_results": model_results,
    }


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
