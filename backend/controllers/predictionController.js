// ─────────────────────────────────────────────────────────────
// Prediction Controller
// Handles tabular + image-based disease predictions
// ─────────────────────────────────────────────────────────────
import asyncHandler from "express-async-handler";
import axios from "axios";
import FormData from "form-data";

import Prediction from "../models/Prediction.js";
import { uploadToCloudinary } from "../config/cloudinary.js";

const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:8001";
const ML_TIMEOUT_MS = Number(process.env.ML_SERVICE_TIMEOUT_MS || 20000);
const SUPPORTED_DISEASES = ["diabetes", "heart", "breast"];

const _firstDetail = (detail) => {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const head = detail[0];
    if (typeof head === "string") return head;
    if (head?.msg) return String(head.msg);
    return JSON.stringify(head);
  }
  if (typeof detail === "object" && detail?.msg) return String(detail.msg);
  return null;
};

const _throwMappedMlError = (err, res, fallbackMessage) => {
  if (axios.isAxiosError(err)) {
    if (err.code === "ECONNABORTED") {
      res.status(504);
      throw new Error("ML service request timed out");
    }

    if (err.response) {
      const status = Number(err.response.status || 502);
      const detail = _firstDetail(err.response.data?.detail);
      const message = err.response.data?.message || detail || fallbackMessage;
      res.status(status);
      throw new Error(String(message));
    }

    res.status(503);
    throw new Error("ML service unavailable");
  }

  res.status(502);
  throw new Error(fallbackMessage);
};

const _throwMappedCloudinaryError = (err, res) => {
  if (err?.message?.includes("Missing Cloudinary configuration")) {
    res.status(500);
    throw new Error(
      "Image storage is not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET"
    );
  }

  res.status(502);
  throw new Error("Failed to upload image to Cloudinary");
};

const _uploadImageIfPresent = async (file, folder) => {
  if (!file?.buffer || !String(file.mimetype || "").startsWith("image/")) {
    return null;
  }

  return uploadToCloudinary(file.buffer, folder);
};

// ── Tabular Prediction (Diabetes / Heart) ────────────────────
/**
 * @route   POST /api/predictions/tabular
 * @desc    Send tabular input to ML service, save result
 * @access  Private
 */
export const predictTabular = asyncHandler(async (req, res) => {
  const { diseaseType, inputData } = req.body;

  if (!["diabetes", "heart"].includes(diseaseType)) {
    res.status(400);
    throw new Error("diseaseType must be 'diabetes' or 'heart'");
  }

  if (!inputData || typeof inputData !== "object" || Array.isArray(inputData)) {
    res.status(400);
    throw new Error("inputData must be a JSON object");
  }

  // Forward to ML microservice
  let mlResponse;
  try {
    mlResponse = await axios.post(`${ML_URL}/predict/${diseaseType}`, inputData, {
      timeout: ML_TIMEOUT_MS,
    });
  } catch (err) {
    _throwMappedMlError(err, res, "Failed to get tabular prediction from ML service");
  }

  const { prediction, probability } = mlResponse.data;

  // Persist prediction
  const record = await Prediction.create({
    userId: req.user._id,
    diseaseType,
    inputData,
    predictionResult: prediction,
    probability,
  });

  res.status(201).json(record);
});

// ── Image Prediction (Breast Cancer only) ────────────────────
/**
 * @route   POST /api/predictions/image
 * @desc    Upload image → Cloudinary, send to ML, save result
 * @access  Private
 */
export const predictImage = asyncHandler(async (req, res) => {
  const { diseaseType } = req.body;

  if (!["breast"].includes(diseaseType)) {
    res.status(400);
    throw new Error("diseaseType must be 'breast'");
  }

  if (!req.file) {
    res.status(400);
    throw new Error("Image file is required");
  }

  let cloudinaryUpload = null;
  try {
    cloudinaryUpload = await _uploadImageIfPresent(req.file, "wellnex/breast");
  } catch (err) {
    _throwMappedCloudinaryError(err, res);
  }

  // Forward image buffer to ML microservice as multipart
  const form = new FormData();
  form.append("file", req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype,
  });

  let mlResponse;
  try {
    mlResponse = await axios.post(`${ML_URL}/predict/${diseaseType}`, form, {
      headers: form.getHeaders(),
      timeout: ML_TIMEOUT_MS,
      maxBodyLength: Infinity,
    });
  } catch (err) {
    _throwMappedMlError(err, res, "Failed to get image prediction from ML service");
  }

  const { prediction, probability } = mlResponse.data;

  // Persist image prediction in history for consistency across model types.
  const record = await Prediction.create({
    userId: req.user._id,
    diseaseType,
    inputData: {},
    imageUrl: cloudinaryUpload?.secure_url || null,
    imagePublicId: cloudinaryUpload?.public_id || null,
    predictionResult: prediction,
    probability,
  });

  res.status(201).json({
    ...record.toObject(),
    uploadedImage: cloudinaryUpload
      ? {
          url: cloudinaryUpload.secure_url,
          publicId: cloudinaryUpload.public_id,
        }
      : null,
  });
});

// ── Unified Prediction (Any Report Format) ──────────────────
/**
 * @route   POST /api/predictions/unified
 * @desc    Upload any report file, run unified pipeline (diabetes + heart + breast), optionally ask for missing fields
 * @access  Private
 */
export const predictUnified = asyncHandler(async (req, res) => {
  if (!req.file) {
    res.status(400);
    throw new Error("Report file is required");
  }

  let supplementalData = {};
  if (req.body?.supplementalData) {
    try {
      supplementalData = JSON.parse(req.body.supplementalData);
    } catch {
      res.status(400);
      throw new Error("supplementalData must be valid JSON");
    }
  }

  const form = new FormData();
  form.append("file", req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype,
  });
  form.append("supplemental_data", JSON.stringify(supplementalData));

  let mlResponse;
  try {
    mlResponse = await axios.post(`${ML_URL}/predict/unified`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      timeout: ML_TIMEOUT_MS,
    });
  } catch (err) {
    _throwMappedMlError(err, res, "Failed to get unified prediction from ML service");
  }

  const analysis = mlResponse.data;
  const likelyDisease = analysis?.summary?.likely_disease;
  const isImageReport = String(req.file.mimetype || "").startsWith("image/");

  let cloudinaryUpload = null;
  if (isImageReport) {
    try {
      cloudinaryUpload = await _uploadImageIfPresent(req.file, "wellnex/reports");
    } catch (err) {
      _throwMappedCloudinaryError(err, res);
    }
  }

  const likelyResult = analysis?.model_results?.[likelyDisease];
  const canPersist =
    SUPPORTED_DISEASES.includes(likelyDisease) &&
    likelyResult?.status === "success";

  let savedPrediction = null;

  if (canPersist) {
    const confidence = Number(analysis?.summary?.confidence ?? 0);
    const modelLabel = String(likelyResult?.prediction || analysis?.summary?.prediction || "Negative");
    const predictionResult = `${modelLabel} (${likelyDisease})`;

    let inputData = {};
    if (["diabetes", "heart"].includes(likelyDisease)) {
      inputData = analysis?.resolved_inputs?.[likelyDisease] || {};
    }

    savedPrediction = await Prediction.create({
      userId: req.user._id,
      diseaseType: likelyDisease,
      inputData,
      imageUrl: cloudinaryUpload?.secure_url || null,
      imagePublicId: cloudinaryUpload?.public_id || null,
      predictionResult,
      probability: confidence,
    });
  }

  res.status(savedPrediction ? 201 : 200).json({
    ...analysis,
    savedPrediction,
  });
});

// ── Get Prediction History ───────────────────────────────────
/**
 * @route   GET /api/predictions/history
 * @desc    Return all predictions for the logged-in user
 * @access  Private
 */
export const getHistory = asyncHandler(async (req, res) => {
  const predictions = await Prediction.find({ userId: req.user._id }).sort({
    createdAt: -1,
  }).lean();
  res.json(predictions);
});

// ── Delete a Single Prediction ───────────────────────────────
/**
 * @route   DELETE /api/predictions/:id
 * @desc    Delete a single prediction (only if owned by user)
 * @access  Private
 */
export const deletePrediction = asyncHandler(async (req, res) => {
  const prediction = await Prediction.findById(req.params.id);

  if (!prediction) {
    res.status(404);
    throw new Error("Prediction not found");
  }

  if (prediction.userId.toString() !== req.user._id.toString()) {
    res.status(403);
    throw new Error("Not authorised to delete this prediction");
  }

  await prediction.deleteOne();
  res.json({ message: "Prediction deleted" });
});
