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
const SUPPORTED_DISEASES = ["diabetes", "heart", "lung", "breast"];

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

  // Forward to ML microservice
  const mlResponse = await axios.post(
    `${ML_URL}/predict/${diseaseType}`,
    inputData
  );

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

// ── Image Prediction (Lung / Breast Cancer) ──────────────────
/**
 * @route   POST /api/predictions/image
 * @desc    Upload image → Cloudinary, send to ML, save result
 * @access  Private
 */
export const predictImage = asyncHandler(async (req, res) => {
  const { diseaseType } = req.body;

  if (!["lung", "breast"].includes(diseaseType)) {
    res.status(400);
    throw new Error("diseaseType must be 'lung' or 'breast'");
  }

  if (!req.file) {
    res.status(400);
    throw new Error("Image file is required");
  }

  // 1. Upload image to Cloudinary
  const cloudResult = await uploadToCloudinary(
    req.file.buffer,
    `wellnex/${diseaseType}`
  );
  const imageUrl = cloudResult.secure_url;

  // 2. Forward image buffer to ML microservice as multipart
  const form = new FormData();
  form.append("file", req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype,
  });

  const mlResponse = await axios.post(
    `${ML_URL}/predict/${diseaseType}`,
    form,
    { headers: form.getHeaders() }
  );

  const { prediction, probability } = mlResponse.data;

  // 3. Persist prediction
  const record = await Prediction.create({
    userId: req.user._id,
    diseaseType,
    imageUrl,
    predictionResult: prediction,
    probability,
  });

  res.status(201).json(record);
});

// ── Unified Prediction (Any Report Format) ──────────────────
/**
 * @route   POST /api/predictions/unified
 * @desc    Upload any report file, run all 4 models, optionally ask for missing fields
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

  const mlResponse = await axios.post(`${ML_URL}/predict/unified`, form, {
    headers: form.getHeaders(),
    maxBodyLength: Infinity,
  });

  const analysis = mlResponse.data;
  const likelyDisease = analysis?.summary?.likely_disease;
  const likelyResult = analysis?.model_results?.[likelyDisease];
  const canPersist =
    SUPPORTED_DISEASES.includes(likelyDisease) &&
    likelyResult?.status === "success";

  let savedPrediction = null;

  if (canPersist) {
    const confidence = Number(analysis?.summary?.confidence ?? 0);
    const predictionResult =
      confidence >= 0.5
        ? `Positive (Likely ${likelyDisease})`
        : `Inconclusive (Likely ${likelyDisease})`;

    let inputData = {};
    if (["diabetes", "heart"].includes(likelyDisease)) {
      inputData = analysis?.resolved_inputs?.[likelyDisease] || {};
    }

    let imageUrl = null;
    if (req.file.mimetype?.startsWith("image/")) {
      try {
        const cloudResult = await uploadToCloudinary(
          req.file.buffer,
          "wellnex/unified"
        );
        imageUrl = cloudResult.secure_url;
      } catch {
        // Continue even if image archival fails; prediction output is still valid.
      }
    }

    savedPrediction = await Prediction.create({
      userId: req.user._id,
      diseaseType: likelyDisease,
      inputData,
      imageUrl,
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
  });
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
