// ─────────────────────────────────────────────────────────────
// Wellnex Backend – Entry Point
// ─────────────────────────────────────────────────────────────
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";

import connectDB from "./config/db.js";
import { ensureConfigured } from "./config/cloudinary.js";
import authRoutes from "./routes/authRoutes.js";
import predictionRoutes from "./routes/predictionRoutes.js";
import { notFound, errorHandler } from "./middleware/errorMiddleware.js";

// Initialize Cloudinary after env vars are loaded
ensureConfigured();

// Connect to MongoDB
await connectDB();

const app = express();

// ── Global Middleware ────────────────────────────────────────
app.use(cors({ origin: process.env.CORS_ORIGIN || "*" }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ── Health Check ─────────────────────────────────────────────
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", service: "wellnex-backend" });
});

// ── Routes ───────────────────────────────────────────────────
app.use("/api/auth", authRoutes);
app.use("/api/predictions", predictionRoutes);

// ── Error Handling ───────────────────────────────────────────
app.use(notFound);
app.use(errorHandler);

// ── Start Server ─────────────────────────────────────────────
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀  Wellnex backend running on port ${PORT}`);
});
