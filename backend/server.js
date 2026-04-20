// ─────────────────────────────────────────────────────────────
// Wellnex Backend – Entry Point
// ─────────────────────────────────────────────────────────────
import dotenv from "dotenv";
dotenv.config();

if (!process.env.JWT_SECRET || !process.env.JWT_SECRET.trim()) {
  throw new Error("Missing required JWT_SECRET environment variable");
}

import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";

import connectDB from "./config/db.js";
import { ensureConfigured } from "./config/cloudinary.js";
import authRoutes from "./routes/authRoutes.js";
import predictionRoutes from "./routes/predictionRoutes.js";
import { notFound, errorHandler } from "./middleware/errorMiddleware.js";

// Initialize Cloudinary if configured. Image routes will require it at request time.
try {
  ensureConfigured();
} catch (error) {
  console.warn(`Cloudinary not configured at startup: ${error.message}`);
}

// Connect to MongoDB
await connectDB();

const app = express();

// ── Global Middleware ────────────────────────────────────────
const allowedOrigins = process.env.CORS_ORIGIN
  ? process.env.CORS_ORIGIN.split(",").map((origin) => origin.trim()).filter(Boolean)
  : ["http://localhost:5173", "http://localhost:3000"];

app.use(
  cors({
    origin: allowedOrigins,
    credentials: true,
  })
);
app.use(helmet());
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: Number(process.env.RATE_LIMIT_MAX || 300),
    standardHeaders: true,
    legacyHeaders: false,
  })
);
app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true, limit: "2mb" }));

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
  console.log(`Wellnex backend running on port ${PORT}`);
});
