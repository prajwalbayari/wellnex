// ─────────────────────────────────────────────────────────────
// Prediction Routes
// ─────────────────────────────────────────────────────────────
import { Router } from "express";
import {
  predictTabular,
  predictImage,
  predictUnified,
  getHistory,
  deletePrediction,
} from "../controllers/predictionController.js";
import protect from "../middleware/authMiddleware.js";
import upload, { uploadAny } from "../middleware/upload.js";

const router = Router();

// All prediction routes require authentication
router.use(protect);

router.post("/tabular", predictTabular);
router.post("/image", upload.single("image"), predictImage);
router.post("/unified", uploadAny.single("file"), predictUnified);
router.get("/history", getHistory);
router.delete("/:id", deletePrediction);

export default router;
