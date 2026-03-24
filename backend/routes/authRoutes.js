// ─────────────────────────────────────────────────────────────
// Auth Routes
// ─────────────────────────────────────────────────────────────
import { Router } from "express";
import { signup, login, getProfile } from "../controllers/authController.js";
import protect from "../middleware/authMiddleware.js";

const router = Router();

router.post("/signup", signup);
router.post("/login", login);
router.get("/profile", protect, getProfile);

export default router;
