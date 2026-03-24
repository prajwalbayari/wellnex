// ─────────────────────────────────────────────────────────────
// JWT Generation Utility
// ─────────────────────────────────────────────────────────────
import jwt from "jsonwebtoken";

/**
 * Generate a signed JWT for the given user id.
 * Token expires in 30 days by default.
 */
const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET, { expiresIn: "30d" });
};

export default generateToken;
