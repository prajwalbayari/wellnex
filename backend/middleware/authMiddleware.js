// ─────────────────────────────────────────────────────────────
// JWT Auth Middleware – protects private routes
// ─────────────────────────────────────────────────────────────
import jwt from "jsonwebtoken";
import User from "../models/User.js";

const protect = async (req, res, next) => {
  let token;

  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith("Bearer")
  ) {
    try {
      token = req.headers.authorization.split(" ")[1];
      const decoded = jwt.verify(token, process.env.JWT_SECRET);

      // Attach user (minus password) to request
      req.user = await User.findById(decoded.id).select("-password");
      if (!req.user) {
        return res.status(401).json({ message: "Not authorised – user no longer exists" });
      }
      return next();
    } catch (error) {
      console.error("Token verification failed:", error.message);
      return res.status(401).json({ message: "Not authorised – token failed" });
    }
  }

  if (!token) {
    return res.status(401).json({ message: "Not authorised – no token" });
  }
};

export default protect;
