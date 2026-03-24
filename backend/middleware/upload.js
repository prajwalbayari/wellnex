// ─────────────────────────────────────────────────────────────
// Multer Configuration – memory storage for Cloudinary
// ─────────────────────────────────────────────────────────────
import multer from "multer";

const storage = multer.memoryStorage();

const imageFileFilter = (_req, file, cb) => {
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (allowed.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error("Only JPEG, PNG and WebP images are allowed"), false);
  }
};

export const uploadImage = multer({
  storage,
  fileFilter: imageFileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB max
});

export const uploadAny = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 }, // 20 MB max for reports
});

export default uploadImage;
