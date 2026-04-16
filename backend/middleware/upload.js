// ─────────────────────────────────────────────────────────────
// Multer Configuration – memory storage for Cloudinary
// ─────────────────────────────────────────────────────────────
import multer from "multer";

const storage = multer.memoryStorage();

const badRequestError = (message) => {
  const error = new Error(message);
  error.statusCode = 400;
  return error;
};

const imageFileFilter = (_req, file, cb) => {
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (allowed.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(badRequestError("Only JPEG, PNG and WebP images are allowed"), false);
  }
};

const reportFileFilter = (_req, file, cb) => {
  const allowedMimeTypes = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/json",
    "application/xml",
    "text/xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ]);

  const lower = String(file.originalname || "").toLowerCase();
  const allowedExtensions = [
    ".pdf",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".docx",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
  ];

  const hasAllowedExtension = allowedExtensions.some((ext) => lower.endsWith(ext));
  if (allowedMimeTypes.has(file.mimetype) || hasAllowedExtension) {
    cb(null, true);
    return;
  }

  cb(
    badRequestError(
      "Unsupported file type. Allowed: PDF, TXT, CSV, JSON, XML, DOCX, JPG, PNG, WEBP"
    ),
    false
  );
};

export const uploadImage = multer({
  storage,
  fileFilter: imageFileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB max
});

export const uploadAny = multer({
  storage,
  fileFilter: reportFileFilter,
  limits: { fileSize: 20 * 1024 * 1024 }, // 20 MB max for reports
});

export default uploadImage;
