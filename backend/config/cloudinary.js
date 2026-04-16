// ─────────────────────────────────────────────────────────────
// Cloudinary Configuration
// ─────────────────────────────────────────────────────────────
import { v2 as cloudinary } from "cloudinary";

// Lazy initialization – called from server.js after dotenv.config()
let isConfigured = false;

const ensureConfigured = () => {
  if (isConfigured) return;

  const cloudName = process.env.CLOUDINARY_CLOUD_NAME;
  const apiKey = process.env.CLOUDINARY_API_KEY;
  const apiSecret = process.env.CLOUDINARY_API_SECRET;

  if (!cloudName || !apiKey || !apiSecret) {
    throw new Error(
      "Missing Cloudinary configuration. Required: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
    );
  }
  
  cloudinary.config({
    cloud_name: cloudName,
    api_key: apiKey,
    api_secret: apiSecret,
  });
  
  isConfigured = true;
};

/**
 * Upload a buffer (from multer memoryStorage) to Cloudinary.
 * @param {Buffer} buffer  – file buffer
 * @param {string} folder  – Cloudinary folder name
 * @returns {Promise<object>} Cloudinary upload result
 */
export const uploadToCloudinary = (buffer, folder = "wellnex") => {
  ensureConfigured(); // Ensure config is loaded before upload
  
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      { folder, resource_type: "image" },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    stream.end(buffer);
  });
};

export { ensureConfigured };
export default cloudinary;
