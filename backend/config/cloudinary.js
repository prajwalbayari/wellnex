// ─────────────────────────────────────────────────────────────
// Cloudinary Configuration
// ─────────────────────────────────────────────────────────────
import { v2 as cloudinary } from "cloudinary";

// Lazy initialization – called from server.js after dotenv.config()
let isConfigured = false;

const ensureConfigured = () => {
  if (isConfigured) return;
  
  cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
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
