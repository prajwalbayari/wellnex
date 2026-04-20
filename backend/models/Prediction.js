// ─────────────────────────────────────────────────────────────
// Prediction Mongoose Model
// ─────────────────────────────────────────────────────────────
import mongoose from "mongoose";

const predictionSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    diseaseType: {
      type: String,
      enum: ["diabetes", "heart", "breast"],
      required: true,
    },
    inputData: {
      type: mongoose.Schema.Types.Mixed, // JSON object holding form fields
      default: {},
    },
    imageUrl: {
      type: String, // Cloudinary secure_url (for image-based predictions)
      default: null,
    },
    imagePublicId: {
      type: String, // Cloudinary public_id for optional cleanup/auditing
      default: null,
    },
    predictionResult: {
      type: String, // "Positive" | "Negative"
      required: true,
    },
    probability: {
      type: Number, // 0–1 float
      required: true,
    },
  },
  { timestamps: true }
);

predictionSchema.index({ userId: 1, createdAt: -1 });

const Prediction = mongoose.model("Prediction", predictionSchema);
export default Prediction;
