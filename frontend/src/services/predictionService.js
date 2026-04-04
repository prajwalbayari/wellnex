// ─────────────────────────────────────────────────────────────
// Prediction Service – tabular + image predictions, history
// ─────────────────────────────────────────────────────────────
import api from "./api";

/**
 * Submit a tabular prediction (diabetes / heart).
 * @param {string} diseaseType  – "diabetes" | "heart"
 * @param {object} inputData    – form fields
 */
export const predictTabular = async (diseaseType, inputData) => {
  const { data } = await api.post("/predictions/tabular", {
    diseaseType,
    inputData,
  });
  return data;
};

/**
 * Submit an image prediction (breast only).
 * @param {string} diseaseType – "breast"
 * @param {File}   imageFile   – user-selected image file
 */
export const predictImage = async (diseaseType, imageFile) => {
  const formData = new FormData();
  formData.append("diseaseType", diseaseType);
  formData.append("image", imageFile);

  const { data } = await api.post("/predictions/image", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

/**
 * Unified prediction from a single uploaded report (diabetes + heart + breast).
 * @param {File} file
 * @param {object} supplementalData - optional missing field values
 */
export const predictUnified = async (file, supplementalData = {}) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("supplementalData", JSON.stringify(supplementalData));

  const { data } = await api.post("/predictions/unified", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

/** Fetch logged-in user's prediction history. */
export const getHistory = async () => {
  const { data } = await api.get("/predictions/history");
  return data;
};

/** Delete a single prediction by ID. */
export const deletePrediction = async (id) => {
  const { data } = await api.delete(`/predictions/${id}`);
  return data;
};
