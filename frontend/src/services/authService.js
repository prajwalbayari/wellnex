// ─────────────────────────────────────────────────────────────
// Auth Service – login, signup, profile
// ─────────────────────────────────────────────────────────────
import api from "./api";

export const loginUser = async (email, password) => {
  const { data } = await api.post("/auth/login", { email, password });
  return data;
};

export const signupUser = async (name, email, password) => {
  const { data } = await api.post("/auth/signup", { name, email, password });
  return data;
};

export const getProfile = async () => {
  const { data } = await api.get("/auth/profile");
  return data;
};
