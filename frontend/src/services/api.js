// ─────────────────────────────────────────────────────────────
// Axios Instance with JWT interceptor
// ─────────────────────────────────────────────────────────────
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const api = axios.create({
  baseURL: `${API_URL}/api`,
  headers: { "Content-Type": "application/json" },
});

// ── Logout callback (set by AuthContext) ─────────────────────
let _logoutCallback = null;
export const setLogoutCallback = (cb) => {
  _logoutCallback = cb;
};

// ── Request Interceptor – attach JWT ─────────────────────────
api.interceptors.request.use(
  (config) => {
    const stored = localStorage.getItem("wellnex_user");
    if (stored) {
      const { token } = JSON.parse(stored);
      if (token) config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response Interceptor – handle 401 globally ───────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (_logoutCallback) {
        _logoutCallback();
      } else {
        localStorage.removeItem("wellnex_user");
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

export default api;
