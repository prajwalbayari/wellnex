// ─────────────────────────────────────────────────────────────
// Auth Context – global auth state via React Context + hooks
// ─────────────────────────────────────────────────────────────
import { createContext, useContext, useState, useEffect } from "react";
import { setLogoutCallback } from "../services/api";

const AuthContext = createContext(null);

const STORAGE_KEY = "wellnex_user";

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : null;
  });

  // Persist user to localStorage whenever it changes
  useEffect(() => {
    if (user) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [user]);

  // Register logout callback so the 401 interceptor uses React state
  useEffect(() => {
    setLogoutCallback(() => {
      setUser(null);
      localStorage.removeItem(STORAGE_KEY);
    });
  }, []);

  const login = (userData) => setUser(userData);

  const logout = () => {
    setUser(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

/** Hook to access auth context */
export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};
