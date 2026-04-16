// ─────────────────────────────────────────────────────────────
// Global Error-Handling Middleware
// ─────────────────────────────────────────────────────────────

/** Catch 404 – no route matched */
export const notFound = (req, res, next) => {
  const error = new Error(`Not Found – ${req.originalUrl}`);
  res.status(404);
  next(error);
};

/** Centralised error handler */
export const errorHandler = (err, _req, res, _next) => {
  if (err?.name === "MulterError") {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ message: "Uploaded file exceeds size limit" });
    }
    return res.status(400).json({ message: err.message || "File upload failed" });
  }

  const statusCode =
    err.statusCode || err.status || (res.statusCode === 200 ? 500 : res.statusCode);
  res.status(statusCode).json({
    message: err.message,
    stack: process.env.NODE_ENV === "production" ? null : err.stack,
  });
};
