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
  const statusCode = res.statusCode === 200 ? 500 : res.statusCode;
  res.status(statusCode).json({
    message: err.message,
    stack: process.env.NODE_ENV === "production" ? null : err.stack,
  });
};
