// ─────────────────────────────────────────────────────────────
// Spinner Component – reusable loading indicator
// ─────────────────────────────────────────────────────────────
const Spinner = ({ size = "md" }) => {
  const sizes = {
    sm: "w-5 h-5",
    md: "w-8 h-8",
    lg: "w-12 h-12",
  };

  return (
    <div className={`flex justify-center items-center ${size === "sm" ? "py-0" : "py-4"}`}>
      <div
        className={`${sizes[size]} border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin`}
      />
    </div>
  );
};

export default Spinner;
