// ─────────────────────────────────────────────────────────────
// HistoryTable – prediction history with expandable detail rows
// ─────────────────────────────────────────────────────────────
import { useState, Fragment } from "react";
import { FiChevronDown, FiChevronUp, FiTrash2, FiImage } from "react-icons/fi";

/* ── Friendly labels for inputData keys ────────────────────── */
const FIELD_LABELS = {
  // Diabetes
  age: "Age",
  gender: "Gender",
  ethnicity: "Ethnicity",
  education_level: "Education Level",
  income_level: "Income Level",
  employment_status: "Employment Status",
  smoking_status: "Smoking Status",
  alcohol_consumption_per_week: "Alcohol (drinks/week)",
  physical_activity_minutes_per_week: "Physical Activity (min/week)",
  diet_score: "Diet Score",
  sleep_hours_per_day: "Sleep (hrs/day)",
  screen_time_hours_per_day: "Screen Time (hrs/day)",
  family_history_diabetes: "Family History",
  hypertension_history: "Hypertension History",
  cardiovascular_history: "Cardiovascular History",
  bmi: "BMI",
  waist_to_hip_ratio: "Waist-to-Hip Ratio",
  systolic_bp: "Systolic BP",
  diastolic_bp: "Diastolic BP",
  heart_rate: "Heart Rate",
  cholesterol_total: "Total Cholesterol",
  hdl_cholesterol: "HDL Cholesterol",
  ldl_cholesterol: "LDL Cholesterol",
  triglycerides: "Triglycerides",
  glucose_fasting: "Fasting Glucose",
  glucose_postprandial: "Postprandial Glucose",
  insulin_level: "Insulin Level",
  hba1c: "HbA1c (%)",
  diabetes_risk_score: "Diabetes Risk Score",
  // Heart
  male: "Male",
  education: "Education Level",
  currentSmoker: "Current Smoker",
  cigsPerDay: "Cigarettes/Day",
  BPMeds: "On BP Meds",
  prevalentStroke: "Prevalent Stroke",
  prevalentHyp: "Prevalent Hypertension",
  diabetes: "Has Diabetes",
  totChol: "Total Cholesterol",
  sysBP: "Systolic BP",
  diaBP: "Diastolic BP",
  BMI: "BMI",
  heartRate: "Heart Rate",
  glucose: "Glucose Level",
};

const formatValue = (key, value) => {
  // Binary toggle fields → Yes / No
  const binaryKeys = [
    "family_history_diabetes", "hypertension_history", "cardiovascular_history",
    "male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
  ];
  if (binaryKeys.includes(key)) return value === 1 || value === "1" ? "Yes" : "No";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(2);
  return String(value);
};

/* ── Detail panel (tabular) ───────────────────────────────── */
const TabularDetails = ({ inputData }) => {
  if (!inputData || Object.keys(inputData).length === 0) {
    return <p className="text-xs text-gray-400 italic">No input data stored.</p>;
  }

  const entries = Object.entries(inputData);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-2">
      {entries.map(([key, value]) => (
        <div key={key} className="flex flex-col">
          <span className="text-[11px] text-gray-400 uppercase tracking-wide">
            {FIELD_LABELS[key] || key.replace(/_/g, " ")}
          </span>
          <span className="text-sm font-medium text-gray-700">
            {formatValue(key, value)}
          </span>
        </div>
      ))}
    </div>
  );
};

/* ── Detail panel (image) ─────────────────────────────────── */
const ImageDetails = ({ imageUrl }) => {
  const [fullscreen, setFullscreen] = useState(false);

  if (!imageUrl) {
    return <p className="text-xs text-gray-400 italic">No image stored.</p>;
  }

  return (
    <>
      <div className="flex items-start gap-4">
        <img
          src={imageUrl}
          alt="Uploaded scan"
          className="w-40 h-40 object-cover rounded-lg border border-gray-200 cursor-pointer hover:opacity-80 transition"
          onClick={() => setFullscreen(true)}
        />
        <div className="text-xs text-gray-500 space-y-1 pt-1">
          <p className="flex items-center gap-1">
            <FiImage className="w-3.5 h-3.5" /> Click image to view full size
          </p>
          <a
            href={imageUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-600 hover:underline"
          >
            Open in new tab →
          </a>
        </div>
      </div>

      {/* Fullscreen overlay */}
      {fullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setFullscreen(false)}
        >
          <img
            src={imageUrl}
            alt="Full scan"
            className="max-h-[90vh] max-w-[90vw] rounded-lg shadow-2xl"
          />
        </div>
      )}
    </>
  );
};

/* ── Main Component ───────────────────────────────────────── */
const HistoryTable = ({ predictions, onDelete }) => {
  const [expandedId, setExpandedId] = useState(null);

  if (!predictions || predictions.length === 0) {
    return (
      <p className="text-sm text-gray-400 text-center py-8">
        No predictions yet. Upload a report above to get started.
      </p>
    );
  }

  const toggle = (id) => setExpandedId((prev) => (prev === id ? null : id));

  const isTabular = (type) => ["diabetes", "heart"].includes(type);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-gray-500 uppercase border-b">
            <th className="pb-2 pr-2 w-8"></th>
            <th className="pb-2 pr-4">Date</th>
            <th className="pb-2 pr-4">Type</th>
            <th className="pb-2 pr-4">Result</th>
            <th className="pb-2 pr-4">Confidence</th>
            <th className="pb-2 pr-4">Data</th>
            {onDelete && <th className="pb-2 w-10"></th>}
          </tr>
        </thead>
        <tbody>
          {predictions.map((p) => {
            const expanded = expandedId === p._id;
            const hasDetails = isTabular(p.diseaseType)
              ? p.inputData && Object.keys(p.inputData).length > 0
              : !!p.imageUrl;

            return (
              <Fragment key={p._id}>
                <tr className="border-b last:border-0 group">
                  {/* Expand toggle */}
                  <td className="py-3 pr-2">
                    {hasDetails && (
                      <button
                        onClick={() => toggle(p._id)}
                        className="text-gray-400 hover:text-primary-600 transition"
                        title={expanded ? "Collapse" : "View details"}
                      >
                        {expanded ? (
                          <FiChevronUp className="w-4 h-4" />
                        ) : (
                          <FiChevronDown className="w-4 h-4" />
                        )}
                      </button>
                    )}
                  </td>

                  {/* Date */}
                  <td className="py-3 pr-4 text-gray-600 whitespace-nowrap">
                    {new Date(p.createdAt).toLocaleDateString()}{" "}
                    <span className="text-gray-400 text-xs">
                      {new Date(p.createdAt).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </td>

                  {/* Type */}
                  <td className="py-3 pr-4 capitalize font-medium">{p.diseaseType}</td>

                  {/* Result badge */}
                  <td className="py-3 pr-4">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        p.predictionResult?.startsWith("Positive")
                          ? "bg-red-100 text-red-700"
                          : p.predictionResult?.startsWith("Inconclusive")
                          ? "bg-amber-100 text-amber-700"
                          : "bg-green-100 text-green-700"
                      }`}
                    >
                      {p.predictionResult}
                    </span>
                  </td>

                  {/* Confidence */}
                  <td className="py-3 pr-4">{(p.probability * 100).toFixed(1)}%</td>

                  {/* Data indicator */}
                  <td className="py-3 pr-4">
                    {isTabular(p.diseaseType) ? (
                      <span className="text-xs text-gray-400">
                        {p.inputData ? `${Object.keys(p.inputData).length} fields` : "—"}
                      </span>
                    ) : (
                      p.imageUrl && (
                        <button
                          type="button"
                          onClick={() => toggle(p._id)}
                          className="rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
                          aria-label="View prediction image details"
                        >
                          <img
                            src={p.imageUrl}
                            alt="Prediction thumbnail"
                            className="w-8 h-8 rounded object-cover border border-gray-200 cursor-pointer hover:ring-2 hover:ring-primary-400 transition"
                          />
                        </button>
                      )
                    )}
                  </td>

                  {/* Delete */}
                  {onDelete && (
                    <td className="py-3">
                      <button
                        onClick={() => onDelete(p._id)}
                        className="text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition"
                        title="Delete prediction"
                      >
                        <FiTrash2 className="w-4 h-4" />
                      </button>
                    </td>
                  )}
                </tr>

                {/* Inline expanded detail row */}
                {expanded && (
                  <tr>
                    <td colSpan={onDelete ? 7 : 6} className="p-4 bg-gray-50 border-b">
                      {isTabular(p.diseaseType) ? (
                        <TabularDetails inputData={p.inputData} />
                      ) : (
                        <ImageDetails imageUrl={p.imageUrl} />
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default HistoryTable;
