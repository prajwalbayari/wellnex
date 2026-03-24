// ─────────────────────────────────────────────────────────────
// PredictionResult – displays prediction outcome card
// ─────────────────────────────────────────────────────────────
import { FiCheckCircle, FiAlertTriangle } from "react-icons/fi";

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const predictionText =
    result.predictionResult || result.summary?.overall_prediction || "No prediction";
  const confidence = Number(result.probability ?? result.summary?.confidence ?? 0);
  const diseaseType = result.diseaseType || result.summary?.likely_disease || "unknown";
  const isPositive =
    predictionText.toLowerCase().includes("positive") ||
    predictionText.toLowerCase().includes("most likely") ||
    confidence >= 0.5;

  const modelRows = result.modelResults
    ? Object.entries(result.modelResults)
    : [];

  return (
    <div
      className={`mt-6 p-6 rounded-xl border-2 ${
        isPositive
          ? "bg-red-50 border-red-300"
          : "bg-green-50 border-green-300"
      }`}
    >
      <div className="flex items-center gap-3 mb-3">
        {isPositive ? (
          <FiAlertTriangle className="w-8 h-8 text-red-500" />
        ) : (
          <FiCheckCircle className="w-8 h-8 text-green-500" />
        )}
        <h3 className="text-xl font-bold">
          {isPositive ? "Risk Detected" : "No Risk Detected"}
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Prediction</span>
          <p
            className={`font-semibold text-lg ${
              isPositive ? "text-red-600" : "text-green-600"
            }`}
          >
            {predictionText}
          </p>
        </div>
        <div>
          <span className="text-gray-500">Confidence</span>
          <p className="font-semibold text-lg">
            {(confidence * 100).toFixed(2)}%
          </p>
        </div>
        <div>
          <span className="text-gray-500">Disease Type</span>
          <p className="font-semibold capitalize">{diseaseType}</p>
        </div>
        <div>
          <span className="text-gray-500">Date</span>
          <p className="font-semibold">
            {new Date(result.createdAt).toLocaleDateString()}
          </p>
        </div>
      </div>

      {result.needsUserInput && (
        <p className="mt-4 text-xs text-amber-700 bg-amber-100 border border-amber-200 px-3 py-2 rounded-md">
          Diabetes and/or heart analysis is provisional. Fill missing fields to complete all tabular checks.
        </p>
      )}

      {modelRows.length > 0 && (
        <div className="mt-4 rounded-lg bg-white/60 border border-gray-200 p-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
            Model-wise Evidence
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {modelRows.map(([disease, model]) => (
              <div key={disease} className="rounded border border-gray-200 px-2 py-1 bg-white">
                <p className="font-semibold capitalize text-gray-700">{disease}</p>
                <p className="text-gray-600">
                  {model.status === "success"
                    ? `${model.prediction} (${(Number(model.positive_probability ?? model.probability ?? 0) * 100).toFixed(1)}%)`
                    : model.status === "needs_input"
                    ? "Needs additional fields"
                    : "Could not evaluate"}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="mt-4 text-xs text-gray-400">
        ⚠️ This is an AI prediction and not a medical diagnosis. Please consult
        a healthcare professional.
      </p>
    </div>
  );
};

export default PredictionResult;
