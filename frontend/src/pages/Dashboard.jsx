// ─────────────────────────────────────────────────────────────
// Dashboard Page – disease selection, forms, results, history
// ─────────────────────────────────────────────────────────────
import { useState, useEffect } from "react";
import toast from "react-hot-toast";

import { useAuth } from "../context/AuthContext";
import UniversalReportUpload from "../components/UniversalReportUpload";
import MissingFieldsForm from "../components/MissingFieldsForm";
import DiabetesForm from "../components/DiabetesForm";
import HeartForm from "../components/HeartForm";
import PredictionResult from "../components/PredictionResult";
import HistoryTable from "../components/HistoryTable";
import Spinner from "../components/Spinner";

import {
  predictTabular,
  predictUnified,
  getHistory,
  deletePrediction,
} from "../services/predictionService";

const Dashboard = () => {
  const { user } = useAuth();

  const [inputMode, setInputMode] = useState("upload");
  const [pendingFile, setPendingFile] = useState(null);
  const [missingFields, setMissingFields] = useState(null);
  const [prefillInputs, setPrefillInputs] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  // Fetch history on mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await getHistory();
        setHistory(data);
      } catch {
        toast.error("Failed to load history");
      } finally {
        setHistoryLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const runUnifiedAnalysis = async (file, supplementalData = {}) => {
    setLoading(true);
    setResult(null);

    try {
      const data = await predictUnified(file, supplementalData);

      setMissingFields(data.missing_fields || null);
      setPrefillInputs(data.prefill_inputs || null);

      const displayResult = data.savedPrediction
        ? {
            ...data.savedPrediction,
            modelResults: data.model_results,
            summary: data.summary,
            needsUserInput: data.needs_user_input,
          }
        : {
            diseaseType: data.summary?.likely_disease || "unknown",
            predictionResult: data.summary?.overall_prediction || "Unable to determine",
            probability: data.summary?.confidence || 0,
            createdAt: new Date().toISOString(),
            modelResults: data.model_results,
            summary: data.summary,
            needsUserInput: data.needs_user_input,
          };

      setResult(displayResult);

      if (data.savedPrediction) {
        setHistory((prev) => [data.savedPrediction, ...prev]);
        toast.success("Unified prediction complete!");
      } else if (data.needs_user_input) {
        toast("Please fill the missing clinical fields to finalize tabular analysis.", {
          icon: "⚠️",
        });
      } else {
        toast.success("Unified analysis complete!");
      }
    } catch (err) {
      toast.error(err.response?.data?.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleManualSubmit = async (diseaseType, inputData) => {
    setLoading(true);
    setResult(null);

    try {
      const data = await predictTabular(diseaseType, inputData);
      setResult(data);
      setHistory((prev) => [data, ...prev]);
      toast.success(`${diseaseType === "diabetes" ? "Diabetes" : "Heart"} prediction complete!`);
    } catch (err) {
      toast.error(err.response?.data?.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  // ── Initial file submission ────────────────────────────────
  const handleInitialUpload = async (file) => {
    setPendingFile(file);
    await runUnifiedAnalysis(file);
  };

  // ── Missing-fields completion submission ───────────────────
  const handleMissingFieldsSubmit = async (supplementalData) => {
    if (!pendingFile) {
      toast.error("Please upload the report again before submitting missing fields.");
      return;
    }
    await runUnifiedAnalysis(pendingFile, supplementalData);
  };

  // ── Delete handler ──────────────────────────────────────────
  const handleDelete = async (id) => {
    try {
      await deletePrediction(id);
      setHistory((prev) => prev.filter((p) => p._id !== id));
      toast.success("Prediction deleted");
    } catch {
      toast.error("Failed to delete prediction");
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold">
          Welcome, <span className="text-primary-600">{user?.name}</span>
        </h1>
        <p className="text-gray-500 text-sm mt-1">
          Upload one medical report and Wellnex will run a unified pipeline across
          diabetes, heart disease, lung cancer, and breast cancer models.
        </p>
      </div>

      {/* Dynamic Form */}
      <div className="mt-8 bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold mb-4">Unified Disease Detection</h2>

        <div className="inline-flex rounded-lg border border-gray-200 p-1 mb-5 bg-gray-50">
          <button
            type="button"
            onClick={() => setInputMode("upload")}
            className={`px-4 py-2 text-sm rounded-md transition ${
              inputMode === "upload"
                ? "bg-primary-600 text-white"
                : "text-gray-600 hover:bg-white"
            }`}
          >
            Upload Report
          </button>
          <button
            type="button"
            onClick={() => setInputMode("manual")}
            className={`px-4 py-2 text-sm rounded-md transition ${
              inputMode === "manual"
                ? "bg-primary-600 text-white"
                : "text-gray-600 hover:bg-white"
            }`}
          >
            Fill Details Manually
          </button>
        </div>

        {inputMode === "upload" ? (
          <>
            <UniversalReportUpload loading={loading} onSubmit={handleInitialUpload} />

            <MissingFieldsForm
              missingFields={missingFields}
              prefillInputs={prefillInputs}
              loading={loading}
              onSubmit={handleMissingFieldsSubmit}
            />
          </>
        ) : (
          <div className="space-y-8">
            <div className="rounded-lg border border-gray-200 p-4">
              <h3 className="text-base font-semibold mb-3">Diabetes Details</h3>
              <DiabetesForm
                loading={loading}
                onSubmit={(data) => handleManualSubmit("diabetes", data)}
              />
            </div>

            <div className="rounded-lg border border-gray-200 p-4">
              <h3 className="text-base font-semibold mb-3">Heart Disease Details</h3>
              <HeartForm
                loading={loading}
                onSubmit={(data) => handleManualSubmit("heart", data)}
              />
            </div>
          </div>
        )}

        {/* Prediction Result Card */}
        {result && <PredictionResult result={result} />}
      </div>

      {/* History Section */}
      <div className="mt-10 bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold mb-4">Prediction History</h2>
        {historyLoading ? (
          <Spinner />
        ) : (
          <HistoryTable predictions={history} onDelete={handleDelete} />
        )}
      </div>
    </div>
  );
};

export default Dashboard;
