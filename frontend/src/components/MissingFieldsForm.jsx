import { useEffect, useMemo, useState } from "react";
import Spinner from "./Spinner";

const FIELD_DEFS = {
  diabetes: {
    age: { label: "Age", type: "number", placeholder: "e.g. 45" },
    gender: { label: "Gender", type: "select", options: ["Female", "Male", "Other"] },
    bmi: { label: "BMI", type: "number", placeholder: "e.g. 27.5" },
    systolic_bp: { label: "Systolic BP", type: "number", placeholder: "e.g. 130" },
    diastolic_bp: { label: "Diastolic BP", type: "number", placeholder: "e.g. 85" },
    glucose_fasting: { label: "Fasting Glucose", type: "number", placeholder: "mg/dL" },
    glucose_postprandial: { label: "Postprandial Glucose", type: "number", placeholder: "mg/dL" },
    insulin_level: { label: "Insulin Level", type: "number", placeholder: "µU/mL" },
    hba1c: { label: "HbA1c", type: "number", placeholder: "e.g. 6.4" },
    cholesterol_total: { label: "Total Cholesterol", type: "number", placeholder: "mg/dL" },
    hdl_cholesterol: { label: "HDL Cholesterol", type: "number", placeholder: "mg/dL" },
    ldl_cholesterol: { label: "LDL Cholesterol", type: "number", placeholder: "mg/dL" },
    triglycerides: { label: "Triglycerides", type: "number", placeholder: "mg/dL" },
  },
  heart: {
    male: { label: "Male", type: "binary" },
    age: { label: "Age", type: "number", placeholder: "e.g. 55" },
    currentSmoker: { label: "Current Smoker", type: "binary" },
    totChol: { label: "Total Cholesterol", type: "number", placeholder: "mg/dL" },
    sysBP: { label: "Systolic BP", type: "number", placeholder: "mmHg" },
    diaBP: { label: "Diastolic BP", type: "number", placeholder: "mmHg" },
    BMI: { label: "BMI", type: "number", placeholder: "e.g. 26" },
    heartRate: { label: "Heart Rate", type: "number", placeholder: "bpm" },
    glucose: { label: "Glucose", type: "number", placeholder: "mg/dL" },
  },
};

const fieldLabel = (disease, field) => {
  const def = FIELD_DEFS[disease]?.[field];
  if (def?.label) return def.label;
  return field.replace(/_/g, " ");
};

const buildInitialValues = (missingFields, prefillInputs) => {
  const initial = {};

  for (const disease of ["diabetes", "heart"]) {
    const list = missingFields?.[disease] || [];
    initial[disease] = {};
    for (const field of list) {
      const prefillValue = prefillInputs?.[disease]?.[field];
      initial[disease][field] =
        prefillValue === null || prefillValue === undefined ? "" : String(prefillValue);
    }
  }

  return initial;
};

const MissingFieldsForm = ({ missingFields, prefillInputs, loading, onSubmit }) => {
  const [form, setForm] = useState(() => buildInitialValues(missingFields, prefillInputs));

  useEffect(() => {
    setForm(buildInitialValues(missingFields, prefillInputs));
  }, [missingFields, prefillInputs]);

  const hasMissing = useMemo(
    () =>
      (missingFields?.diabetes?.length || 0) > 0 ||
      (missingFields?.heart?.length || 0) > 0,
    [missingFields]
  );

  if (!hasMissing) return null;

  const setField = (disease, field, value) => {
    setForm((prev) => ({
      ...prev,
      [disease]: {
        ...(prev[disease] || {}),
        [field]: value,
      },
    }));
  };

  const inputClass =
    "w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-400 focus:border-transparent outline-none transition";

  const renderField = (disease, field) => {
    const def = FIELD_DEFS[disease]?.[field] || { type: "number" };
    const value = form?.[disease]?.[field] ?? "";

    if (def.type === "select") {
      return (
        <select
          value={value}
          onChange={(e) => setField(disease, field, e.target.value)}
          className={inputClass}
          required
        >
          <option value="">Select...</option>
          {(def.options || []).map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      );
    }

    if (def.type === "binary") {
      return (
        <select
          value={value}
          onChange={(e) => setField(disease, field, e.target.value)}
          className={inputClass}
          required
        >
          <option value="">Select...</option>
          <option value="1">Yes</option>
          <option value="0">No</option>
        </select>
      );
    }

    return (
      <input
        type="number"
        value={value}
        onChange={(e) => setField(disease, field, e.target.value)}
        placeholder={def.placeholder || "Enter value"}
        step="any"
        className={inputClass}
        required
      />
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const payload = { diabetes: {}, heart: {} };

    for (const disease of ["diabetes", "heart"]) {
      const fields = missingFields?.[disease] || [];
      for (const field of fields) {
        const raw = form?.[disease]?.[field];
        const def = FIELD_DEFS[disease]?.[field] || { type: "number" };

        if (def.type === "number") {
          payload[disease][field] = parseFloat(raw);
        } else if (def.type === "binary") {
          payload[disease][field] = raw === "1" ? 1 : 0;
        } else {
          payload[disease][field] = raw;
        }
      }
    }

    onSubmit(payload);
  };

  return (
    <form onSubmit={handleSubmit} className="mt-6 p-4 rounded-lg border border-amber-200 bg-amber-50 space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-amber-900">Additional Clinical Fields Needed</h3>
        <p className="text-xs text-amber-800 mt-1">
          Some required values could not be extracted from the uploaded report. Fill these fields once and the same file will be re-evaluated across all 4 models.
        </p>
      </div>

      {["diabetes", "heart"].map((disease) => {
        const fields = missingFields?.[disease] || [];
        if (!fields.length) return null;

        return (
          <fieldset key={disease}>
            <legend className="text-xs font-semibold uppercase tracking-wide text-amber-900 mb-2">
              {disease} fields
            </legend>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {fields.map((field) => (
                <div key={`${disease}-${field}`}>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    {fieldLabel(disease, field)}
                  </label>
                  {renderField(disease, field)}
                </div>
              ))}
            </div>
          </fieldset>
        );
      })}

      <button
        type="submit"
        disabled={loading}
        className="w-full sm:w-auto bg-primary-600 text-white font-medium px-8 py-2.5 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition"
      >
        {loading ? <Spinner size="sm" /> : "Re-run Unified Analysis"}
      </button>
    </form>
  );
};

export default MissingFieldsForm;
