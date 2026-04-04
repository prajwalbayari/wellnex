// ─────────────────────────────────────────────────────────────
// HeartForm – dynamic form for heart disease prediction
// ─────────────────────────────────────────────────────────────
import { useState } from "react";
import Spinner from "./Spinner";

const numberFields = [
  { name: "age", label: "Age", placeholder: "e.g. 55" },
  { name: "cigsPerDay", label: "Cigarettes / Day", placeholder: "e.g. 10" },
  { name: "totChol", label: "Total Cholesterol", placeholder: "mg/dL" },
  { name: "sysBP", label: "Systolic BP", placeholder: "mmHg" },
  { name: "diaBP", label: "Diastolic BP", placeholder: "mmHg" },
  { name: "BMI", label: "BMI", placeholder: "e.g. 26.5" },
  { name: "heartRate", label: "Heart Rate", placeholder: "bpm" },
  { name: "glucose", label: "Glucose Level", placeholder: "mg/dL" },
];

const toggleFields = [
  { name: "male", label: "Male" },
  { name: "currentSmoker", label: "Current Smoker" },
  { name: "BPMeds", label: "On BP Meds" },
  { name: "prevalentStroke", label: "Prevalent Stroke" },
  { name: "prevalentHyp", label: "Prevalent Hypertension" },
  { name: "diabetes", label: "Has Diabetes" },
];

const defaultState = {
  // number fields start empty
  ...Object.fromEntries(numberFields.map((f) => [f.name, ""])),
  // toggles
  ...Object.fromEntries(toggleFields.map((f) => [f.name, 0])),
};

const HeartForm = ({ onSubmit, loading }) => {
  const [form, setForm] = useState({ ...defaultState });

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const handleToggle = (name) =>
    setForm((prev) => ({ ...prev, [name]: prev[name] === 1 ? 0 : 1 }));

  const handleSubmit = (e) => {
    e.preventDefault();
    const payload = {};
    for (const nf of numberFields)
      payload[nf.name] = parseFloat(form[nf.name]) || 0;
    payload.education = 2; // default model value; education is not manually collected
    for (const tf of toggleFields) payload[tf.name] = form[tf.name];
    onSubmit(payload);
  };

  const inputClass =
    "w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-400 focus:border-transparent outline-none transition";

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* ── Toggle switches ── */}
      <fieldset>
        <legend className="text-sm font-semibold text-gray-700 mb-2">
          Patient Profile
        </legend>
        <div className="flex flex-wrap gap-6">
          {toggleFields.map((tf) => (
            <label
              key={tf.name}
              className="flex items-center gap-2 cursor-pointer select-none text-sm text-gray-700"
            >
              <button
                type="button"
                role="switch"
                aria-checked={form[tf.name] === 1}
                onClick={() => handleToggle(tf.name)}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${
                  form[tf.name] === 1 ? "bg-primary-600" : "bg-gray-300"
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition ${
                    form[tf.name] === 1 ? "translate-x-4" : "translate-x-0.5"
                  }`}
                />
              </button>
              {tf.label}
            </label>
          ))}
        </div>
      </fieldset>

      {/* ── Education + Numeric inputs ── */}
      <fieldset>
        <legend className="text-sm font-semibold text-gray-700 mb-2">
          Health Metrics
        </legend>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {numberFields.map((nf) => (
            <div key={nf.name}>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                {nf.label}
              </label>
              <input
                type="number"
                name={nf.name}
                value={form[nf.name]}
                onChange={handleChange}
                placeholder={nf.placeholder}
                step="any"
                required
                className={inputClass}
              />
            </div>
          ))}
        </div>
      </fieldset>

      <button
        type="submit"
        disabled={loading}
        className="w-full sm:w-auto bg-primary-600 text-white font-medium px-8 py-2.5 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition"
      >
        {loading ? <Spinner size="sm" /> : "Predict Heart Disease Risk"}
      </button>
    </form>
  );
};

export default HeartForm;
