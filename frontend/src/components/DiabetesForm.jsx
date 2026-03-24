// ─────────────────────────────────────────────────────────────
// DiabetesForm – dynamic form for diabetes prediction
// ─────────────────────────────────────────────────────────────
import { useState } from "react";
import Spinner from "./Spinner";

/* ── Field definitions ─────────────────────────────────────── */

const selectFields = [
  {
    name: "gender",
    label: "Gender",
    options: ["Female", "Male", "Other"],
  },
  {
    name: "ethnicity",
    label: "Ethnicity",
    options: ["Asian", "Black", "Hispanic", "Other", "White"],
  },
  {
    name: "education_level",
    label: "Education Level",
    options: ["Bachelor", "Highschool", "No formal", "Postgraduate"],
  },
  {
    name: "income_level",
    label: "Income Level",
    options: ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"],
  },
  {
    name: "employment_status",
    label: "Employment Status",
    options: ["Employed", "Retired", "Student", "Unemployed"],
  },
  {
    name: "smoking_status",
    label: "Smoking Status",
    options: ["Current", "Former", "Never"],
  },
];

const numberFields = [
  { name: "age", label: "Age", placeholder: "e.g. 45" },
  { name: "bmi", label: "BMI", placeholder: "e.g. 27.5" },
  { name: "waist_to_hip_ratio", label: "Waist-to-Hip Ratio", placeholder: "e.g. 0.85" },
  { name: "systolic_bp", label: "Systolic BP", placeholder: "e.g. 130" },
  { name: "diastolic_bp", label: "Diastolic BP", placeholder: "e.g. 85" },
  { name: "heart_rate", label: "Heart Rate", placeholder: "bpm, e.g. 75" },
  { name: "glucose_fasting", label: "Fasting Glucose", placeholder: "mg/dL" },
  { name: "glucose_postprandial", label: "Postprandial Glucose", placeholder: "mg/dL" },
  { name: "hba1c", label: "HbA1c (%)", placeholder: "e.g. 6.5" },
  { name: "insulin_level", label: "Insulin Level", placeholder: "µU/mL" },
  { name: "cholesterol_total", label: "Total Cholesterol", placeholder: "mg/dL" },
  { name: "hdl_cholesterol", label: "HDL Cholesterol", placeholder: "mg/dL" },
  { name: "ldl_cholesterol", label: "LDL Cholesterol", placeholder: "mg/dL" },
  { name: "triglycerides", label: "Triglycerides", placeholder: "mg/dL" },
  { name: "alcohol_consumption_per_week", label: "Alcohol (drinks/week)", placeholder: "e.g. 3" },
  { name: "physical_activity_minutes_per_week", label: "Physical Activity (min/week)", placeholder: "e.g. 150" },
  { name: "diet_score", label: "Diet Score (1-10)", placeholder: "e.g. 5" },
  { name: "sleep_hours_per_day", label: "Sleep (hours/day)", placeholder: "e.g. 7" },
  { name: "screen_time_hours_per_day", label: "Screen Time (hours/day)", placeholder: "e.g. 4" },
  { name: "diabetes_risk_score", label: "Diabetes Risk Score (0-10)", placeholder: "e.g. 0" },
];

const toggleFields = [
  { name: "family_history_diabetes", label: "Family History of Diabetes" },
  { name: "hypertension_history", label: "Hypertension History" },
  { name: "cardiovascular_history", label: "Cardiovascular History" },
];

/* ── Default form state ────────────────────────────────────── */

const defaultState = {
  // selects
  gender: "Male",
  ethnicity: "White",
  education_level: "Bachelor",
  income_level: "Middle",
  employment_status: "Employed",
  smoking_status: "Never",
  // numbers
  ...Object.fromEntries(numberFields.map((f) => [f.name, ""])),
  // toggles
  ...Object.fromEntries(toggleFields.map((f) => [f.name, 0])),
};

/* ── Component ─────────────────────────────────────────────── */

const DiabetesForm = ({ onSubmit, loading }) => {
  const [form, setForm] = useState({ ...defaultState });

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const handleToggle = (name) =>
    setForm((prev) => ({ ...prev, [name]: prev[name] === 1 ? 0 : 1 }));

  const handleSubmit = (e) => {
    e.preventDefault();
    const payload = {};
    // keep strings for categorical fields
    for (const sf of selectFields) payload[sf.name] = form[sf.name];
    // convert numbers
    for (const nf of numberFields)
      payload[nf.name] = parseFloat(form[nf.name]) || 0;
    // toggles already 0/1
    for (const tf of toggleFields) payload[tf.name] = form[tf.name];
    onSubmit(payload);
  };

  const inputClass =
    "w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-400 focus:border-transparent outline-none transition";

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* ── Categorical selects ── */}
      <fieldset>
        <legend className="text-sm font-semibold text-gray-700 mb-2">
          Demographics &amp; Lifestyle
        </legend>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {selectFields.map((sf) => (
            <div key={sf.name}>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                {sf.label}
              </label>
              <select
                name={sf.name}
                value={form[sf.name]}
                onChange={handleChange}
                className={inputClass}
              >
                {sf.options.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
      </fieldset>

      {/* ── Numeric inputs ── */}
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

      {/* ── Toggle switches ── */}
      <fieldset>
        <legend className="text-sm font-semibold text-gray-700 mb-2">
          Medical History
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

      <button
        type="submit"
        disabled={loading}
        className="w-full sm:w-auto bg-primary-600 text-white font-medium px-8 py-2.5 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition"
      >
        {loading ? <Spinner size="sm" /> : "Predict Diabetes Risk"}
      </button>
    </form>
  );
};

export default DiabetesForm;
