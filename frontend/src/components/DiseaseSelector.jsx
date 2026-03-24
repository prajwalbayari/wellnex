// ─────────────────────────────────────────────────────────────
// DiseaseSelector – disease type selection cards
// ─────────────────────────────────────────────────────────────
import { FiHeart, FiDroplet, FiWind, FiTarget } from "react-icons/fi";

const diseases = [
  {
    key: "diabetes",
    label: "Diabetes",
    icon: FiDroplet,
    color: "text-blue-600 bg-blue-100",
    description: "Predict diabetes risk from health metrics",
  },
  {
    key: "heart",
    label: "Heart Disease",
    icon: FiHeart,
    color: "text-red-600 bg-red-100",
    description: "Assess cardiovascular disease risk",
  },
  {
    key: "lung",
    label: "Lung Cancer",
    icon: FiWind,
    color: "text-teal-600 bg-teal-100",
    description: "Analyse CT scan images for lung cancer",
  },
  {
    key: "breast",
    label: "Breast Cancer",
    icon: FiTarget,
    color: "text-pink-600 bg-pink-100",
    description: "Analyse histopathology images",
  },
];

const DiseaseSelector = ({ selected, onSelect }) => {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {diseases.map(({ key, label, icon: Icon, color, description }) => (
        <button
          key={key}
          onClick={() => onSelect(key)}
          className={`p-5 rounded-xl border-2 text-left transition-all duration-200 hover:shadow-md ${
            selected === key
              ? "border-primary-500 bg-primary-50 shadow-md"
              : "border-gray-200 bg-white hover:border-gray-300"
          }`}
        >
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-3 ${color}`}>
            <Icon className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-sm">{label}</h3>
          <p className="text-xs text-gray-500 mt-1">{description}</p>
        </button>
      ))}
    </div>
  );
};

export default DiseaseSelector;
