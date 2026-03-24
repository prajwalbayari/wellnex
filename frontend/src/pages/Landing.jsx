// ─────────────────────────────────────────────────────────────
// Landing Page
// ─────────────────────────────────────────────────────────────
import { Link } from "react-router-dom";
import {
  FiHeart,
  FiDroplet,
  FiWind,
  FiTarget,
  FiShield,
  FiZap,
  FiBarChart2,
} from "react-icons/fi";

const features = [
  {
    icon: FiDroplet,
    title: "Diabetes Prediction",
    desc: "Analyse 29 key health metrics to assess diabetes risk using deep learning.",
    color: "text-blue-600 bg-blue-50",
  },
  {
    icon: FiHeart,
    title: "Heart Disease",
    desc: "Predict cardiovascular risk from clinical parameters like BP, cholesterol and more.",
    color: "text-red-600 bg-red-50",
  },
  {
    icon: FiWind,
    title: "Lung Cancer Detection",
    desc: "Upload CT scan images and let AI analyse them for early lung cancer detection.",
    color: "text-teal-600 bg-teal-50",
  },
  {
    icon: FiTarget,
    title: "Breast Cancer Screening",
    desc: "Histopathology image analysis for breast cancer classification.",
    color: "text-pink-600 bg-pink-50",
  },
];

const stats = [
  { icon: FiShield, value: "4", label: "Disease Models" },
  { icon: FiZap, value: "<2s", label: "Prediction Time" },
  { icon: FiBarChart2, value: "95%+", label: "Accuracy Target" },
];

const Landing = () => {
  return (
    <div className="overflow-hidden">
      {/* ── Hero Section ──────────────────────────────────────── */}
      <section className="relative bg-gradient-to-br from-primary-600 via-primary-700 to-primary-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
          <div className="max-w-3xl">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight">
              AI-Powered
              <br />
              <span className="text-primary-200">Multi-Disease</span>
              <br />
              Prediction Platform
            </h1>
            <p className="mt-6 text-lg text-primary-100 max-w-xl">
              Wellnex leverages state-of-the-art deep learning models to predict
              diabetes, heart disease, lung cancer and breast cancer — all from a
              single dashboard.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link
                to="/signup"
                className="bg-white text-primary-700 font-semibold px-6 py-3 rounded-lg hover:bg-primary-50 transition shadow-lg"
              >
                Get Started Free
              </Link>
              <Link
                to="/login"
                className="border border-primary-300 text-white font-semibold px-6 py-3 rounded-lg hover:bg-primary-600 transition"
              >
                Login
              </Link>
            </div>
          </div>
        </div>

        {/* Decorative circles */}
        <div className="absolute top-10 right-10 w-64 h-64 bg-primary-400 rounded-full filter blur-3xl opacity-20" />
        <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-primary-300 rounded-full filter blur-3xl opacity-10" />
      </section>

      {/* ── Stats Bar ─────────────────────────────────────────── */}
      <section className="bg-white border-b">
        <div className="max-w-5xl mx-auto px-4 py-8 flex flex-wrap justify-center gap-12">
          {stats.map(({ icon: Icon, value, label }) => (
            <div key={label} className="flex items-center gap-3">
              <Icon className="w-8 h-8 text-primary-500" />
              <div>
                <p className="text-2xl font-bold">{value}</p>
                <p className="text-xs text-gray-500">{label}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features Grid ─────────────────────────────────────── */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold">
              Predictions You Can <span className="text-primary-600">Trust</span>
            </h2>
            <p className="mt-3 text-gray-500 max-w-xl mx-auto">
              Trained on high-quality datasets and validated with medical
              benchmarks.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map(({ icon: Icon, title, desc, color }) => (
              <div
                key={title}
                className="bg-white rounded-xl p-6 border border-gray-100 hover:shadow-lg transition"
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${color}`}
                >
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="font-semibold mb-2">{title}</h3>
                <p className="text-sm text-gray-500">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────── */}
      <section className="bg-primary-700 text-white py-16">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold">Ready to check your health?</h2>
          <p className="mt-3 text-primary-200">
            Create a free account and get instant predictions.
          </p>
          <Link
            to="/signup"
            className="inline-block mt-6 bg-white text-primary-700 font-semibold px-8 py-3 rounded-lg hover:bg-primary-50 transition shadow"
          >
            Sign Up Now
          </Link>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────── */}
      <footer className="bg-gray-900 text-gray-400 text-sm py-8">
        <div className="max-w-7xl mx-auto px-4 text-center">
          © {new Date().getFullYear()} Wellnex. Built for educational &amp;
          research purposes. Not a substitute for professional medical advice.
        </div>
      </footer>
    </div>
  );
};

export default Landing;
