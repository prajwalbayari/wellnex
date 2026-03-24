// ─────────────────────────────────────────────────────────────
// ImageUpload – reusable image upload component (lung / breast)
// ─────────────────────────────────────────────────────────────
import { useState, useRef, useEffect } from "react";
import { FiUploadCloud, FiX } from "react-icons/fi";
import Spinner from "./Spinner";

const ImageUpload = ({ diseaseType, onSubmit, loading }) => {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const inputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  // Revoke object URL on unmount or when preview changes
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onSubmit(file);
  };

  const labels = {
    lung: { title: "Lung Cancer Scan", desc: "Upload a CT scan image" },
    breast: { title: "Breast Cancer Image", desc: "Upload a histopathology image" },
  };

  const { title, desc } = labels[diseaseType] || labels.lung;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <p className="text-sm text-gray-500">{desc}</p>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-primary-400 transition"
      >
        {preview ? (
          <div className="relative inline-block">
            <img
              src={preview}
              alt="Preview"
              className="max-h-64 rounded-lg mx-auto"
            />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1"
            >
              <FiX className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <FiUploadCloud className="w-12 h-12" />
            <p className="font-medium">Click to upload {title}</p>
            <p className="text-xs">JPEG, PNG or WebP — max 10 MB</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      <button
        type="submit"
        disabled={loading || !file}
        className="w-full sm:w-auto bg-primary-600 text-white font-medium px-8 py-2.5 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition"
      >
        {loading ? <Spinner size="sm" /> : `Analyse ${title}`}
      </button>
    </form>
  );
};

export default ImageUpload;
