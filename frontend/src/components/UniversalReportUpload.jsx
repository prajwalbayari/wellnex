import { useEffect, useRef, useState } from "react";
import { FiFileText, FiUploadCloud, FiX } from "react-icons/fi";
import Spinner from "./Spinner";

const UniversalReportUpload = ({ loading, onSubmit }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const inputRef = useRef(null);

  const isImage = !!file && file.type?.startsWith("image/");

  const handleFileChange = (e) => {
    const picked = e.target.files?.[0];
    if (!picked) return;

    if (preview) URL.revokeObjectURL(preview);

    setFile(picked);
    if (picked.type?.startsWith("image/")) {
      setPreview(URL.createObjectURL(picked));
    } else {
      setPreview(null);
    }
  };

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setFile(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file || loading) return;
    onSubmit(file);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <p className="text-sm text-gray-500">
        Upload one report file (PDF, TXT, DOCX, CSV, JSON, image, or similar).
        Wellnex will process it once and evaluate diabetes, heart disease, and breast cancer models.
      </p>

      <div
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload report file"
        className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-primary-400 transition"
      >
        {file ? (
          <div className="relative inline-block max-w-full">
            {isImage && preview ? (
              <img
                src={preview}
                alt="Report preview"
                className="max-h-64 rounded-lg mx-auto"
              />
            ) : (
              <div className="flex items-center justify-center gap-2 text-gray-600">
                <FiFileText className="w-5 h-5" />
                <span className="font-medium">{file.name}</span>
              </div>
            )}
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                handleClear();
              }}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1"
            >
              <FiX className="w-4 h-4" />
            </button>
            <p className="text-xs text-gray-500 mt-2">
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <FiUploadCloud className="w-12 h-12" />
            <p className="font-medium">Click to upload report</p>
            <p className="text-xs">PDF, TXT, DOCX, CSV, JSON, PNG, JPG, WebP and more</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      <button
        type="submit"
        disabled={!file || loading}
        className="w-full sm:w-auto bg-primary-600 text-white font-medium px-8 py-2.5 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition"
      >
        {loading ? <Spinner size="sm" /> : "Analyze Full Report"}
      </button>
    </form>
  );
};

export default UniversalReportUpload;
