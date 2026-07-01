/**
 * Drag-and-drop X-ray image uploader — mint & white theme.
 */
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, AlertCircle, ImagePlus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

export default function XRayUploader({ onFileSelect, isLoading }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[], rejected: any[]) => {
      setError(null);
      if (rejected.length > 0) {
        setError("Please upload a PNG or JPEG image under 10 MB.");
        return;
      }
      const file = accepted[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      setPreview(url);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
    disabled: isLoading,
  });

  const clear = () => {
    setPreview(null);
    setError(null);
  };

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
          transition-all duration-300
          ${isDragActive
            ? "border-emerald-400 bg-emerald-50"
            : "border-emerald-200 hover:border-emerald-400 hover:bg-emerald-50/50 bg-white"}
          ${isLoading ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        <input {...getInputProps()} />
        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="relative"
            >
              <img
                src={preview}
                alt="X-ray preview"
                className="mx-auto max-h-64 rounded-xl object-contain border border-emerald-100"
              />
              {!isLoading && (
                <button
                  onClick={(e) => { e.stopPropagation(); clear(); }}
                  className="absolute top-2 right-2 bg-red-100 text-red-500 rounded-full p-1 hover:bg-red-200 transition-colors"
                >
                  <X size={14} />
                </button>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              <div className="mx-auto w-16 h-16 bg-emerald-50 border-2 border-emerald-200 rounded-2xl flex items-center justify-center">
                <ImagePlus className="text-emerald-500" size={28} />
              </div>
              <div>
                <p className="font-semibold text-gray-800">
                  {isDragActive ? "Drop the X-ray here" : "Upload Chest X-Ray"}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Drag & drop or <span className="text-emerald-600 font-medium">browse files</span>
                </p>
                <p className="text-xs text-gray-400 mt-1">PNG or JPEG · Max 10 MB</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3 text-sm"
        >
          <AlertCircle size={16} className="flex-shrink-0" />
          {error}
        </motion.div>
      )}
    </div>
  );
}
