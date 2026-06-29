/**
 * Drag-and-drop X-ray image uploader with preview.
 */
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, AlertCircle } from "lucide-react";
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
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
          transition-all duration-300
          ${isDragActive ? "border-blue-400 bg-blue-500/10" : "border-slate-600 hover:border-blue-500 hover:bg-slate-800/50"}
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
                className="mx-auto max-h-64 rounded-xl object-contain"
              />
              {!isLoading && (
                <button
                  onClick={(e) => { e.stopPropagation(); clear(); }}
                  className="absolute top-2 right-2 bg-red-500 rounded-full p-1 hover:bg-red-600 transition-colors"
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
              <Upload className="mx-auto text-slate-400" size={48} />
              <div>
                <p className="text-slate-200 font-medium">
                  {isDragActive ? "Drop the X-ray here" : "Drag & drop chest X-ray"}
                </p>
                <p className="text-slate-500 text-sm mt-1">PNG or JPEG · Max 10 MB</p>
              </div>
              <button className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors">
                Browse Files
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 px-4 py-3 rounded-lg border border-red-500/20"
        >
          <AlertCircle size={16} />
          {error}
        </motion.div>
      )}
    </div>
  );
}
