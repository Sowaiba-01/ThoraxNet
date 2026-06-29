/**
 * Main page — X-ray upload + analysis results.
 */
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Github, AlertCircle, Loader2 } from "lucide-react";
import XRayUploader from "@/components/XRayUploader";
import FindingsPanel from "@/components/FindingsPanel";
import ReportViewer from "@/components/ReportViewer";
import UncertaintyChart from "@/components/UncertaintyChart";
import { analyzeXray, PredictionResponse } from "@/lib/api";

type AppState = "idle" | "loading" | "results" | "error";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [patientAge, setPatientAge] = useState<string>("");
  const [patientGender, setPatientGender] = useState<string>("");

  const handleFileSelect = async (file: File) => {
    setState("loading");
    setResults(null);
    setErrorMsg("");
    try {
      const age = patientAge ? parseFloat(patientAge) : undefined;
      const gender = patientGender || undefined;
      const data = await analyzeXray(file, age, gender);
      setResults(data);
      setState("results");
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Analysis failed. Please check the API is running.";
      setErrorMsg(msg);
      setState("error");
    }
  };

  const reset = () => {
    setState("idle");
    setResults(null);
    setErrorMsg("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800/50 sticky top-0 z-50 backdrop-blur-md bg-slate-950/80">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Activity size={20} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-white tracking-tight">ChestAI</h1>
              <p className="text-xs text-slate-500">Chest X-Ray Diagnostic Platform</p>
            </div>
          </div>
          <a
            href="https://github.com/yourusername/chestai"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors text-sm"
          >
            <Github size={18} />
            <span className="hidden sm:inline">View on GitHub</span>
          </a>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-3 pt-4"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-white">
            AI Chest X-Ray Analysis
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto text-sm sm:text-base">
            BioMedCLIP-powered detection of 14 pathologies with Monte Carlo uncertainty
            estimation and auto-generated radiology reports.
          </p>
          <div className="flex items-center justify-center gap-4 text-xs text-slate-500 pt-1">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full inline-block" />
              NIH ChestX-ray14 trained
            </span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full inline-block" />
              MC Dropout uncertainty
            </span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-purple-400 rounded-full inline-block" />
              GradCAM explainability
            </span>
          </div>
        </motion.div>

        {/* Upload card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card p-6 space-y-5"
        >
          {/* Optional patient metadata */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-slate-400 mb-1.5 block">Patient Age (optional)</label>
              <input
                type="number"
                min={0}
                max={120}
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                placeholder="e.g. 45"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 mb-1.5 block">Patient Gender (optional)</label>
              <select
                value={patientGender}
                onChange={(e) => setPatientGender(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 transition-colors"
              >
                <option value="">Not specified</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </div>
          </div>

          <XRayUploader onFileSelect={handleFileSelect} isLoading={state === "loading"} />
        </motion.div>

        {/* Loading state */}
        <AnimatePresence>
          {state === "loading" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4 py-12"
            >
              <Loader2 className="animate-spin text-blue-400" size={40} />
              <div className="text-center">
                <p className="text-slate-200 font-medium">Analyzing X-ray...</p>
                <p className="text-slate-500 text-sm mt-1">Running 20 MC Dropout passes · Generating GradCAM · Building report</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error state */}
        <AnimatePresence>
          {state === "error" && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-start gap-3 p-5 bg-red-500/10 border border-red-500/20 rounded-xl"
            >
              <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="text-red-300 font-medium">Analysis Failed</p>
                <p className="text-red-400/70 text-sm mt-1">{errorMsg}</p>
                <button onClick={reset} className="mt-3 text-xs text-blue-400 hover:underline">
                  Try again
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {state === "results" && results && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white">Analysis Results</h2>
                <button
                  onClick={reset}
                  className="text-sm text-slate-400 hover:text-white transition-colors"
                >
                  Analyze another
                </button>
              </div>

              <div className="grid lg:grid-cols-2 gap-6">
                {/* Findings */}
                <div className="glass-card p-6">
                  <h3 className="font-semibold text-slate-200 mb-4">Pathology Findings</h3>
                  <FindingsPanel findings={results.findings} />
                </div>

                {/* Uncertainty chart */}
                <UncertaintyChart findings={results.findings} />
              </div>

              {/* Radiology Report */}
              <ReportViewer
                report={results.report}
                inferenceTimeMs={results.inference_time_ms}
                modelVersion={results.model_version}
              />

              {/* GradCAM notice */}
              {results.gradcam_available && (
                <div className="glass-card p-4 text-sm text-slate-400">
                  <span className="text-blue-400 font-medium">GradCAM heatmaps</span> available for:{" "}
                  {results.gradcam_classes.map((c) => c.replace("_", " ")).join(", ")}. 
                  Retrieve via <code className="text-slate-300 bg-slate-800 px-1 rounded">/api/v1/gradcam/&#123;session_id&#125;/&#123;class&#125;</code>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="text-center text-xs text-slate-600 pb-8 pt-4">
          ChestAI is a research tool. Not for clinical use. · 
          Model trained on NIH ChestX-ray14 · 
          Built with BioMedCLIP + FastAPI + Next.js
        </footer>
      </main>
    </div>
  );
}
