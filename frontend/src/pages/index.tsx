/**
 * Main page — mint & white mobile-first design.
 */
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Github, AlertCircle, Loader2, ChevronDown, ChevronUp, BarChart2 } from "lucide-react";
import XRayUploader from "@/components/XRayUploader";
import FindingsPanel from "@/components/FindingsPanel";
import ReportViewer from "@/components/ReportViewer";
import UncertaintyChart from "@/components/UncertaintyChart";
import { analyzeXray, PredictionResponse } from "@/lib/api";

type AppState = "idle" | "loading" | "results" | "error";

export default function Home() {
  const [state, setState]               = useState<AppState>("idle");
  const [results, setResults]           = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg]         = useState<string>("");
  const [patientAge, setPatientAge]     = useState<string>("");
  const [patientGender, setPatientGender] = useState<string>("");
  const [showChart, setShowChart]       = useState(false);

  const handleFileSelect = async (file: File) => {
    setState("loading");
    setResults(null);
    setErrorMsg("");
    try {
      const age    = patientAge    ? parseFloat(patientAge) : undefined;
      const gender = patientGender || undefined;
      const data   = await analyzeXray(file, age, gender);
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
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white border-b border-emerald-100 sticky top-0 z-50">
        <div className="max-w-2xl mx-auto px-4 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 bg-emerald-500 rounded-xl flex items-center justify-center shadow-sm">
              <Activity size={18} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-gray-900 leading-tight">ChestAI</h1>
              <p className="text-[10px] text-emerald-600 font-medium tracking-wide">X-RAY DIAGNOSTIC PLATFORM</p>
            </div>
          </div>
          <a
            href="https://github.com/Sowaiba-01/chestai"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-gray-500 hover:text-emerald-600 transition-colors text-sm"
          >
            <Github size={17} />
            <span className="hidden sm:inline text-xs font-medium">GitHub</span>
          </a>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-6 space-y-5">

        {/* Hero — only show on idle */}
        <AnimatePresence>
          {state === "idle" && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="text-center pt-2 pb-1 space-y-2"
            >
              <div className="inline-flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium px-3 py-1 rounded-full">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                BioMedCLIP · NIH ChestX-ray14 · Val AUC 0.8215
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-900">
                AI Chest X-Ray Analysis
              </h2>
              <p className="text-gray-500 text-sm max-w-md mx-auto leading-relaxed">
                Detects 14 pathologies with Monte Carlo uncertainty estimation
                and auto-generated radiology reports.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Patient info */}
        <div className="mint-card p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Patient Info <span className="text-gray-300 font-normal normal-case">(optional)</span>
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Age</label>
              <input
                type="number"
                min={0}
                max={120}
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                placeholder="e.g. 45"
                disabled={state === "loading"}
                className="w-full border border-emerald-100 rounded-xl px-3 py-2.5 text-sm text-gray-800 focus:outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all placeholder-gray-300 disabled:opacity-50 bg-white"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Sex</label>
              <select
                value={patientGender}
                onChange={(e) => setPatientGender(e.target.value)}
                disabled={state === "loading"}
                className="w-full border border-emerald-100 rounded-xl px-3 py-2.5 text-sm text-gray-800 focus:outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all disabled:opacity-50 bg-white"
              >
                <option value="">Unknown</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </div>
          </div>
        </div>

        {/* Upload */}
        <div className="mint-card p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            X-Ray Image
          </h3>
          <XRayUploader onFileSelect={handleFileSelect} isLoading={state === "loading"} />
        </div>

        {/* Loading */}
        <AnimatePresence>
          {state === "loading" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="mint-card p-6 flex flex-col items-center gap-4 text-center"
            >
              <div className="relative">
                <div className="w-14 h-14 rounded-full bg-emerald-50 border-2 border-emerald-200 flex items-center justify-center">
                  <Loader2 className="text-emerald-500 animate-spin" size={24} />
                </div>
              </div>
              <div>
                <p className="font-semibold text-gray-800">Analyzing X-Ray</p>
                <p className="text-sm text-gray-400 mt-1">Running 20 MC Dropout passes · GradCAM · Report generation</p>
              </div>
              <div className="w-full bg-emerald-50 rounded-full h-1.5 overflow-hidden">
                <motion.div
                  className="h-full bg-emerald-400 rounded-full"
                  animate={{ width: ["0%", "90%"] }}
                  transition={{ duration: 4, ease: "easeInOut" }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        <AnimatePresence>
          {state === "error" && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mint-card p-5"
            >
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <AlertCircle className="text-red-500" size={18} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-gray-900 text-sm">Analysis Failed</p>
                  <p className="text-sm text-gray-500 mt-1 break-words">{errorMsg}</p>
                </div>
              </div>
              <button
                onClick={reset}
                className="mt-4 w-full py-2.5 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl text-sm font-medium transition-colors"
              >
                Try Again
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {state === "results" && results && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Summary pill */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-gray-800">Results</span>
                  <span className="bg-emerald-100 text-emerald-700 text-xs font-medium px-2 py-0.5 rounded-full">
                    {results.findings.filter(f => f.present).length} findings
                  </span>
                </div>
                <button
                  onClick={reset}
                  className="text-xs text-gray-400 hover:text-emerald-600 transition-colors font-medium"
                >
                  New scan
                </button>
              </div>

              {/* Findings */}
              <div className="mint-card p-4">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                  Pathology Detection
                </h3>
                <FindingsPanel findings={results.findings} />
              </div>

              {/* Uncertainty chart — collapsible */}
              <div className="mint-card overflow-hidden">
                <button
                  onClick={() => setShowChart(!showChart)}
                  className="w-full p-4 flex items-center justify-between text-left"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 bg-indigo-50 rounded-lg flex items-center justify-center">
                      <BarChart2 size={14} className="text-indigo-500" />
                    </div>
                    <span className="text-sm font-semibold text-gray-800">Uncertainty Analysis</span>
                  </div>
                  {showChart
                    ? <ChevronUp size={16} className="text-gray-400" />
                    : <ChevronDown size={16} className="text-gray-400" />
                  }
                </button>
                <AnimatePresence>
                  {showChart && (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: "auto" }}
                      exit={{ height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="px-4 pb-4 border-t border-emerald-50">
                        <div className="pt-4">
                          <UncertaintyChart findings={results.findings} />
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Report */}
              {results.report && (
                <ReportViewer
                  report={results.report}
                  inferenceTimeMs={results.inference_time_ms}
                  modelVersion={results.model_version}
                />
              )}

              {/* Disclaimer footer */}
              <p className="text-center text-xs text-gray-300 pb-4">
                ChestAI · BioMedCLIP · For research use only · Not FDA cleared
              </p>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}
