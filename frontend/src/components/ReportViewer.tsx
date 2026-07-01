/**
 * Structured radiology report display — mint & white theme.
 */
import { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Copy, Check, Clock, AlertTriangle } from "lucide-react";

interface Props {
  report: string;
  inferenceTimeMs: number;
  modelVersion: string;
}

function parseReport(report: string): Record<string, string> {
  const sections: Record<string, string> = {};
  const sectionPattern = /^(FINDINGS|IMPRESSION|RECOMMENDATION):\s*$/gm;
  const parts = report.split(sectionPattern);
  for (let i = 1; i < parts.length; i += 2) {
    sections[parts[i].trim()] = parts[i + 1]?.trim() ?? "";
  }
  if (Object.keys(sections).length === 0) sections["REPORT"] = report;
  return sections;
}

const SECTION_STYLES: Record<string, { label: string; bg: string; border: string; text: string; dot: string }> = {
  FINDINGS:       { label: "Findings",       bg: "bg-emerald-50",  border: "border-emerald-200", text: "text-emerald-800", dot: "bg-emerald-500" },
  IMPRESSION:     { label: "Impression",     bg: "bg-indigo-50",   border: "border-indigo-200",  text: "text-indigo-800",  dot: "bg-indigo-500" },
  RECOMMENDATION: { label: "Recommendation", bg: "bg-amber-50",    border: "border-amber-200",   text: "text-amber-800",   dot: "bg-amber-500" },
  REPORT:         { label: "Report",         bg: "bg-gray-50",     border: "border-gray-200",    text: "text-gray-800",    dot: "bg-gray-400" },
};

export default function ReportViewer({ report, inferenceTimeMs, modelVersion }: Props) {
  const [copied, setCopied] = useState(false);
  const sections = parseReport(report);

  const handleCopy = () => {
    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="mint-card p-5 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-emerald-100 rounded-lg flex items-center justify-center">
            <FileText className="text-emerald-600" size={16} />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900 text-sm">Radiology Report</h2>
            <span className="text-xs text-gray-400">AI-generated · v{modelVersion}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-xs text-gray-400">
            <Clock size={11} />
            {inferenceTimeMs.toFixed(0)}ms
          </span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-emerald-600 bg-gray-100 hover:bg-emerald-50 border border-gray-200 hover:border-emerald-200 px-3 py-1.5 rounded-lg transition-all"
          >
            {copied ? <Check size={13} className="text-emerald-500" /> : <Copy size={13} />}
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2.5 text-xs text-amber-700">
        <AlertTriangle size={13} className="flex-shrink-0 mt-0.5" />
        <span>AI-generated report. Not a substitute for professional radiologist review.</span>
      </div>

      {/* Sections */}
      <div className="space-y-3">
        {Object.entries(sections).map(([key, text]) => {
          const style = SECTION_STYLES[key] ?? SECTION_STYLES["REPORT"];
          return (
            <div key={key} className={`${style.bg} border ${style.border} rounded-xl p-4`}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-2 h-2 rounded-full ${style.dot}`} />
                <span className={`text-xs font-semibold uppercase tracking-wider ${style.text}`}>
                  {style.label}
                </span>
              </div>
              <p className={`text-sm leading-relaxed ${style.text}`}>{text}</p>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
