/**
 * Structured radiology report display with section parsing and copy button.
 */
import { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Copy, Check, Clock } from "lucide-react";

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

const SECTION_COLORS: Record<string, string> = {
  FINDINGS: "text-blue-400 border-blue-500/30",
  IMPRESSION: "text-purple-400 border-purple-500/30",
  RECOMMENDATION: "text-amber-400 border-amber-500/30",
  REPORT: "text-slate-400 border-slate-500/30",
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
      className="glass-card p-6 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="text-blue-400" size={18} />
          <h2 className="font-semibold text-slate-100">Radiology Report</h2>
          <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded">
            AI-generated · v{modelVersion}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-xs text-slate-500">
            <Clock size={12} />
            {inferenceTimeMs.toFixed(0)}ms
          </span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-2.5 text-xs text-amber-300">
        ⚠️ This report is AI-generated for research purposes only. Not a substitute for radiologist interpretation.
      </div>

      {/* Report sections */}
      <div className="space-y-4">
        {Object.entries(sections).map(([section, content]) => (
          <div key={section}>
            <h3 className={`text-xs font-bold tracking-widest uppercase mb-2 ${SECTION_COLORS[section]?.split(" ")[0] ?? "text-slate-400"}`}>
              {section}
            </h3>
            <div className={`border-l-2 pl-4 ${SECTION_COLORS[section]?.split(" ")[1] ?? "border-slate-500/30"}`}>
              <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{content}</p>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
