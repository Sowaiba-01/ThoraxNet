/**
 * Per-class probability bars with uncertainty visualization.
 * Bars show mean probability; error bands show ± std deviation.
 */
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { FindingResult } from "@/lib/api";

interface Props {
  findings: FindingResult[];
}

const CLASS_COLORS: Record<string, string> = {
  Atelectasis: "#3b82f6",
  Cardiomegaly: "#8b5cf6",
  Effusion: "#06b6d4",
  Infiltration: "#f59e0b",
  Mass: "#ef4444",
  Nodule: "#f97316",
  Pneumonia: "#dc2626",
  Pneumothorax: "#b91c1c",
  Consolidation: "#0ea5e9",
  Edema: "#6366f1",
  Emphysema: "#84cc16",
  Fibrosis: "#14b8a6",
  Pleural_Thickening: "#a78bfa",
  Hernia: "#fb923c",
};

export default function FindingsPanel({ findings }: Props) {
  const sorted = [...findings].sort((a, b) => b.probability - a.probability);
  const present = sorted.filter((f) => f.present);
  const absent  = sorted.filter((f) => !f.present);

  return (
    <div className="space-y-6">
      {/* Present findings */}
      {present.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Findings Detected ({present.length})
          </h3>
          <div className="space-y-3">
            {present.map((f) => (
              <FindingRow key={f.name} finding={f} />
            ))}
          </div>
        </section>
      )}

      {present.length === 0 && (
        <div className="flex items-center gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-xl">
          <CheckCircle className="text-green-400 flex-shrink-0" size={20} />
          <p className="text-green-300 text-sm">No significant pathology detected above threshold.</p>
        </div>
      )}

      {/* Absent findings (collapsed) */}
      <section>
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Not Detected ({absent.length})
        </h3>
        <div className="space-y-2">
          {absent.map((f) => (
            <FindingRow key={f.name} finding={f} compact />
          ))}
        </div>
      </section>
    </div>
  );
}

function FindingRow({ finding: f, compact = false }: { finding: FindingResult; compact?: boolean }) {
  const color = CLASS_COLORS[f.name] || "#64748b";
  const pct   = Math.round(f.probability * 100);
  const uncPct = Math.round(f.uncertainty * 100);

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      className={`${compact ? "py-1.5" : "p-4"} rounded-xl ${
        f.present ? "bg-slate-800/60 border border-slate-700/50" : "bg-slate-900/30"
      }`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          {f.present ? (
            f.high_uncertainty ? (
              <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
            ) : (
              <CheckCircle size={14} className="text-red-400 flex-shrink-0" />
            )
          ) : (
            <XCircle size={14} className="text-slate-600 flex-shrink-0" />
          )}
          <span className={`text-sm font-medium ${f.present ? "text-slate-100" : "text-slate-500"}`}>
            {f.name.replace("_", " ")}
          </span>
          {f.high_uncertainty && (
            <span className="text-xs px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded border border-amber-500/30">
              uncertain
            </span>
          )}
        </div>
        <span className={`text-xs font-mono ${f.present ? "text-slate-300" : "text-slate-600"}`}>
          {pct}% ±{uncPct}%
        </span>
      </div>

      {/* Probability bar with uncertainty band */}
      <div className="relative h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          className="absolute top-0 left-0 h-full rounded-full"
          style={{ backgroundColor: f.present ? color : "#334155" }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        />
        {/* Uncertainty band overlay */}
        <div
          className="absolute top-0 h-full opacity-20 rounded-full"
          style={{
            left: `${Math.max(0, pct - uncPct)}%`,
            width: `${Math.min(uncPct * 2, 100 - Math.max(0, pct - uncPct))}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </motion.div>
  );
}
