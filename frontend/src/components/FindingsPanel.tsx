/**
 * Per-class probability bars with uncertainty visualization — mint & white theme.
 */
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { FindingResult } from "@/lib/api";

interface Props {
  findings: FindingResult[];
}

const CLASS_COLORS: Record<string, string> = {
  Atelectasis:       "#10b981",
  Cardiomegaly:      "#6366f1",
  Effusion:          "#0ea5e9",
  Infiltration:      "#f59e0b",
  Mass:              "#ef4444",
  Nodule:            "#f97316",
  Pneumonia:         "#dc2626",
  Pneumothorax:      "#b91c1c",
  Consolidation:     "#06b6d4",
  Edema:             "#8b5cf6",
  Emphysema:         "#84cc16",
  Fibrosis:          "#14b8a6",
  Pleural_Thickening:"#a78bfa",
  Hernia:            "#fb923c",
};

export default function FindingsPanel({ findings }: Props) {
  const sorted  = [...findings].sort((a, b) => b.probability - a.probability);
  const present = sorted.filter((f) => f.present);
  const absent  = sorted.filter((f) => !f.present);

  return (
    <div className="space-y-5">
      {present.length > 0 ? (
        <section>
          <h3 className="text-xs font-semibold text-emerald-700 uppercase tracking-wider mb-3">
            Findings Detected ({present.length})
          </h3>
          <div className="space-y-2.5">
            {present.map((f) => <FindingRow key={f.name} finding={f} />)}
          </div>
        </section>
      ) : (
        <div className="flex items-center gap-3 p-4 bg-emerald-50 border border-emerald-200 rounded-2xl">
          <CheckCircle className="text-emerald-500 flex-shrink-0" size={20} />
          <p className="text-emerald-800 text-sm font-medium">
            No significant pathology detected above threshold.
          </p>
        </div>
      )}

      <section>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Not Detected ({absent.length})
        </h3>
        <div className="space-y-1.5">
          {absent.map((f) => <FindingRow key={f.name} finding={f} compact />)}
        </div>
      </section>
    </div>
  );
}

function FindingRow({ finding: f, compact = false }: { finding: FindingResult; compact?: boolean }) {
  const color  = CLASS_COLORS[f.name] || "#10b981";
  const pct    = Math.round(f.probability * 100);
  const uncPct = Math.round(f.uncertainty * 100);

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      className={`${compact ? "py-2 px-3" : "p-4"} rounded-xl border ${
        f.present
          ? "bg-white border-emerald-100 shadow-sm"
          : "bg-gray-50/50 border-gray-100"
      }`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          {f.present ? (
            f.high_uncertainty ? (
              <AlertTriangle size={14} className="text-amber-500 flex-shrink-0" />
            ) : (
              <CheckCircle size={14} className="text-emerald-500 flex-shrink-0" />
            )
          ) : (
            <XCircle size={14} className="text-gray-300 flex-shrink-0" />
          )}
          <span className={`text-sm font-medium ${f.present ? "text-gray-800" : "text-gray-400"}`}>
            {f.name.replace("_", " ")}
          </span>
          {f.high_uncertainty && f.present && (
            <span className="text-[10px] bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full font-medium">
              Review needed
            </span>
          )}
        </div>
        <span className={`text-sm font-bold tabular-nums ${f.present ? "text-gray-800" : "text-gray-400"}`}>
          {pct}%
        </span>
      </div>

      {/* Probability bar */}
      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full probability-bar"
          style={{ backgroundColor: f.present ? color : "#d1d5db" }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
        />
      </div>

      {/* Uncertainty band */}
      {f.present && uncPct > 0 && (
        <div className="mt-1 flex items-center gap-1.5">
          <div className="flex-1 h-0.5 bg-gray-100 rounded-full relative overflow-hidden">
            <div
              className="absolute h-full bg-amber-200 rounded-full"
              style={{
                left:  `${Math.max(0, pct - uncPct)}%`,
                width: `${Math.min(uncPct * 2, 100 - Math.max(0, pct - uncPct))}%`,
              }}
            />
          </div>
          <span className="text-[10px] text-gray-400">±{uncPct}%</span>
        </div>
      )}
    </motion.div>
  );
}
