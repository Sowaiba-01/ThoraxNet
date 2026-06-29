/**
 * Radar chart showing per-class uncertainty (MC Dropout std deviation).
 * High uncertainty classes are flagged for radiologist review.
 */
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from "recharts";
import { FindingResult } from "@/lib/api";

interface Props {
  findings: FindingResult[];
}

export default function UncertaintyChart({ findings }: Props) {
  const data = findings
    .filter((f) => f.present || f.uncertainty > 0.08)
    .map((f) => ({
      class: f.name.replace("_", " ").replace("Pleural Thickening", "Pleural Th."),
      uncertainty: Math.round(f.uncertainty * 100),
      probability: Math.round(f.probability * 100),
    }));

  if (data.length === 0) return null;

  return (
    <div className="glass-card p-6">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">Uncertainty Distribution</h3>
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="class" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <Radar
            name="Probability"
            dataKey="probability"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.2}
          />
          <Radar
            name="Uncertainty"
            dataKey="uncertainty"
            stroke="#f59e0b"
            fill="#f59e0b"
            fillOpacity={0.15}
          />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
            labelStyle={{ color: "#f1f5f9" }}
            itemStyle={{ color: "#94a3b8" }}
          />
        </RadarChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-500 text-center mt-2">
        Blue = probability · Amber = uncertainty (MC Dropout)
      </p>
    </div>
  );
}
