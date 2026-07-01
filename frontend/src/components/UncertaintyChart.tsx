/**
 * Radar chart showing per-class uncertainty — mint & white theme.
 */
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
} from "recharts";
import { FindingResult } from "@/lib/api";

interface Props {
  findings: FindingResult[];
}

export default function UncertaintyChart({ findings }: Props) {
  const data = findings.map((f) => ({
    subject: f.name.replace("_", " "),
    uncertainty: Math.round(f.uncertainty * 100),
    probability: Math.round(f.probability * 100),
  }));

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-emerald-400 inline-block rounded" />
          Uncertainty (MC Dropout std)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-indigo-300 inline-block rounded" />
          Probability
        </span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
          <PolarGrid stroke="#d1fae5" />
          <PolarAngleAxis
            dataKey="subject"
            tick={{ fontSize: 10, fill: "#6b7280" }}
          />
          <Radar
            name="Uncertainty"
            dataKey="uncertainty"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.2}
            strokeWidth={1.5}
          />
          <Radar
            name="Probability"
            dataKey="probability"
            stroke="#a5b4fc"
            fill="#a5b4fc"
            fillOpacity={0.1}
            strokeWidth={1}
            strokeDasharray="4 2"
          />
          <Tooltip
            contentStyle={{
              background: "#ffffff",
              border: "1px solid #d1fae5",
              borderRadius: "10px",
              fontSize: "12px",
              color: "#111827",
              boxShadow: "0 4px 12px rgba(16,185,129,0.08)",
            }}
            formatter={(val: number, name: string) => [`${val}%`, name]}
          />
        </RadarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 text-center">
        High uncertainty classes require radiologist review.
      </p>
    </div>
  );
}
