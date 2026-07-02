import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { Loader2 } from "lucide-react";
import Layout from "@/components/Layout";
import { getScanHistory, ScanRecord } from "@/lib/api";

const CLASS_AUC: { name: string; auc: number }[] = [
  { name: "Cardiomegaly",       auc: 0.888 },
  { name: "Hernia",             auc: 0.872 },
  { name: "Edema",              auc: 0.851 },
  { name: "Effusion",           auc: 0.834 },
  { name: "Emphysema",          auc: 0.823 },
  { name: "Pneumothorax",       auc: 0.793 },
  { name: "Fibrosis",           auc: 0.782 },
  { name: "Mass",               auc: 0.776 },
  { name: "Nodule",             auc: 0.754 },
  { name: "Atelectasis",        auc: 0.745 },
  { name: "Consolidation",      auc: 0.736 },
  { name: "Pleural_Thickening", auc: 0.728 },
  { name: "Infiltration",       auc: 0.704 },
  { name: "Pneumonia",          auc: 0.695 },
];

function aucColor(auc: number) {
  if (auc >= 0.85) return "#10b981";
  if (auc >= 0.75) return "#34d399";
  return "#6ee7b7";
}

export default function StatsPage() {
  const { status } = useSession();
  const router = useRouter();
  const [history, setHistory] = useState<ScanRecord[]>([]);

  useEffect(() => {
    if (status === "unauthenticated") router.replace("/login");
  }, [status, router]);

  useEffect(() => {
    if (status === "authenticated") setHistory(getScanHistory());
  }, [status]);

  if (status === "loading" || status === "unauthenticated") {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <Loader2 className="text-emerald-500 animate-spin" size={28} />
      </div>
    );
  }

  const totalScans = history.length;
  const positiveScans = history.filter(r => r.findings.some(f => f.present)).length;
  const avgInferenceMs = totalScans > 0
    ? Math.round(history.reduce((s, r) => s + r.inferenceMs, 0) / totalScans)
    : 0;

  const findingCounts: Record<string, number> = {};
  history.forEach(r => r.findings.filter(f => f.present).forEach(f => {
    findingCounts[f.name] = (findingCounts[f.name] || 0) + 1;
  }));
  const sortedFindings = Object.entries(findingCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);

  const meanAUC = (CLASS_AUC.reduce((s, c) => s + c.auc, 0) / CLASS_AUC.length).toFixed(4);

  return (
    <Layout title="Analytics">
      <div className="space-y-5">

        <div>
          <p className="text-xs text-gray-400 mb-1">Model performance</p>
          <h2 className="text-xl font-bold text-gray-900">Analytics</h2>
        </div>

        {/* Model metrics */}
        <div className="grid grid-cols-3 gap-3">
          <div className="mint-card p-3 text-center">
            <p className="text-xl font-bold text-emerald-600">{meanAUC}</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Mean AUC</p>
          </div>
          <div className="mint-card p-3 text-center">
            <p className="text-xl font-bold text-emerald-600">14</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Pathologies</p>
          </div>
          <div className="mint-card p-3 text-center">
            <p className="text-xl font-bold text-emerald-600">112k</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Training imgs</p>
          </div>
        </div>

        {/* Your usage */}
        <div className="grid grid-cols-3 gap-3">
          <div className="mint-card p-3 text-center">
            <p className="text-xl font-bold text-gray-800">{totalScans}</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Your scans</p>
          </div>
          <div className="mint-card p-3 text-center">
            <p className="text-xl font-bold text-gray-800">{positiveScans}</p>
            <p className="text-[10px] text-gray-400 mt-0.5">With findings</p>
          </div>
          <div className="mint-card p-3 text-center">
            <p className="text-xl font-bold text-gray-800">{avgInferenceMs > 0 ? `${avgInferenceMs}ms` : "—"}</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Avg speed</p>
          </div>
        </div>

        {/* Per-class AUC */}
        <div className="mint-card p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Per-class AUC (NIH ChestX-ray14)
          </h3>
          <div className="space-y-2.5">
            {CLASS_AUC.map(({ name, auc }) => (
              <div key={name} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-36 flex-shrink-0 truncate">
                  {name.replace("_", " ")}
                </span>
                <div className="flex-1 h-2 bg-emerald-50 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${auc * 100}%`, background: aucColor(auc) }}
                  />
                </div>
                <span className="text-xs font-semibold text-emerald-700 w-10 text-right flex-shrink-0">
                  {auc.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Your top findings */}
        {totalScans > 0 && sortedFindings.length > 0 && (
          <div className="mint-card p-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
              Your most frequent findings
            </h3>
            <div className="space-y-2.5">
              {sortedFindings.map(([name, count]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-xs text-gray-500 w-36 flex-shrink-0 truncate">
                    {name.replace("_", " ")}
                  </span>
                  <div className="flex-1 h-2 bg-emerald-50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-400 rounded-full"
                      style={{ width: `${(count / totalScans) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-semibold text-gray-600 w-10 text-right flex-shrink-0">
                    {count}×
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Model info */}
        <div className="mint-card p-4 space-y-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Model info</h3>
          {[
            ["Architecture", "BioMedCLIP ViT-B/16"],
            ["Uncertainty", "Monte Carlo Dropout (20 passes)"],
            ["Explainability", "ViT-GradCAM"],
            ["Report", "Groq LLaMA-3.3-70b-versatile"],
            ["Dataset", "NIH ChestX-ray14 (112,120 images)"],
            ["Version", "v1.0.0"],
          ].map(([label, value]) => (
            <div key={label} className="flex justify-between items-start gap-4">
              <span className="text-xs text-gray-400 flex-shrink-0">{label}</span>
              <span className="text-xs font-medium text-gray-700 text-right">{value}</span>
            </div>
          ))}
        </div>

        <p className="text-center text-xs text-gray-300 pb-2">
          ChestAI · BioMedCLIP · For research use only · Not FDA cleared
        </p>
      </div>
    </Layout>
  );
}
