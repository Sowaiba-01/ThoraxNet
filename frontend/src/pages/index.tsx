import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import Link from "next/link";
import { Loader2, ChevronRight, Scan, TrendingUp } from "lucide-react";
import Layout from "@/components/Layout";
import { getScanHistory, ScanRecord } from "@/lib/api";

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function getTopFinding(record: ScanRecord): string {
  const present = record.findings.filter(f => f.present);
  if (present.length === 0) return "No findings";
  const top = present.sort((a, b) => b.probability - a.probability)[0];
  return top.name.replace("_", " ");
}

export default function Home() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [history, setHistory] = useState<ScanRecord[]>([]);

  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace("/login");
    }
  }, [status, router]);

  useEffect(() => {
    if (status === "authenticated") {
      setHistory(getScanHistory());
    }
  }, [status]);

  if (status === "loading") {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <Loader2 className="text-emerald-500 animate-spin" size={28} />
      </div>
    );
  }

  if (status === "unauthenticated") return null;

  const firstName = session?.user?.name?.split(" ")[0] ?? "there";
  const initials = session?.user?.name
    ? session.user.name.split(" ").map((n: string) => n[0]).join("").slice(0, 2).toUpperCase()
    : "U";

  const totalScans = history.length;
  const positiveScans = history.filter(r => r.findings.some(f => f.present)).length;

  return (
    <Layout title="Home">
      <div className="space-y-5">

        {/* Greeting */}
        <div className="flex items-center justify-between pt-1">
          <div>
            <p className="text-xs text-gray-400">Good day,</p>
            <h2 className="text-xl font-bold text-gray-900">{firstName}</h2>
          </div>
          <div className="w-10 h-10 rounded-full bg-emerald-100 border-2 border-emerald-200 flex items-center justify-center text-emerald-700 font-bold text-sm overflow-hidden">
            {session?.user?.image ? (
              <img src={session.user.image} className="w-full h-full object-cover" alt="avatar" />
            ) : (
              initials
            )}
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3">
          <div className="mint-card p-3 text-center">
            <p className="text-2xl font-bold text-emerald-600">{totalScans}</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Total scans</p>
          </div>
          <div className="mint-card p-3 text-center">
            <p className="text-2xl font-bold text-emerald-600">0.82</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Model AUC</p>
          </div>
          <div className="mint-card p-3 text-center">
            <p className="text-2xl font-bold text-emerald-600">14</p>
            <p className="text-[10px] text-gray-400 mt-0.5">Pathologies</p>
          </div>
        </div>

        {/* New scan CTA */}
        <Link href="/scan">
          <div className="rounded-2xl p-5 flex items-center gap-4 cursor-pointer hover:opacity-95 transition-opacity"
               style={{ background: "linear-gradient(135deg, #10b981, #059669)" }}>
            <div className="w-11 h-11 rounded-xl bg-white/20 flex items-center justify-center flex-shrink-0">
              <Scan size={22} className="text-white" />
            </div>
            <div className="flex-1">
              <p className="font-bold text-white text-base">New scan</p>
              <p className="text-xs text-white/75 mt-0.5">Upload a chest X-ray to analyze</p>
            </div>
            <ChevronRight size={18} className="text-white/70" />
          </div>
        </Link>

        {/* Model info card */}
        <div className="mint-card p-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center flex-shrink-0">
            <TrendingUp size={16} className="text-emerald-600" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-800">BioMedCLIP ViT-B/16</p>
            <p className="text-xs text-gray-400 mt-0.5">Monte Carlo Dropout · 20 passes · GradCAM · Groq report</p>
          </div>
          <span className="text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full flex-shrink-0">
            v1.0.0
          </span>
        </div>

        {/* Recent scans */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Recent scans</p>
            {totalScans > 0 && (
              <Link href="/stats" className="text-xs text-emerald-600 font-medium">View all</Link>
            )}
          </div>

          {history.length === 0 ? (
            <div className="mint-card p-6 text-center">
              <div className="w-12 h-12 bg-emerald-50 rounded-2xl flex items-center justify-center mx-auto mb-3">
                <Scan size={22} className="text-emerald-400" />
              </div>
              <p className="text-sm font-medium text-gray-600">No scans yet</p>
              <p className="text-xs text-gray-400 mt-1">Upload a chest X-ray to get started</p>
              <Link href="/scan">
                <button className="mt-4 px-5 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl text-sm font-medium transition-colors">
                  Start first scan
                </button>
              </Link>
            </div>
          ) : (
            <div className="mint-card divide-y divide-emerald-50">
              {history.slice(0, 5).map((record, i) => {
                const presentCount = record.findings.filter(f => f.present).length;
                const topFinding = getTopFinding(record);
                return (
                  <div key={record.id} className="flex items-center gap-3 p-3">
                    <div className="w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center flex-shrink-0 text-base">
                      🫁
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-gray-800">Scan #{totalScans - i}</p>
                      <p className="text-xs text-gray-400">{timeAgo(record.timestamp)}</p>
                    </div>
                    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full flex-shrink-0 ${
                      presentCount === 0
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-red-50 text-red-600"
                    }`}>
                      {presentCount === 0 ? "Normal" : topFinding}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <p className="text-center text-xs text-gray-300 pb-2">
          ChestAI · BioMedCLIP · For research use only · Not FDA cleared
        </p>
      </div>
    </Layout>
  );
}
