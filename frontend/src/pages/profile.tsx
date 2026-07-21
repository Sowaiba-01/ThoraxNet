import { useEffect, useState } from "react";
import { useSession, signOut } from "next-auth/react";
import { useRouter } from "next/router";
import { Loader2, ChevronRight, Github, LogOut, Trash2 } from "lucide-react";
import Layout from "@/components/Layout";
import { clearScanHistory, getScanHistory } from "@/lib/api";

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 flex-shrink-0 items-center rounded-full transition-colors ${checked ? "bg-emerald-500" : "bg-gray-200"}`}
      role="switch" aria-checked={checked}
    >
      <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${checked ? "translate-x-4" : "translate-x-0.5"}`} />
    </button>
  );
}

function SettingRow({ icon, iconBg, label, sub, right, onClick, last }: {
  icon: React.ReactNode; iconBg: string; label: string; sub?: string;
  right?: React.ReactNode; onClick?: () => void; last?: boolean;
}) {
  return (
    <div
      onClick={onClick}
      className={`flex items-center gap-3 py-3 ${!last ? "border-b border-emerald-50" : ""} ${onClick ? "cursor-pointer active:bg-gray-50" : ""}`}
    >
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 ${iconBg}`}>{icon}</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-800">{label}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
      </div>
      {right ?? (onClick ? <ChevronRight size={15} className="text-gray-300 flex-shrink-0" /> : null)}
    </div>
  );
}

export default function ProfilePage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  const [autoReport, setAutoReport]           = useState(true);
  const [uncertaintyWarn, setUncertaintyWarn] = useState(true);
  const [emailNotifs, setEmailNotifs]         = useState(false);
  const [scanCount, setScanCount]             = useState(0);
  const [cleared, setCleared]                 = useState(false);

  useEffect(() => {
    if (status === "unauthenticated") router.replace("/login");
  }, [status, router]);

  useEffect(() => {
    if (status === "authenticated") setScanCount(getScanHistory().length);
  }, [status]);

  if (status === "loading" || status === "unauthenticated") {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <Loader2 className="text-emerald-500 animate-spin" size={28} />
      </div>
    );
  }

  const name     = session?.user?.name ?? "User";
  const email    = session?.user?.email ?? "";
  const image    = session?.user?.image;
  const initials = name.split(" ").map((n: string) => n[0]).join("").slice(0, 2).toUpperCase();

  const handleClearHistory = () => {
    clearScanHistory();
    setScanCount(0);
    setCleared(true);
    setTimeout(() => setCleared(false), 2000);
  };

  return (
    <Layout title="Profile">
      <div className="space-y-5">

        {/* User card */}
        <div className="mint-card p-4 flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-emerald-100 border-2 border-emerald-200 flex items-center justify-center overflow-hidden flex-shrink-0">
            {image
              ? <img src={image} className="w-full h-full object-cover" alt="avatar" />
              : <span className="text-emerald-700 font-bold text-lg">{initials}</span>
            }
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-bold text-gray-900 text-base truncate">{name}</p>
            <p className="text-xs text-gray-400 mt-0.5 truncate">{email}</p>
            <span className="inline-flex items-center gap-1 mt-1.5 bg-emerald-50 border border-emerald-200 text-emerald-700 text-[10px] font-semibold px-2 py-0.5 rounded-full">
              <svg width="10" height="10" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Google account
            </span>
          </div>
        </div>

        {/* Preferences */}
        <div className="mint-card px-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider pt-3 pb-1">Preferences</p>
          <SettingRow
            icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#d97706" strokeWidth="2.2" strokeLinecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>}
            iconBg="bg-amber-50" label="Uncertainty warnings" sub="Flag high-uncertainty findings"
            right={<Toggle checked={uncertaintyWarn} onChange={setUncertaintyWarn} />}
          />
          <SettingRow
            icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>}
            iconBg="bg-emerald-50" label="Auto-generate report" sub="Groq LLaMA radiology report"
            right={<Toggle checked={autoReport} onChange={setAutoReport} />}
          />
          <SettingRow
            icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinecap="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>}
            iconBg="bg-emerald-50" label="Email notifications" sub="Report summaries (coming soon)"
            right={<Toggle checked={emailNotifs} onChange={setEmailNotifs} />} last
          />
        </div>

        {/* Data */}
        <div className="mint-card px-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider pt-3 pb-1">Data</p>
          <SettingRow
            icon={<Trash2 size={14} className="text-red-400" />}
            iconBg="bg-red-50" label="Clear scan history"
            sub={cleared ? "Cleared!" : `${scanCount} scan${scanCount !== 1 ? "s" : ""} stored locally`}
            onClick={scanCount > 0 ? handleClearHistory : undefined} last
          />
        </div>

        {/* About */}
        <div className="mint-card px-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider pt-3 pb-1">About</p>
          <SettingRow
            icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>}
            iconBg="bg-violet-50" label="Model version" sub="BioMedCLIP ViT-B/16 · v1.0.0" right={null}
          />
          <SettingRow
            icon={<Github size={14} className="text-gray-600" />}
            iconBg="bg-gray-100" label="GitHub repo" sub="Sowaiba-01/ThoraxNet"
            onClick={() => window.open("https://github.com/Sowaiba-01/ThoraxNet", "_blank")} last
          />
        </div>

        {/* Sign out */}
        <button
          onClick={() => signOut({ callbackUrl: "/login" })}
          className="w-full flex items-center justify-center gap-2 py-3 border border-red-200 text-red-500 rounded-2xl text-sm font-medium hover:bg-red-50 transition-colors"
        >
          <LogOut size={16} />
          Sign out
        </button>

        <p className="text-center text-xs text-gray-300 pb-2">
          ChestAI · BioMedCLIP · For research use only · Not FDA cleared
        </p>
      </div>
    </Layout>
  );
}
