import { useEffect } from "react";
import { signIn, useSession } from "next-auth/react";
import { useRouter } from "next/router";
import Head from "next/head";
import { Activity, Loader2 } from "lucide-react";

export default function LoginPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "authenticated") {
      router.replace("/");
    }
  }, [status, router]);

  if (status === "loading" || status === "authenticated") {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <Loader2 className="text-emerald-500 animate-spin" size={28} />
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Sign in — ChestAI</title>
      </Head>
      <div className="min-h-screen bg-white flex flex-col items-center justify-center px-6">
        <div className="w-full max-w-sm flex flex-col items-center gap-6">

          {/* Logo */}
          <div className="flex flex-col items-center gap-3">
            <div className="w-16 h-16 bg-emerald-500 rounded-2xl flex items-center justify-center shadow-mint-md">
              <Activity size={30} className="text-white" />
            </div>
            <div className="text-center">
              <h1 className="text-2xl font-bold text-gray-900 tracking-tight">ChestAI</h1>
              <p className="text-[10px] text-emerald-600 font-semibold tracking-widest mt-0.5">
                X-RAY DIAGNOSTIC PLATFORM
              </p>
            </div>
          </div>

          {/* Badge */}
          <div className="flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium px-3 py-1.5 rounded-full">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
            BioMedCLIP · NIH ChestX-ray14 · AUC 0.82
          </div>

          {/* Tagline */}
          <div className="text-center space-y-1">
            <p className="text-base font-semibold text-gray-900">AI-powered chest X-ray analysis</p>
            <p className="text-sm text-gray-400 leading-relaxed">
              Detects 14 pathologies with uncertainty quantification and auto-generated radiology reports
            </p>
          </div>

          {/* Sign in button */}
          <div className="w-full">
            <button
              onClick={() => signIn("google", { callbackUrl: "/" })}
              className="w-full flex items-center justify-center gap-3 rounded-2xl px-4 py-3.5 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-all"
              style={{ border: "1.5px solid #e5e7eb" }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </button>
          </div>

          <p className="text-xs text-gray-300 text-center">
            For research use only · Not FDA cleared
          </p>
        </div>
      </div>
    </>
  );
}
