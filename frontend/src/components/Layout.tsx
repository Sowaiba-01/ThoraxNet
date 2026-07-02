import { useRouter } from "next/router";
import Link from "next/link";
import { ReactNode } from "react";
import { Activity } from "lucide-react";

interface LayoutProps {
  children: ReactNode;
  title?: string;
}

const NAV_ITEMS = [
  {
    href: "/",
    label: "Home",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
        <polyline points="9 22 9 12 15 12 15 22" />
      </svg>
    ),
  },
  {
    href: "/scan",
    label: "Scan",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="12" cy="12" r="3" />
        <path d="M3 9h2M3 15h2M19 9h2M19 15h2M9 3v2M15 3v2M9 19v2M15 19v2" />
      </svg>
    ),
  },
  {
    href: "/stats",
    label: "Stats",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    ),
  },
  {
    href: "/profile",
    label: "Profile",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="8" r="4" />
        <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
      </svg>
    ),
  },
];

export default function Layout({ children, title }: LayoutProps) {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-emerald-100 sticky top-0 z-50">
        <div className="max-w-lg mx-auto px-4 py-3 flex items-center gap-2.5">
          <div className="w-8 h-8 bg-emerald-500 rounded-xl flex items-center justify-center">
            <Activity size={16} className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-gray-900 text-sm leading-tight">ChestAI</h1>
            <p className="text-[9px] text-emerald-600 font-semibold tracking-widest">X-RAY DIAGNOSTIC PLATFORM</p>
          </div>
          {title && (
            <span className="ml-auto text-sm font-semibold text-gray-700">{title}</span>
          )}
        </div>
      </header>

      {/* Page content */}
      <main className="flex-1 max-w-lg mx-auto w-full px-4 py-5 pb-24">
        {children}
      </main>

      {/* Bottom nav */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-emerald-100 z-50">
        <div className="max-w-lg mx-auto flex">
          {NAV_ITEMS.map(({ href, label, icon }) => {
            const active = router.pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex-1 flex flex-col items-center gap-1 py-2.5 transition-colors ${
                  active ? "text-emerald-500" : "text-gray-400 hover:text-gray-600"
                }`}
              >
                {icon}
                <span className="text-[10px] font-medium">{label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
