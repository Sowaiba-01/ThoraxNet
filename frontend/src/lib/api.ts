const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://Sowaiba01-chestai-api.hf.space";

export interface FindingResult {
  name: string;
  probability: number;
  uncertainty: number;
  present: boolean;
  high_uncertainty: boolean;
}

export interface PredictionResponse {
  findings: FindingResult[];
  report: string;
  entropy: number;
  inference_time_ms: number;
  model_version: string;
  gradcam_available: boolean;
  gradcam_classes: string[];
}

export async function analyzeXray(
  file: File,
  patientAge?: number,
  patientGender?: string
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  if (patientAge !== undefined) form.append("patient_age", String(patientAge));
  if (patientGender) form.append("patient_gender", patientGender);

  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

export function gradcamUrl(sessionId: string, className: string): string {
  return `${API_BASE}/api/v1/gradcam/${sessionId}/${encodeURIComponent(className)}`;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    return data.model_loaded === true;
  } catch {
    return false;
  }
}

// ── Scan history (localStorage) ──────────────────────────────────────────────

export interface ScanRecord {
  id: string;
  timestamp: number;
  findings: FindingResult[];
  inferenceMs: number;
  patientAge?: number;
  patientGender?: string;
}

const STORAGE_KEY = "chestai_scan_history";
const MAX_HISTORY = 50;

export function saveScan(
  result: PredictionResponse,
  patientAge?: number,
  patientGender?: string
): ScanRecord {
  const record: ScanRecord = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    findings: result.findings,
    inferenceMs: result.inference_time_ms,
    patientAge,
    patientGender,
  };
  const history = getScanHistory();
  const updated = [record, ...history].slice(0, MAX_HISTORY);
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  }
  return record;
}

export function getScanHistory(): ScanRecord[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as ScanRecord[];
  } catch {
    return [];
  }
}

export function clearScanHistory(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
  }
}
