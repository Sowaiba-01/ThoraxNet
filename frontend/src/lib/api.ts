const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://Sowaiba01-ThoraxNet.hf.space";

export interface FindingResult {
  name: string;
  probability: number;
  uncertainty: number;
  present: boolean;
  high_uncertainty: boolean;
}

/** Per-stage server-side latency breakdown, in milliseconds. */
export interface StageTimings {
  preprocess?: number;
  mc_dropout?: number;
  gradcam?: number;
  report?: number;
}

export interface PredictionResponse {
  findings: FindingResult[];
  /** null when the request was made with generateReport: false */
  report: string | null;
  entropy: number;
  inference_time_ms: number;
  stage_timings_ms?: StageTimings;
  model_version: string;
  gradcam_available: boolean;
  gradcam_classes: string[];
  /** Pass to gradcamUrl() to fetch heatmap overlays. */
  gradcam_session_id?: string | null;
}

export interface AnalyzeOptions {
  patientAge?: number;
  patientGender?: string;
  /** Skip the LLM narrative report (~1.3s faster). Default true. */
  generateReport?: boolean;
  /** Skip GradCAM heatmap generation. Default true. */
  generateGradcam?: boolean;
}

export async function analyzeXray(
  file: File,
  patientAge?: number,
  patientGender?: string,
  options: AnalyzeOptions = {}
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  const age = options.patientAge ?? patientAge;
  const gender = options.patientGender ?? patientGender;
  if (age !== undefined) form.append("patient_age", String(age));
  if (gender) form.append("patient_gender", gender);

  form.append("generate_report", String(options.generateReport ?? true));
  form.append("generate_gradcam", String(options.generateGradcam ?? true));

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
  stageTimings?: StageTimings;
  modelVersion?: string;
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
    stageTimings: result.stage_timings_ms,
    modelVersion: result.model_version,
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
