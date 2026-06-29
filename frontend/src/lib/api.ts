/**
 * API client for ChestAI backend.
 * Uses axios with typed responses matching the FastAPI Pydantic schemas.
 */

import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

export interface FindingResult {
  name: string;
  probability: number;
  uncertainty: number;
  present: boolean;
  high_uncertainty: boolean;
}

export interface PredictionResponse {
  findings: FindingResult[];
  entropy: number;
  report: string;
  gradcam_available: boolean;
  gradcam_classes: string[];
  inference_time_ms: number;
  model_version: string;
}

export async function analyzeXray(
  file: File,
  patientAge?: number,
  patientGender?: string
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (patientAge !== undefined) formData.append("patient_age", String(patientAge));
  if (patientGender) formData.append("patient_gender", patientGender);

  const { data } = await axios.post<PredictionResponse>(
    `${API_URL}/api/v1/predict`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" }, timeout: 60000 }
  );
  return data;
}

export function getGradCAMUrl(sessionId: string, className: string): string {
  return `${API_URL}/api/v1/gradcam/${sessionId}/${encodeURIComponent(className)}`;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const { data } = await axios.get(`${API_URL}/health`, { timeout: 5000 });
    return data.model_loaded === true;
  } catch {
    return false;
  }
}
