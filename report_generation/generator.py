from __future__ import annotations

import os
from typing import Optional

from groq import Groq

from data.dataset import CLASSES


_SYSTEM_PROMPT = """\
You are a senior radiologist AI assistant. You receive structured data from an
automated chest X-ray analysis model and produce a concise, structured radiology
report. You must:
- Only report findings explicitly provided in the data.
- Never hallucinate or infer additional findings.
- Flag high-uncertainty predictions with "[UNCERTAIN - radiologist review advised]".
- Use standard radiological terminology.
- Be concise: findings should be 1-2 sentences per pathology.
- Always end with a clear RECOMMENDATION section.
Output format (strictly follow this):

FINDINGS:
[findings here]

IMPRESSION:
[1-3 sentence clinical summary]

RECOMMENDATION:
[urgency and next steps]
"""


class RadiologyReportGenerator:
    """
    Generates structured radiology reports from model predictions.

    Args:
        api_key: Groq API key. Defaults to GROQ_API_KEY env variable.
        model: Groq model ID (llama3-70b-8192 recommended).
        max_tokens: Maximum tokens for the generated report.
        temperature: Sampling temperature (low = more deterministic).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama3-70b-8192",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        probs: list[float],
        stds: list[float],
        threshold: float = 0.5,
        uncertainty_threshold: float = 0.15,
        patient_info: dict | None = None,
    ) -> str:
        """
        Generate a radiology report.

        Args:
            probs: List of 14 predicted probabilities (same order as CLASSES).
            stds: List of 14 uncertainty std values from MC Dropout.
            threshold: Probability above which a finding is considered present.
            uncertainty_threshold: Std above which a finding is flagged uncertain.
            patient_info: Optional dict with 'age' and 'gender' for context.

        Returns:
            Formatted radiology report as a string.
        """
        user_prompt = self._build_prompt(
            probs, stds, threshold, uncertainty_threshold, patient_info
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _build_prompt(
        self,
        probs: list[float],
        stds: list[float],
        threshold: float,
        uncertainty_threshold: float,
        patient_info: dict | None,
    ) -> str:
        lines = []

        if patient_info:
            age = patient_info.get("age", "Unknown")
            gender = "Male" if patient_info.get("gender") == "M" else "Female"
            lines.append(f"Patient: {gender}, Age {age}")
            lines.append("")

        lines.append("Automated model predictions (probability | uncertainty):")
        lines.append("")

        present, absent, uncertain_flags = [], [], []

        for cls, prob, std in zip(CLASSES, probs, stds):
            status = "PRESENT" if prob >= threshold else "absent"
            uncertain = std >= uncertainty_threshold
            uncertain_note = " [HIGH UNCERTAINTY]" if uncertain else ""

            lines.append(
                f"  {cls}: {status} | prob={prob:.3f} | uncertainty={std:.3f}{uncertain_note}"
            )

            if prob >= threshold:
                if uncertain:
                    uncertain_flags.append(cls)
                else:
                    present.append(cls)
            else:
                absent.append(cls)

        lines.append("")
        lines.append(f"Confirmed findings: {', '.join(present) if present else 'None'}")
        lines.append(f"Uncertain findings (flag for review): {', '.join(uncertain_flags) if uncertain_flags else 'None'}")
        lines.append(f"Absent: {', '.join(absent)}")
        lines.append("")
        lines.append("Please generate the structured radiology report.")

        return "\n".join(lines)


def generate_report_fallback(
    probs: list[float],
    stds: list[float],
    threshold: float = 0.5,
) -> str:
    """
    Rule-based report fallback when Groq API is unavailable.
    Produces a minimal but structured report without LLM.
    """
    present = [
        f"{cls} (p={prob:.2f})"
        for cls, prob in zip(CLASSES, probs)
        if prob >= threshold
    ]
    if not present:
        findings = "No significant pathology identified above the detection threshold."
        impression = "Chest X-ray within normal limits per automated analysis."
        recommendation = "Routine follow-up. Clinical correlation advised."
    else:
        findings = "The following findings were identified: " + "; ".join(present) + "."
        impression = f"Automated analysis suggests {len(present)} potential finding(s) requiring attention."
        recommendation = "Clinical correlation strongly advised. Radiologist review recommended."
    return (
        f"FINDINGS:\n{findings}\n\n"
        f"IMPRESSION:\n{impression}\n\n"
        f"RECOMMENDATION:\n{recommendation}\n\n"
        f"[Note: This report was generated by automated analysis. "
        f"It is not a substitute for radiologist interpretation.]"
    )
