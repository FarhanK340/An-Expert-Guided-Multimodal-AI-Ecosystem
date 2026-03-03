"""
LLM utilities for report generation.

Pipeline:
    structured_findings (dict) → generate_json_descriptor() → JSON descriptor (dict)
    JSON descriptor → generate_report_from_descriptor() → report text (str)

Uses Google Gemini as primary LLM with a rule-based template fallback.
"""

import os
import json
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Step 1 – Convert raw segmentation findings into a clean JSON descriptor
# ---------------------------------------------------------------------------

def generate_json_descriptor(
    structured_findings: Dict,
    patient_info: Optional[Dict] = None
) -> Dict:
    """
    Converts the SegmentationResult.structured_findings into a
    structured medical JSON descriptor suitable for LLM prompting.
    """
    volumes = structured_findings.get('volumes', {})
    confidence = structured_findings.get('confidence_scores', {})
    timestamp = structured_findings.get('timestamp', '')
    model_version = structured_findings.get('model_version', 'MoME+ v1.0')

    # Grade tumor burden (rough clinical heuristic for demonstration)
    wt_vol = volumes.get('whole_tumor', 0) or 0
    tc_vol = volumes.get('tumor_core', 0) or 0
    et_vol = volumes.get('enhancing_tumor', 0) or 0

    burden = 'minimal'
    if wt_vol > 50000:
        burden = 'extensive'
    elif wt_vol > 20000:
        burden = 'moderate'
    elif wt_vol > 5000:
        burden = 'mild'

    descriptor = {
        "patient_info": patient_info or {},
        "scan_info": {
            "analysis_timestamp": timestamp,
            "model_version": model_version,
        },
        "tumor_metrics": {
            "volumes": {
                "whole_tumor": round(float(wt_vol), 2),
                "tumor_core": round(float(tc_vol), 2),
                "enhancing_tumor": round(float(et_vol), 2),
            },
            "confidence_scores": {
                "whole_tumor": round(float(confidence.get('whole_tumor', 0)), 3),
                "tumor_core": round(float(confidence.get('tumor_core', 0)), 3),
                "enhancing_tumor": round(float(confidence.get('enhancing_tumor', 0)), 3),
            },
            "tumor_burden": burden,
            "enhancing_component_present": et_vol > 100,
        },
        "clinical_interpretation": {
            "primary_finding": _classify_tumor(wt_vol, tc_vol, et_vol),
            "recommendation": _recommend(wt_vol, et_vol),
        },
    }

    return descriptor


def _classify_tumor(wt: float, tc: float, et: float) -> str:
    if wt < 100:
        return "No significant tumor region detected"
    if et > 500:
        return "High-grade glioma pattern with active enhancing component"
    if tc > 1000:
        return "Necrotic/cystic tumor core identified"
    return "Non-enhancing tumor region identified"


def _recommend(wt: float, et: float) -> str:
    if wt < 100:
        return "Continue routine monitoring; no immediate intervention indicated"
    if et > 500:
        return "Recommend neurosurgery and neuro-oncology consultation; consider biopsy or resection"
    return "Recommend further imaging correlation and multidisciplinary review"


# ---------------------------------------------------------------------------
# Step 2 – Generate report text from descriptor (LLM or fallback)
# ---------------------------------------------------------------------------

def generate_report_from_descriptor(json_descriptor: Dict) -> str:
    """
    Calls Google Gemini API. Falls back to a template-based report on failure.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '')

    if api_key:
        try:
            return _call_gemini(json_descriptor, api_key)
        except Exception as e:
            print(f"[LLM] Gemini API failed: {e}. Using template fallback.")

    return _template_report(json_descriptor)


def _call_gemini(descriptor: Dict, api_key: str) -> str:
    """Call Google Gemini 1.5 Flash to generate the radiology report."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    prompt = f"""You are an expert neuroradiologist. Generate a formal, structured brain MRI radiology report based on the following AI segmentation analysis data. Use clear clinical language and section headers.

Segmentation Analysis Data:
{json.dumps(descriptor, indent=2)}

Format the report with these sections:
1. CLINICAL INFORMATION
2. TECHNIQUE
3. FINDINGS
4. IMPRESSION
5. RECOMMENDATION

Be specific about tumor volumes and their clinical significance. Note the confidence scores of the AI predictions.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text


def _template_report(descriptor: Dict) -> str:
    """
    Rule-based template report — used when LLM is unavailable.
    """
    pi = descriptor.get('patient_info', {})
    metrics = descriptor.get('tumor_metrics', {})
    volumes = metrics.get('volumes', {})
    confidence = metrics.get('confidence_scores', {})
    interp = descriptor.get('clinical_interpretation', {})
    scan_info = descriptor.get('scan_info', {})

    patient_id = pi.get('patient_id', 'Unknown')
    age = pi.get('age', 'Unknown')
    sex = pi.get('sex', 'Unknown')
    sex_str = {'M': 'male', 'F': 'female'}.get(str(sex), 'unknown sex')
    clinical_history = pi.get('clinical_history', 'Not provided')
    scan_date = pi.get('scan_date', 'Unknown')

    wt = volumes.get('whole_tumor', 0)
    tc = volumes.get('tumor_core', 0)
    et = volumes.get('enhancing_tumor', 0)
    wt_conf = confidence.get('whole_tumor', 0)
    tc_conf = confidence.get('tumor_core', 0)
    et_conf = confidence.get('enhancing_tumor', 0)

    burden = metrics.get('tumor_burden', 'unknown')
    primary = interp.get('primary_finding', 'N/A')
    recommendation = interp.get('recommendation', 'N/A')
    model_ver = scan_info.get('model_version', 'MoME+ v1.0')

    report = f"""BRAIN MRI – SEGMENTATION ANALYSIS REPORT

**CLINICAL INFORMATION**
Patient ID: {patient_id}
Age: {age} | Sex: {sex_str.capitalize()}
Clinical History: {clinical_history}
Date of Scan: {scan_date}

**TECHNIQUE**
Multi-modal MRI brain was performed including T1-weighted, T1 contrast-enhanced (T1ce), T2-weighted, and FLAIR sequences. Automated AI-assisted segmentation was performed using {model_ver}.

**FINDINGS**
AI-assisted segmentation identified the following tumor sub-regions:

• Whole Tumor (WT): {wt:.1f} mm³  (confidence: {wt_conf:.1%})
• Tumor Core (TC): {tc:.1f} mm³  (confidence: {tc_conf:.1%})
• Enhancing Tumor (ET): {et:.1f} mm³  (confidence: {et_conf:.1%})

Overall tumor burden is classified as {burden}.

{'An actively enhancing tumor component is present, suggesting high vascularity and possible high-grade features.' if et > 100 else 'No significant enhancing component is identified.'}

**IMPRESSION**
{primary}.

**RECOMMENDATION**
{recommendation}.

---
This report was generated by the Expert-Guided Multimodal AI Ecosystem and is intended for research and decision-support purposes only. All findings must be reviewed and confirmed by a qualified radiologist or clinician before clinical action.
"""
    return report
