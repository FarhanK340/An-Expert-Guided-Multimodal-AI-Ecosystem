"""
LLM utilities for report generation.

Pipeline:
    segmentation nifti → TumorDescriptorGenerator → JSON descriptor
    JSON descriptor → generate_report_from_descriptor() → report text (str)

Uses Med Gemma 4B (via local/HF endpoint) as primary LLM with a rule-based template fallback.
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, Optional
from django.conf import settings

# Add 'src' to path so we can import the native modules
src_path = str(Path(settings.BASE_DIR).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from atlas_mapping.atlas_mapper import BrainAtlasMapper
from json_generation.descriptor_generator import TumorDescriptorGenerator


# ---------------------------------------------------------------------------
# Step 1 – Convert raw segmentation findings into a clean JSON descriptor
# ---------------------------------------------------------------------------

def generate_json_descriptor(
    structured_findings: Dict,
    patient_info: Optional[Dict] = None,
    case_id: str = "Unknown"
) -> Dict:
    """
    Uses the native BrainAtlasMapper and TumorDescriptorGenerator to build
    a deeply detailed, atlas-mapped clinical JSON descriptor.
    """
    # 1. Grab the full_segmentation_mask from findings
    seg_rel_path = structured_findings.get('full_segmentation_mask')
    if not seg_rel_path:
        raise ValueError("No full_segmentation_mask found in structured_findings.")
        
    abs_seg_path = Path(settings.MEDIA_ROOT) / seg_rel_path
    if not abs_seg_path.exists():
        raise FileNotFoundError(f"Segmentation mask not found at {abs_seg_path}")
        
    # 2. Extract patient metadata for the generator
    patient_metadata = {}
    if patient_info:
        if 'age' in patient_info:
            patient_metadata['age'] = patient_info['age']
        if 'sex' in patient_info:
            patient_metadata['sex'] = patient_info['sex']

    # 3. Instantiate mapper and generate descriptor
    try:
        mapper = BrainAtlasMapper(atlas_name='harvard_oxford', use_ants=False)
        generator = TumorDescriptorGenerator(mapper)
        
        descriptor = generator.generate_descriptor(
            case_id=case_id,
            seg_path=str(abs_seg_path),
            patient_metadata=patient_metadata,
            model_name=structured_findings.get('model_version', 'MoME+ v1.0')
        )
        return descriptor
    except Exception as e:
        print(f"Failed to generate atlas-mapped JSON: {e}")
        # Fallback to empty/basic dictionary so template engine doesn't crash completely
        return {"patient_info": patient_info, "error": str(e)}


# ---------------------------------------------------------------------------
# Step 2 – Generate report text from descriptor (LLM or fallback)
# ---------------------------------------------------------------------------

def generate_report_from_descriptor(json_descriptor: Dict) -> str:
    """
    Calls Med Gemma 4B api (local Ollama or HuggingFace endpoint). 
    Falls back to a template-based report on failure.
    """
    # Use HF token if calling HuggingFace Inference Endpoint, or use local Ollama URL
    hf_api_key = os.environ.get('HF_API_KEY', '')
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
    
    # We will try Med Gemma 4B via Ollama first, then HF, then fallback
    try:
        return _call_med_gemma(json_descriptor, ollama_url, hf_api_key)
    except Exception as e:
        print(f"[LLM] Med Gemma 4B failed or not reachable: {e}. Using deterministic template fallback.")
        return _template_report(json_descriptor)


def _call_med_gemma(descriptor: Dict, ollama_url: str, hf_api_key: str) -> str:
    """Generate a cohesive clinical report using Med Gemma 4B."""
    
    prompt = f"""You are an expert clinical neuroradiologist. Write a highly professional, formal radiology report based on the following AI-generated Brain Tumor Atlas descriptor.
    
ATLAS DESCRIPTOR DATA:
{json.dumps(descriptor, indent=2)}

Format the report exactly as follows with no extra conversational text:
**CLINICAL INFORMATION**
(Patient details, scan info)

**TECHNIQUE**
(Model info, scan type)

**FINDINGS**
(Detailed volumetric breakdown, anatomic regions affected, hemisphere, midline crossing)

**IMPRESSION**
(Summary of findings)

**RECOMMENDATION**
(Clinical next steps)
"""
    
    # Attempt local Ollama (assuming user has 'medgemma4b' or similar model pulled)
    try:
        payload = {
            "model": "medgemma",  # Assuming they named the ollama model 'medgemma' or 'gemma:7b'
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(ollama_url, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json().get('response', '')
    except requests.exceptions.RequestException:
        pass # Ollama not running
        
    # If Ollama not running but HF key is present, try Hugging Face Inference API
    if hf_api_key:
        hf_url = "https://api-inference.huggingface.co/models/google/gemma-7b-it" # Fallback to standard gemma if medgemma not hosted
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        hf_payload = {"inputs": prompt}
        hf_resp = requests.post(hf_url, headers=headers, json=hf_payload, timeout=30)
        if hf_resp.status_code == 200:
            result = hf_resp.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').replace(prompt, '').strip()

    raise ConnectionError("No Med Gemma 4B endpoint could be reached.")


def _template_report(descriptor: Dict) -> str:
    """
    Rule-based template report — used when LLM is unavailable.
    """
    pi = descriptor.get('patient_info', {})
    seg_res = descriptor.get('segmentation_results', {})
    vol_analysis = seg_res.get('volumetric_analysis', {})
    anat = descriptor.get('anatomical_mapping', {})

    patient_id = pi.get('case_id', 'Unknown')
    age = pi.get('age', 'Unknown')
    sex = pi.get('sex', 'Unknown')
    sex_str = {'M': 'male', 'F': 'female'}.get(str(sex), 'unknown sex')
    scan_date = pi.get('scan_date', 'Unknown')

    wt = vol_analysis.get('total_tumor_volume_mm3', 0)
    tc = vol_analysis.get('tumor_core_volume_mm3', 0)
    et = vol_analysis.get('enhancing_volume_mm3', 0)
    
    hemi = anat.get('hemisphere', 'Unknown')
    midline = 'Yes' if anat.get('crossing_midline') else 'No'

    regions = anat.get('affected_regions', [])
    regions_str = ", ".join([r['region_name'] for r in regions[:3]]) if regions else "None explicitly mapped"

    report = f"""BRAIN MRI – SEGMENTATION ANALYSIS REPORT

**CLINICAL INFORMATION**
Patient ID: {patient_id}
Age: {age} | Sex: {sex_str.capitalize()}
Date of Scan: {scan_date}

**TECHNIQUE**
Automated AI-assisted segmentation mapped against {anat.get('atlas_name', 'Brain Atlas')}.

**FINDINGS**
Tumor localized primarily in the {hemi} hemisphere. 
Crossing midline: {midline}.
Primary affected regions: {regions_str}.

Volumetric Analysis:
• Whole Tumor (WT): {wt:.1f} mm³
• Tumor Core (TC): {tc:.1f} mm³
• Enhancing Tumor (ET): {et:.1f} mm³

{'An actively enhancing tumor component is present, suggesting high vascularity.' if et > 100 else 'No significant enhancing component is identified.'}

**IMPRESSION**
Abnormal mass in the {hemi} hemisphere {f'involving {regions_str}' if regions else ''}.

**RECOMMENDATION**
Recommend further imaging correlation and multidisciplinary review.

---
This report was generated by the Expert-Guided Multimodal AI Ecosystem and is intended for research and decision-support purposes only. All findings must be reviewed and confirmed by a qualified radiologist or clinician before clinical action.
"""
    return report
