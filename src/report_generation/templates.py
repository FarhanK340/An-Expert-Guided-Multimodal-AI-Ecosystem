"""
Clinical report templates for baseline generation and synthetic data creation.
"""

import random
from typing import Dict, Any

# Template variations for report generation
REPORT_TEMPLATES = [
    # Template 1: Standard clinical format
    """CLINICAL HISTORY: {clinical_history}

TECHNIQUE: Multi-sequence MRI brain examination including {modalities}.

FINDINGS:
{findings}

IMPRESSION:
{impression}""",
    
    # Template 2: Detailed structured format
    """MRI BRAIN WITH AND WITHOUT CONTRAST

CLINICAL INDICATION: {clinical_history}

IMAGING TECHNIQUE: {modalities} sequences acquired.

FINDINGS:
{findings}

IMPRESSION:
{impression}""",
    
    # Template 3: Concise format
    """EXAMINATION: MRI Brain
INDICATION: {clinical_history}

FINDINGS:
{findings}

IMPRESSION:
{impression}""",
    
    # Template 4: Comprehensive format
    """BRAIN MRI REPORT

Clinical History: {clinical_history}

Technique: Multiparametric MRI including {modalities} sequences.

Findings:
{findings}

Impression:
{impression}""",
    
    # Template 5: Academic format
    """MRI BRAIN EXAMINATION

HISTORY: {clinical_history}

PROTOCOL: {modalities}

FINDINGS:
{findings}

CONCLUSION:
{impression}"""
]


def generate_findings_from_json(json_data: Dict[str, Any], template_variation: int = 0) -> str:
    """
    Generate Findings section from JSON anatomical descriptors.
    
    Args:
        json_data: Structured JSON with segmentation results and anatomical mapping
        template_variation: Which variation of phrasing to use (0-4)
    
    Returns:
        Formatted findings text
    """
    findings_parts = []
    
    # Get segmentation results
    seg_results = json_data.get('segmentation_results', {})
    anatomical = json_data.get('anatomical_mapping', {})
    
    # Variations for different components
    tumor_descriptions = [
        "A {type} tumor measuring {volume} cm³ is identified",
        "There is a {type} lesion measuring approximately {volume} cm³",
        "{type} mass lesion identified, measuring {volume} cm³",
        "An abnormal {type} mass measuring {volume} cm³ is present",
        "Focal {type} lesion measuring {volume} cm³"
    ]
    
    location_descriptions = [
        "centered in the {location}",
        "primarily involving the {location}",
        "located in the {location}",
        "within the {location}",
        "affecting the {location}"
    ]
    
    involvement_descriptions = [
        "with {overlap}% regional involvement",
        "demonstrating {overlap}% involvement of the region",
        "occupying approximately {overlap}% of the {location}",
        "with significant involvement ({overlap}%)",
        "involving {overlap}% of the {location}"
    ]
    
    # Use template variation to select phrasing
    var_idx = template_variation % 5
    
    # Enhancing Tumor
    if 'enhancing_tumor' in seg_results:
        et = seg_results['enhancing_tumor']
        volume = et.get('volume_cm3', 0)
        if volume > 0.1:  # Only report if significant
            affected_regions = anatomical.get('enhancing_tumor_regions', [])
            if affected_regions:
                top_region = affected_regions[0]
                overlap_pct = top_region['overlap_percent']
                region_name = top_region['name']
                findings_parts.append(
                    f"{tumor_descriptions[var_idx].format(type='enhancing', volume=f'{volume:.1f}')} "
                    f"{location_descriptions[var_idx].format(location=region_name)} "
                    f"{involvement_descriptions[var_idx].format(overlap=f'{overlap_pct:.1f}', location=region_name)}."
                )
    
    # Tumor Core
    if 'tumor_core' in seg_results:
        tc = seg_results['tumor_core']
        volume = tc.get('volume_cm3', 0)
        if volume > 0.1:
            findings_parts.append(
                f"The tumor core measures {volume:.1f} cm³."
            )
    
    # Peritumoral Edema
    if 'whole_tumor' in seg_results and 'tumor_core' in seg_results:
        wt_vol = seg_results['whole_tumor'].get('volume_cm3', 0)
        tc_vol = seg_results['tumor_core'].get('volume_cm3', 0)
        edema_vol = wt_vol - tc_vol
        if edema_vol > 0.1:
            edema_descriptions = [
                f"Peritumoral edema extending into adjacent white matter measuring approximately {edema_vol:.1f} cm³.",
                f"Surrounding vasogenic edema measures {edema_vol:.1f} cm³.",
                f"Associated peritumoral edema ({edema_vol:.1f} cm³) is present.",
                f"There is vasogenic edema measuring {edema_vol:.1f} cm³.",
                f"Perilesional edema totaling {edema_vol:.1f} cm³ is observed."
            ]
            findings_parts.append(edema_descriptions[var_idx])
    
    # Laterality
    laterality = anatomical.get('laterality', 'bilateral')
    if laterality != 'bilateral':
        lat_descriptions = [
            f"The lesion is {laterality} hemispheric in distribution.",
            f"{laterality.capitalize()} hemispheric predominance.",
            f"Predominantly {laterality}-sided distribution.",
            f"The tumor is {laterality} hemispheric.",
            f"{laterality.capitalize()}-sided hemispheric involvement."
        ]
        findings_parts.append(lat_descriptions[var_idx])
    else:
        findings_parts.append("Bilateral hemispheric involvement is noted.")
    
    # Mass effect (if available)
    if json_data.get('mass_effect'):
        findings_parts.append("There is associated mass effect with midline shift.")
    else:
        findings_parts.append("No significant midline shift is observed.")
    
    return "\n".join(findings_parts) if findings_parts else "No significant abnormality detected."


def generate_impression_from_json(json_data: Dict[str, Any], template_variation: int = 0) -> str:
    """
    Generate Impression section from JSON data.
    
    Args:
        json_data: Structured JSON with segmentation results
        template_variation: Which variation to use
    
    Returns:
        Formatted impression text
    """
    seg_results = json_data.get('segmentation_results', {})
    
    # Check for tumor presence
    has_tumor = any(
        seg_results.get(key, {}).get('volume_cm3', 0) > 0.1 
        for key in ['enhancing_tumor', 'tumor_core', 'whole_tumor']
    )
    
    if not has_tumor:
        return "No evidence of intracranial mass lesion."
    
    # Impression variations
    impressions = [
        "The imaging findings are consistent with a primary brain neoplasm, likely high-grade glioma given the enhancement pattern and degree of edema. Clinical correlation and tissue diagnosis recommended.",
        
        "Findings suggestive of high-grade glial neoplasm. Recommend clinical correlation, possible biopsy, and neurosurgical consultation.",
        
        "Brain mass with enhancement and peritumoral edema, most consistent with high-grade glioma. Further evaluation with tissue sampling is recommended.",
        
        "Primary brain tumor with imaging characteristics consistent with high-grade glioma. Correlation with clinical presentation and consideration of tissue diagnosis is advised.",
        
        "Enhancing intra-axial mass lesion favoring high-grade glioma. Clinical correlation and histopathological confirmation recommended."
    ]
    
    var_idx = template_variation % 5
    return impressions[var_idx]


def generate_template_report(json_data: Dict[str, Any], template_idx: int = 0) -> str:
    """
    Generate a complete report using deterministic templates.
    
    Args:
        json_data: Structured JSON anatomical descriptors
        template_idx: Which template to use (0-4)
    
    Returns:
        Complete formatted report
    """
    # Use same variation for consistency
    variation = template_idx % 5
    
    # Generate sections
    clinical_history = json_data.get('clinical_history', 'Brain tumor evaluation')
    modalities = json_data.get('imaging_parameters', {}).get('modalities', 'T1, T1ce, T2, FLAIR')
    
    if isinstance(modalities, list):
        modalities = ', '.join(modalities)
    
    findings = generate_findings_from_json(json_data, variation)
    impression = generate_impression_from_json(json_data, variation)
    
    # Fill template
    template = REPORT_TEMPLATES[variation]
    report = template.format(
        clinical_history=clinical_history,
        modalities=modalities,
        findings=findings,
        impression=impression
    )
    
    return report


def generate_random_report(json_data: Dict[str, Any]) -> str:
    """Generate report with random template variation."""
    template_idx = random.randint(0, 4)
    return generate_template_report(json_data, template_idx)
