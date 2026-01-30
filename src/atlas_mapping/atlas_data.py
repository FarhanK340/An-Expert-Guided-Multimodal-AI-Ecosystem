"""
Atlas data management and region name lookups.
"""

import os
from pathlib import Path
from nilearn import datasets


def get_region_names(atlas_name='harvard_oxford'):
    """
    Load region names from atlas lookup table.
    
    Args:
        atlas_name: Atlas identifier ('harvard_oxford', 'AAL3', etc.)
    
    Returns:
        Dictionary mapping region IDs to anatomical names
    """
    if atlas_name == 'harvard_oxford':
        return get_harvard_oxford_names()
    elif atlas_name == 'AAL3':
        return get_aal3_names()
    else:
        raise ValueError(f"Atlas {atlas_name} not supported")


def get_harvard_oxford_names():
    """
    Harvard-Oxford cortical and subcortical atlas region names.
    
    Returns complete mapping of region IDs to anatomical names.
    """
    # Cortical regions
    cortical_regions = {
        0: 'Background',
        1: 'Frontal Pole',
        2: 'Insular Cortex',
        3: 'Superior Frontal Gyrus',
        4: 'Middle Frontal Gyrus',
        5: 'Inferior Frontal Gyrus, pars triangularis',
        6: 'Inferior Frontal Gyrus, pars opercularis',
        7: 'Precentral Gyrus',
        8: 'Temporal Pole',
        9: 'Superior Temporal Gyrus, anterior',
        10: 'Superior Temporal Gyrus, posterior',
        11: 'Middle Temporal Gyrus, anterior',
        12: 'Middle Temporal Gyrus, posterior',
        13: 'Middle Temporal Gyrus, temporooccipital',
        14: 'Inferior Temporal Gyrus, anterior',
        15: 'Inferior Temporal Gyrus, posterior',
        16: 'Inferior Temporal Gyrus, temporooccipital',
        17: 'Postcentral Gyrus',
        18: 'Superior Parietal Lobule',
        19: 'Supramarginal Gyrus, anterior',
        20: 'Supramarginal Gyrus, posterior',
        21: 'Angular Gyrus',
        22: 'Lateral Occipital Cortex, superior',
        23: 'Lateral Occipital Cortex, inferior',
        24: 'Intracalcarine Cortex',
        25: 'Frontal Medial Cortex',
        26: 'Juxtapositional Lobule Cortex',
        27: 'Subcallosal Cortex',
        28: 'Paracingulate Gyrus',
        29: 'Cingulate Gyrus, anterior',
        30: 'Cingulate Gyrus, posterior',
        31: 'Precuneous Cortex',
        32: 'Cuneal Cortex',
        33: 'Frontal Orbital Cortex',
        34: 'Parahippocampal Gyrus, anterior',
        35: 'Parahippocampal Gyrus, posterior',
        36: 'Lingual Gyrus',
        37: 'Temporal Fusiform Cortex, anterior',
        38: 'Temporal Fusiform Cortex, posterior',
        39: 'Temporal Occipital Fusiform Cortex',
        40: 'Occipital Fusiform Gyrus',
        41: 'Frontal Operculum Cortex',
        42: 'Central Opercular Cortex',
        43: 'Parietal Operculum Cortex',
        44: 'Planum Polare',
        45: 'Heschl\'s Gyrus',
        46: 'Planum Temporale',
        47: 'Supracalcarine Cortex',
        48: 'Occipital Pole',
    }
    
    # Subcortical regions (offset by 100 to avoid conflicts)
    subcortical_regions = {
        100: 'Left Cerebral White Matter',
        101: 'Left Cerebral Cortex',
        102: 'Left Lateral Ventricle',
        103: 'Left Thalamus',
        104: 'Left Caudate',
        105: 'Left Putamen',
        106: 'Left Pallidum',
        107: 'Brain-Stem',
        108: 'Left Hippocampus',
        109: 'Left Amygdala',
        110: 'Left Accumbens',
        111: 'Right Cerebral White Matter',
        112: 'Right Cerebral Cortex',
        113: 'Right Lateral Ventricle',
        114: 'Right Thalamus',
        115: 'Right Caudate',
        116: 'Right Putamen',
        117: 'Right Pallidum',
        118: 'Right Hippocampus',
        119: 'Right Amygdala',
        120: 'Right Accumbens',
    }
    
    # Combine both
    all_regions = {**cortical_regions, **subcortical_regions}
    return all_regions


def get_aal3_names():
    """
    AAL3 (Automated Anatomical Labeling) atlas region names.
    
    Returns:
        Dictionary of AAL3 region IDs to names
    """
    # Placeholder - add full AAL3 mapping if needed
    return {
        0: 'Background',
        1: 'Precentral_L',
        2: 'Precentral_R',
        3: 'Frontal_Sup_2_L',
        4: 'Frontal_Sup_2_R',
        # ... Add all 170 AAL3 regions as needed
    }


def download_harvard_oxford_atlas(output_dir='./atlases'):
    """
    Download Harvard-Oxford atlas using nilearn.
    
    Args:
        output_dir: Directory to save atlas files
    
    Returns:
        Dictionary with paths to atlas files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading Harvard-Oxford cortical atlas...")
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    
    print("Downloading Harvard-Oxford subcortical atlas...")
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
    
    # nilearn returns a Bunch object with 'filename' attribute for the actual file path
    # The 'maps' attribute might be a loaded image, so we use 'filename' instead
    cortical_path = ho_cort.filename if hasattr(ho_cort, 'filename') else ho_cort.maps
    subcortical_path = ho_sub.filename if hasattr(ho_sub, 'filename') else ho_sub.maps
    
    # If maps is a string, use it directly; if it's an image with file_map, extract path
    if not isinstance(cortical_path, str):
        if hasattr(cortical_path, 'file_map') and 'image' in cortical_path.file_map:
            cortical_path = cortical_path.file_map['image'].filename
    
    if not isinstance(subcortical_path, str):
        if hasattr(subcortical_path, 'file_map') and 'image' in subcortical_path.file_map:
            subcortical_path = subcortical_path.file_map['image'].filename
    
    return {
        'cortical_atlas': cortical_path,
        'subcortical_atlas': subcortical_path,
        'cortical_labels': ho_cort.labels,
        'subcortical_labels': ho_sub.labels,
    }


def get_mni152_template():
    """
    Download MNI152 template for registration.
    
    Returns:
        Path to MNI152 T1 template (as string)
    """
    print("Downloading MNI152 template...")
    mni = datasets.fetch_icbm152_2009()
    
    # Extract file path - mni.t1 might be a string or an image object
    template_path = mni.t1
    
    if not isinstance(template_path, str):
        if hasattr(template_path, 'file_map') and 'image' in template_path.file_map:
            template_path = template_path.file_map['image'].filename
    
    return template_path
