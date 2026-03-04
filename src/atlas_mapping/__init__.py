"""
Brain Atlas Mapping Module

This module provides functionality for mapping brain tumor segmentations
to anatomical regions using standard brain atlases (Harvard-Oxford, AAL3, etc.)
"""

from .atlas_mapper import BrainAtlasMapper
from .registration import register_to_atlas_affine, register_to_atlas_ants
from .region_analysis import (
    compute_region_overlap,
    calculate_percentage_involvement,
    analyze_tumor_subregions
)
from .atlas_data import get_region_names, download_harvard_oxford_atlas

__all__ = [
    'BrainAtlasMapper',
    'register_to_atlas_affine',
    'register_to_atlas_ants',
    'compute_region_overlap',
    'calculate_percentage_involvement',
    'analyze_tumor_subregions',
    'get_region_names',
    'download_harvard_oxford_atlas'
]

__version__ = '1.0.0'
