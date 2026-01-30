"""
JSON Generation Module

Converts atlas mapping results into structured JSON descriptors
for LLM-based medical report generation.
"""

from .descriptor_generator import TumorDescriptorGenerator
from .schema_validator import load_schema, validate_descriptor

__all__ = [
    'TumorDescriptorGenerator',
    'load_schema',
    'validate_descriptor'
]

__version__ = '1.0.0'
