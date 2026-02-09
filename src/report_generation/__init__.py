"""
Report Generation Module

Transforms structured JSON anatomical descriptors into fluent radiology reports
using LoRA fine-tuned language models.
"""

from .report_generator import ReportGenerator, TemplateBasedGenerator
from .metrics import compute_metrics, evaluate_report_quality

__all__ = [
    'ReportGenerator',
    'TemplateBasedGenerator',
    'compute_metrics',
    'evaluate_report_quality'
]

__version__ = '1.0.0'
