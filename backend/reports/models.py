"""
Report models for managing AI-generated diagnostic reports.
Supports LLM-generated narratives, clinician edits, and PDF export.
"""

from django.db import models
from django.conf import settings
from django.utils import timezone
from cases.models import Case
import uuid


class Report(models.Model):
    """
    Stores AI-generated diagnostic reports with clinician edits.
    Supports interactive traceability and version control.
    """
    
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('reviewed', 'Reviewed'),
        ('finalized', 'Finalized'),
        ('exported', 'Exported'),
    ]
    
    # Identification
    report_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    
    # Relationships
    case = models.OneToOneField(
        Case,
        on_delete=models.CASCADE,
        related_name='report'
    )
    
    # Report Content
    ai_generated_text = models.TextField(help_text='Original AI-generated narrative')
    finalized_text = models.TextField(help_text='Clinician-edited final text')
    
    # Structured Data
    findings_json = models.JSONField(
        help_text='Structured findings from segmentation'
    )
    
    # Traceability Mapping
    traceability_map = models.JSONField(
        null=True,
        blank=True,
        help_text='Maps report sentences to evidence in findings_json'
    )
    
    # Template Information
    template_name = models.CharField(max_length=100, default='standard_brain_tumor')
    template_version = models.CharField(max_length=20, default='1.0')
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    
    # Editing History
    edit_count = models.IntegerField(default=0, help_text='Number of clinician edits')
    last_edited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reports_edited'
    )
    
    # Timestamps
    generated_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    finalized_at = models.DateTimeField(null=True, blank=True)
    exported_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Model Information
    llm_model_name = models.CharField(max_length=100, default='MedAlpaca-7B')
    llm_model_version = models.CharField(max_length=50, blank=True)
    generation_parameters = models.JSONField(
        null=True,
        blank=True,
        help_text='LLM generation parameters used'
    )
    
    class Meta:
        db_table = 'reports'
        verbose_name = 'Report'
        verbose_name_plural = 'Reports'
        ordering = ['-generated_at']
    
    def __str__(self):
        return f"Report {self.report_id} for Case {self.case.case_id}"
    
    def finalize(self, user):
        """Mark report as finalized by a clinician."""
        self.status = 'finalized'
        self.finalized_at = timezone.now()
        self.last_edited_by = user
        self.save()
    
    def increment_edit_count(self):
        """Increment edit count when clinician makes changes."""
        self.edit_count += 1
        self.save()


class ReportEdit(models.Model):
    """
    Tracks individual edits made by clinicians to reports.
    Maintains a complete audit trail for quality assurance.
    """
    
    # Relationships
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='edits')
    edited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='report_edits'
    )
    
    # Edit Details
    original_text = models.TextField()
    edited_text = models.TextField()
    section = models.CharField(
        max_length=50,
        help_text='Section of report that was edited'
    )
    
    # Metadata
    edit_reason = models.TextField(blank=True, help_text='Reason for edit (optional)')
    character_change_count = models.IntegerField(default=0)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'report_edits'
        verbose_name = 'Report Edit'
        verbose_name_plural = 'Report Edits'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Edit to Report {self.report.report_id} by {self.edited_by.get_full_name()}"


class ReportPDF(models.Model):
    """
    Stores generated PDF versions of reports.
    Includes visualizations and metadata.
    """
    
    # Relationships
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='pdf_versions')
    
    # File Information
    file_path = models.FileField(upload_to='reports/pdfs/', max_length=500)
    file_size = models.BigIntegerField(help_text='File size in bytes')
    
    # Generation Details
    generated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    
    # Content Configuration
    include_3d_render = models.BooleanField(default=True)
    include_all_slices = models.BooleanField(default=False)
    selected_slice_ids = models.JSONField(
        null=True,
        blank=True,
        help_text='List of Slice2DVisualization IDs to include'
    )
    
    # Metadata
    version_number = models.IntegerField(default=1)
    watermark = models.CharField(
        max_length=200,
        default='DRAFT - FOR RESEARCH PURPOSES ONLY',
        help_text='Watermark text for the PDF'
    )
    
    # Timestamps
    generated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'report_pdfs'
        verbose_name = 'Report PDF'
        verbose_name_plural = 'Report PDFs'
        ordering = ['-generated_at']
    
    def __str__(self):
        return f"PDF v{self.version_number} of Report {self.report.report_id}"


class ReportTemplate(models.Model):
    """
    Stores report templates for different clinical scenarios.
    Used by LLM for structured report generation.
    """
    
    TEMPLATE_TYPE_CHOICES = [
        ('brain_tumor', 'Brain Tumor'),
        ('stroke', 'Stroke/ISLES'),
        ('alzheimers', "Alzheimer's/OASIS"),
        ('custom', 'Custom'),
    ]
    
    # Template Information
    name = models.CharField(max_length=100, unique=True)
    template_type = models.CharField(max_length=30, choices=TEMPLATE_TYPE_CHOICES)
    version = models.CharField(max_length=20, default='1.0')
    
    # Template Content
    template_text = models.TextField(
        help_text='Template with placeholders for AI-generated content'
    )
    required_fields = models.JSONField(
        help_text='List of required fields in findings_json'
    )
    
    # Sections
    sections_structure = models.JSONField(
        help_text='Defines report sections and their order'
    )
    
    # Instructions for LLM
    llm_instructions = models.TextField(
        help_text='Instructions for the LLM on how to populate this template'
    )
    
    # Status
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False)
    
    # Metadata
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'report_templates'
        verbose_name = 'Report Template'
        verbose_name_plural = 'Report Templates'
        ordering = ['template_type', 'name']
    
    def __str__(self):
        return f"{self.name} v{self.version}"


class TraceabilityLink(models.Model):
    """
    Maps sentences in the report to specific evidence in the structured findings.
    Enables interactive traceability feature (REQ-28).
    """
    
    # Relationships
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='traceability_links')
    
    # Sentence Information
    sentence_text = models.TextField()
    sentence_index = models.IntegerField(help_text='Position in the report')
    section = models.CharField(max_length=50)
    
    # Evidence Link
    evidence_path = models.CharField(
        max_length=255,
        help_text='JSON path to evidence in findings_json (e.g., "tumor_metrics.whole_tumor_volume")'
    )
    evidence_value = models.JSONField(help_text='The actual value from findings_json')
    
    # Confidence
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        help_text='Confidence in this traceability mapping (0-1)'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'traceability_links'
        verbose_name = 'Traceability Link'
        verbose_name_plural = 'Traceability Links'
        ordering = ['report', 'sentence_index']
        unique_together = [['report', 'sentence_index']]
    
    def __str__(self):
        return f"Link for sentence {self.sentence_index} in Report {self.report.report_id}"
