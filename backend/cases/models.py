"""
Case models for managing patient cases, MRI scans, and analysis results.
Implements the core database schema for medical image processing.
"""

from django.db import models
from django.conf import settings
from django.utils import timezone
import uuid
import os


def case_upload_path(instance, filename):
    """Generate upload path for case files."""
    return f'cases/{instance.case_id}/{filename}'


class Case(models.Model):
    """
    Represents a patient case with multi-modal MRI scans.
    Main entity for the diagnostic workflow.
    """
    
    STATUS_CHOICES = [
        ('created', 'Created'),
        ('uploading', 'Uploading'),
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('archived', 'Archived'),
    ]
    
    # Identification
    case_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    patient_id = models.CharField(
        max_length=100,
        help_text='De-identified patient ID (no PII)'
    )
    
    # Ownership
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='cases'
    )
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='created')
    priority = models.IntegerField(default=0, help_text='Higher number = higher priority')
    
    # Clinical Information
    age = models.IntegerField(null=True, blank=True, help_text='Patient age at scan')
    sex = models.CharField(
        max_length=10,
        choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')],
        null=True,
        blank=True
    )
    clinical_history = models.TextField(blank=True, help_text='Relevant clinical history')
    indication = models.TextField(blank=True, help_text='Indication for imaging')
    
    # Scan Information
    scan_date = models.DateField(null=True, blank=True)
    scanner_type = models.CharField(max_length=100, blank=True, help_text='MRI scanner model')
    field_strength = models.CharField(
        max_length=10,
        choices=[('1.5T', '1.5 Tesla'), ('3T', '3 Tesla'), ('7T', '7 Tesla')],
        blank=True
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Processing Info
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_ended_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    # Model Versioning
    segmentation_model_version = models.CharField(max_length=50, blank=True)
    llm_model_version = models.CharField(max_length=50, blank=True)
    
    # Metadata
    notes = models.TextField(blank=True, help_text='Additional notes')
    is_training_data = models.BooleanField(default=False, help_text='Use for continual learning')
    
    class Meta:
        db_table = 'cases'
        verbose_name = 'Case'
        verbose_name_plural = 'Cases'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['patient_id']),
            models.Index(fields=['status']),
            models.Index(fields=['created_by', '-created_at']),
        ]
    
    def __str__(self):
        return f"Case {self.case_id} - {self.patient_id}"
    
    @property
    def processing_duration(self):
        """Calculate processing duration in seconds."""
        if self.processing_started_at and self.processing_ended_at:
            delta = self.processing_ended_at - self.processing_started_at
            return delta.total_seconds()
        return None
    
    def complete(self):
        """Mark case as completed."""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()
    
    def fail(self, error_message):
        """Mark case as failed with error message."""
        self.status = 'failed'
        self.error_message = error_message
        self.processing_ended_at = timezone.now()
        self.save()


class MRIImage(models.Model):
    """
    Stores individual MRI modality images for a case.
    Supports T1, T1ce, T2, and FLAIR modalities.
    """
    
    MODALITY_CHOICES = [
        ('t1', 'T1'),
        ('t1ce', 'T1 Contrast Enhanced'),
        ('t2', 'T2'),
        ('flair', 'FLAIR'),
    ]
    
    # Relationships
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='mri_images')
    
    # Image Information
    modality = models.CharField(max_length=10, choices=MODALITY_CHOICES)
    file_path = models.FileField(upload_to=case_upload_path, max_length=500)
    file_size = models.BigIntegerField(help_text='File size in bytes')
    original_filename = models.CharField(max_length=255)
    
    # Technical Metadata
    dimensions = models.JSONField(null=True, blank=True, help_text='Image dimensions [x, y, z]')
    spacing = models.JSONField(null=True, blank=True, help_text='Voxel spacing [x, y, z]')
    orientation = models.CharField(max_length=10, blank=True)
    
    # Validation
    is_valid = models.BooleanField(default=True)
    validation_errors = models.JSONField(null=True, blank=True)
    
    # Timestamps
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'mri_images'
        verbose_name = 'MRI Image'
        verbose_name_plural = 'MRI Images'
        unique_together = [['case', 'modality']]
        ordering = ['case', 'modality']
    
    def __str__(self):
        return f"{self.case.case_id} - {self.get_modality_display()}"


class SegmentationResult(models.Model):
    """
    Stores AI-generated segmentation results for a case.
    Includes tumor regions: WT, TC, ET.
    """
    
    # Relationships
    case = models.OneToOneField(
        Case,
        on_delete=models.CASCADE,
        related_name='segmentation_result',
        primary_key=True
    )
    
    # Segmentation Masks
    whole_tumor_mask = models.FileField(upload_to=case_upload_path, max_length=500)
    tumor_core_mask = models.FileField(upload_to=case_upload_path, max_length=500)
    enhancing_tumor_mask = models.FileField(upload_to=case_upload_path, max_length=500)
    
    # 3D Visualization
    gltf_file = models.FileField(upload_to=case_upload_path, max_length=500, null=True, blank=True)
    
    # Quantitative Metrics (in cubic mm)
    whole_tumor_volume = models.FloatField(help_text='WT volume in mm³')
    tumor_core_volume = models.FloatField(help_text='TC volume in mm³')
    enhancing_tumor_volume = models.FloatField(help_text='ET volume in mm³')
    
    # Confidence Scores (0-1)
    whole_tumor_confidence = models.FloatField(null=True, blank=True)
    tumor_core_confidence = models.FloatField(null=True, blank=True)
    enhancing_tumor_confidence = models.FloatField(null=True, blank=True)
    
    # Additional Metrics
    tumor_location = models.JSONField(null=True, blank=True, help_text='Centroid coordinates [x, y, z]')
    boundary_features = models.JSONField(null=True, blank=True)
    
    # Structured Data for Report Generation
    structured_findings = models.JSONField(
        help_text='Structured JSON for LLM report generation'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'segmentation_results'
        verbose_name = 'Segmentation Result'
        verbose_name_plural = 'Segmentation Results'
    
    def __str__(self):
        return f"Segmentation for Case {self.case.case_id}"


class Slice2DVisualization(models.Model):
    """
    Stores 2D slice visualizations with segmentation overlays.
    Used for report generation and clinician review.
    """
    
    PLANE_CHOICES = [
        ('axial', 'Axial'),
        ('coronal', 'Coronal'),
        ('sagittal', 'Sagittal'),
    ]
    
    # Relationships
    segmentation_result = models.ForeignKey(
        SegmentationResult,
        on_delete=models.CASCADE,
        related_name='slice_visualizations'
    )
    
    # Slice Information
    plane = models.CharField(max_length=10, choices=PLANE_CHOICES)
    slice_index = models.IntegerField(help_text='Slice index in the volume')
    image_file = models.ImageField(upload_to=case_upload_path, max_length=500)
    
    # Metadata
    modality = models.CharField(max_length=10, choices=MRIImage.MODALITY_CHOICES)
    has_overlay = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'slice_2d_visualizations'
        verbose_name = '2D Slice Visualization'
        verbose_name_plural = '2D Slice Visualizations'
        unique_together = [['segmentation_result', 'plane', 'slice_index', 'modality']]
        ordering = ['plane', 'slice_index']
    
    def __str__(self):
        return f"{self.get_plane_display()} slice {self.slice_index} - {self.segmentation_result.case.case_id}"


class ClinicianFeedback(models.Model):
    """
    Captures clinician feedback on AI-generated results.
    Used for continual learning and model improvement.
    """
    
    FEEDBACK_TYPE_CHOICES = [
        ('segmentation_error', 'Segmentation Error'),
        ('report_error', 'Report Error'),
        ('general', 'General Feedback'),
    ]
    
    SEVERITY_CHOICES = [
        ('minor', 'Minor'),
        ('moderate', 'Moderate'),
        ('major', 'Major'),
        ('critical', 'Critical'),
    ]
    
    # Relationships
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='feedback')
    clinician = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='feedback_given'
    )
    
    # Feedback Details
    feedback_type = models.CharField(max_length=30, choices=FEEDBACK_TYPE_CHOICES)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='minor')
    description = models.TextField()
    correction_data = models.JSONField(
        null=True,
        blank=True,
        help_text='Corrected segmentation or report data'
    )
    
    # Status
    is_resolved = models.BooleanField(default=False)
    resolution_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'clinician_feedback'
        verbose_name = 'Clinician Feedback'
        verbose_name_plural = 'Clinician Feedback'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback on Case {self.case.case_id} by {self.clinician.get_full_name()}"
