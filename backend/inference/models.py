"""
Inference models for tracking async ML tasks and model management.
Supports continual learning and model versioning.
"""

from django.db import models
from django.conf import settings
from django.utils import timezone
from cases.models import Case
import uuid


class InferenceTask(models.Model):
    """
    Tracks async inference tasks for segmentation and report generation.
    Linked to Celery tasks for monitoring.
    """
    
    TASK_TYPE_CHOICES = [
        ('segmentation', 'Segmentation'),
        ('report_generation', 'Report Generation'),
        ('visualization', '3D Visualization'),
        ('pdf_export', 'PDF Export'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    # Identification
    task_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    celery_task_id = models.CharField(max_length=255, unique=True, null=True, blank=True)
    
    # Relationships
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='inference_tasks')
    initiated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    
    # Task Details
    task_type = models.CharField(max_length=30, choices=TASK_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Progress Tracking
    progress_percentage = models.IntegerField(default=0, help_text='0-100')
    current_step = models.CharField(max_length=100, blank=True)
    total_steps = models.IntegerField(default=1)
    
    # Results
    result_data = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    error_traceback = models.TextField(blank=True)
    
    # Performance Metrics
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    
    # Model Information
    model_version = models.CharField(max_length=50, blank=True)
    device_used = models.CharField(max_length=20, blank=True, help_text='cuda/cpu')
    
    # Retries
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'inference_tasks'
        verbose_name = 'Inference Task'
        verbose_name_plural = 'Inference Tasks'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['celery_task_id']),
            models.Index(fields=['status']),
            models.Index(fields=['task_type']),
        ]
    
    def __str__(self):
        return f"{self.get_task_type_display()} for Case {self.case.case_id} - {self.status}"
    
    def start(self):
        """Mark task as started."""
        self.status = 'running'
        self.started_at = timezone.now()
        self.save()
    
    def complete(self, result_data=None):
        """Mark task as completed."""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.progress_percentage = 100
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        if result_data:
            self.result_data = result_data
        self.save()
    
    def fail(self, error_message, traceback=''):
        """Mark task as failed."""
        self.status = 'failed'
        self.completed_at = timezone.now()
        self.error_message = error_message
        self.error_traceback = traceback
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.save()
    
    def update_progress(self, percentage, current_step=''):
        """Update task progress."""
        self.progress_percentage = min(percentage, 100)
        self.current_step = current_step
        self.save()


class ModelVersion(models.Model):
    """
    Tracks different versions of ML models (segmentation and LLM).
    Supports continual learning and model rollback.
    """
    
    MODEL_TYPE_CHOICES = [
        ('segmentation', 'Segmentation Model (MoME+)'),
        ('llm', 'Language Model'),
    ]
    
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('testing', 'Testing'),
        ('active', 'Active'),
        ('deprecated', 'Deprecated'),
        ('archived', 'Archived'),
    ]
    
    # Identification
    model_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    
    # File Information
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField(help_text='Model size in bytes')
    checksum = models.CharField(max_length=64, help_text='SHA256 checksum')
    
    # Training Information
    trained_on_task = models.CharField(max_length=100, blank=True)
    training_dataset = models.CharField(max_length=200, blank=True)
    training_date = models.DateTimeField(null=True, blank=True)
    trained_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='trained_models'
    )
    
    # Performance Metrics
    validation_metrics = models.JSONField(
        null=True,
        blank=True,
        help_text='Dice scores, IoU, etc.'
    )
    benchmark_results = models.JSONField(null=True, blank=True)
    
    # Continual Learning Metrics
    forgetting_score = models.FloatField(
        null=True,
        blank=True,
        help_text='Backward transfer metric'
    )
    previous_version = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='subsequent_versions'
    )
    
    # Configuration
    hyperparameters = models.JSONField(null=True, blank=True)
    architecture_config = models.JSONField(null=True, blank=True)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    is_default = models.BooleanField(default=False)
    
    # Metadata
    description = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    deployed_at = models.DateTimeField(null=True, blank=True)
    deprecated_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'model_versions'
        verbose_name = 'Model Version'
        verbose_name_plural = 'Model Versions'
        ordering = ['-created_at']
        unique_together = [['name', 'version']]
    
    def __str__(self):
        return f"{self.name} v{self.version} ({self.get_model_type_display()})"
    
    def activate(self):
        """Set this model as the active version."""
        # Deactivate other models of same type
        ModelVersion.objects.filter(
            model_type=self.model_type,
            status='active'
        ).update(status='deprecated', deprecated_at=timezone.now())
        
        self.status = 'active'
        self.deployed_at = timezone.now()
        self.is_default = True
        self.save()


class ContinualLearningTask(models.Model):
    """
    Tracks continual learning training tasks.
    Manages dataset introduction and model updates.
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('preparing_data', 'Preparing Data'),
        ('training', 'Training'),
        ('evaluating', 'Evaluating'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    # Identification
    task_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, primary_key=True)
    celery_task_id = models.CharField(max_length=255, unique=True, null=True, blank=True)
    
    # Task Details
    task_name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Dataset Information
    new_dataset_name = models.CharField(max_length=200)
    new_dataset_path = models.CharField(max_length=500)
    num_samples = models.IntegerField(default=0)
    
    # Model Management
    base_model = models.ForeignKey(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name='cl_tasks_from_this_model'
    )
    new_model_version = models.OneToOneField(
        ModelVersion,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='cl_task'
    )
    
    # CL Strategy
    use_ewc = models.BooleanField(default=True)
    use_replay = models.BooleanField(default=True)
    ewc_lambda = models.FloatField(default=1000.0)
    replay_buffer_size = models.IntegerField(default=500)
    
    # Training Configuration
    config = models.JSONField(help_text='Training configuration')
    
    # Results
    training_metrics = models.JSONField(null=True, blank=True)
    evaluation_results = models.JSONField(null=True, blank=True)
    
    # Performance
    learning_accuracy = models.FloatField(null=True, blank=True, help_text='Performance on new task')
    retention_accuracy = models.FloatField(null=True, blank=True, help_text='Performance on old tasks')
    forgetting_score = models.FloatField(null=True, blank=True)
    
    # Metadata
    initiated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'continual_learning_tasks'
        verbose_name = 'Continual Learning Task'
        verbose_name_plural = 'Continual Learning Tasks'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"CL Task: {self.task_name} - {self.status}"
