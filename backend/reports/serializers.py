"""
Serializers for Report models.
"""

from rest_framework import serializers
from .models import Report, ReportEdit


class ReportSerializer(serializers.ModelSerializer):
    """Serializer for Report with camelCase fields for frontend."""

    reportId = serializers.UUIDField(source='report_id', read_only=True)
    caseId = serializers.UUIDField(source='case.case_id', read_only=True)
    patientId = serializers.CharField(source='case.patient_id', read_only=True)
    aiGeneratedText = serializers.CharField(source='ai_generated_text', read_only=True)
    finalizedText = serializers.CharField(source='finalized_text')
    findingsJson = serializers.JSONField(source='findings_json', read_only=True)
    editCount = serializers.IntegerField(source='edit_count', read_only=True)
    generatedAt = serializers.DateTimeField(source='generated_at', read_only=True)
    updatedAt = serializers.DateTimeField(source='updated_at', read_only=True)
    lastEditedBy = serializers.SerializerMethodField()

    class Meta:
        model = Report
        fields = [
            'reportId', 'caseId', 'patientId', 'status',
            'aiGeneratedText', 'finalizedText', 'findingsJson',
            'editCount', 'generatedAt', 'updatedAt', 'lastEditedBy',
        ]
        read_only_fields = [
            'reportId', 'caseId', 'patientId', 'aiGeneratedText',
            'findingsJson', 'editCount', 'generatedAt', 'updatedAt',
        ]

    def get_lastEditedBy(self, obj):
        if obj.last_edited_by:
            return obj.last_edited_by.get_full_name()
        return None


class ReportUpdateSerializer(serializers.Serializer):
    """Serializer for clinician edits to a report."""

    finalizedText = serializers.CharField()
    editReason = serializers.CharField(required=False, allow_blank=True, default='')
