"""
Serializers for Case models.
"""

from rest_framework import serializers
from .models import Case


class CaseSerializer(serializers.ModelSerializer):
    """Serializer for Case model with camelCase fields."""
    
    patientId = serializers.CharField(source='patient_id')
    createdBy = serializers.SerializerMethodField()
    scanDate = serializers.DateField(source='scan_date', required=False, allow_null=True)
    fieldStrength = serializers.CharField(source='field_strength', required=False, allow_blank=True)
    clinicalHistory = serializers.CharField(source='clinical_history', required=False, allow_blank=True)
    createdAt = serializers.DateTimeField(source='created_at', read_only=True)
    updatedAt = serializers.DateTimeField(source='updated_at', read_only=True)
    completedAt = serializers.DateTimeField(source='completed_at', read_only=True, allow_null=True)
    caseId = serializers.UUIDField(source='case_id', read_only=True)
    
    class Meta:
        model = Case
        fields = [
            'caseId', 'patientId', 'createdBy', 'status', 'age', 'sex',
            'scanDate', 'fieldStrength', 'clinicalHistory', 'indication',
            'createdAt', 'updatedAt', 'completedAt'
        ]
        read_only_fields = ['caseId', 'createdBy', 'createdAt', 'updatedAt', 'completedAt']
    
    def get_createdBy(self, obj):
        """Get creator's name."""
        return obj.created_by.get_full_name() if obj.created_by else 'Unknown'
