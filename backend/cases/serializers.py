"""
Serializers for Case models.
"""

from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Case

User = get_user_model()


class CaseSerializer(serializers.ModelSerializer):
    """Serializer for Case model with camelCase fields."""

    patientId       = serializers.CharField(source='patient_id')
    createdBy       = serializers.SerializerMethodField()
    scanDate        = serializers.DateField(source='scan_date', required=False, allow_null=True)
    fieldStrength   = serializers.CharField(source='field_strength', required=False, allow_blank=True)
    clinicalHistory = serializers.CharField(source='clinical_history', required=False, allow_blank=True)
    createdAt       = serializers.DateTimeField(source='created_at', read_only=True)
    updatedAt       = serializers.DateTimeField(source='updated_at', read_only=True)
    completedAt     = serializers.DateTimeField(source='completed_at', read_only=True, allow_null=True)
    caseId          = serializers.UUIDField(source='case_id', read_only=True)

    # Write-only: doctor provides patient email to link their account
    patientEmail    = serializers.EmailField(write_only=True, required=False, allow_blank=True)
    # Read-only: show linked patient email on GET
    patientUserEmail = serializers.SerializerMethodField()

    class Meta:
        model = Case
        fields = [
            'caseId', 'patientId', 'createdBy', 'status', 'age', 'sex',
            'scanDate', 'fieldStrength', 'clinicalHistory', 'indication',
            'createdAt', 'updatedAt', 'completedAt',
            'patientEmail', 'patientUserEmail',
        ]
        read_only_fields = ['caseId', 'createdBy', 'createdAt', 'updatedAt', 'completedAt']

    def get_createdBy(self, obj):
        return obj.created_by.get_full_name() if obj.created_by else 'Unknown'

    def get_patientUserEmail(self, obj):
        return obj.patient_user.email if obj.patient_user else None

    def create(self, validated_data):
        patient_email = validated_data.pop('patientEmail', None) or validated_data.pop('patient_email', None)
        case = super().create(validated_data)
        if patient_email:
            self._link_patient(case, patient_email)
        return case

    def update(self, instance, validated_data):
        patient_email = validated_data.pop('patientEmail', None) or validated_data.pop('patient_email', None)
        case = super().update(instance, validated_data)
        if patient_email:
            self._link_patient(case, patient_email)
        return case

    def _link_patient(self, case, email: str):
        """Lookup a patient user by email and link them to this case."""
        try:
            patient = User.objects.get(email__iexact=email.strip(), role='patient')
            case.patient_user = patient
            case.save(update_fields=['patient_user'])
        except User.DoesNotExist:
            pass  # silently skip if no matching patient account found
