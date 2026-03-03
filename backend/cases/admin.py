from django.contrib import admin
from .models import Case, MRIImage, SegmentationResult


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    list_display  = ('case_id', 'patient_id', 'created_by', 'status', 'created_at')
    list_filter   = ('status',)
    search_fields = ('patient_id', 'created_by__email')
    ordering      = ('-created_at',)
    readonly_fields = ('case_id', 'created_at', 'updated_at')


@admin.register(MRIImage)
class MRIImageAdmin(admin.ModelAdmin):
    list_display  = ('case', 'modality', 'original_filename', 'uploaded_at')
    list_filter   = ('modality',)
    search_fields = ('case__patient_id',)


@admin.register(SegmentationResult)
class SegmentationResultAdmin(admin.ModelAdmin):
    list_display  = ('case', 'whole_tumor_volume', 'tumor_core_volume', 'enhancing_tumor_volume')
    search_fields = ('case__patient_id',)
