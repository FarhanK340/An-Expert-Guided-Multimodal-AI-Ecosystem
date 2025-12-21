"""
Serializers for MRI images.
"""

from rest_framework import serializers
from .models import MRIImage


class MRIImageSerializer(serializers.ModelSerializer):
    """Serializer for MRI Image model."""
    
    originalFilename = serializers.CharField(source='original_filename', read_only=True)
    fileSize = serializers.IntegerField(source='file_size', read_only=True)
    filePath = serializers.SerializerMethodField()
    uploadedAt = serializers.DateTimeField(source='uploaded_at', read_only=True)
    isValid = serializers.BooleanField(source='is_valid', read_only=True)
    
    class Meta:
        model = MRIImage
        fields = [
            'id', 'modality', 'originalFilename', 'fileSize',
            'filePath', 'dimensions', 'spacing', 'isValid',
            'uploadedAt'
        ]
        read_only_fields = ['id', 'originalFilename', 'fileSize', 'filePath', 'uploadedAt', 'isValid']
    
    def get_filePath(self, obj):
        """Get the file URL."""
        if obj.file_path:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file_path.url)
        return None
