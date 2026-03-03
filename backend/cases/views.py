"""
Case management views.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import Case
from .serializers import CaseSerializer


class CaseListCreateView(APIView):
    """List all cases or create new case."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get cases scoped to the current user's role."""
        if request.user.is_staff or getattr(request.user, 'is_admin', False):
            # Admins see everything
            cases = Case.objects.all().order_by('-created_at')
        elif getattr(request.user, 'role', None) == 'patient':
            # Patients see cases where they are the linked patient user
            cases = Case.objects.filter(patient_user=request.user).order_by('-created_at')
        else:
            # Doctors / radiologists / researchers see their own created cases
            cases = Case.objects.filter(created_by=request.user).order_by('-created_at')

        serializer = CaseSerializer(cases, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        """Create new case."""
        serializer = CaseSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CaseDetailView(APIView):
    """Get case details."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, case_id):
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            serializer = CaseSerializer(case)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class CaseUpdateView(APIView):
    """Update case."""
    permission_classes = [IsAuthenticated]
    
    def patch(self, request, case_id):
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            serializer = CaseSerializer(case, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class CaseDeleteView(APIView):
    """Delete case."""
    permission_classes = [IsAuthenticated]
    
    def delete(self, request, case_id):
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            case.delete()
            return Response(
                {'message': 'Case deleted successfully'},
                status=status.HTTP_200_OK
            )
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )

class MRIImageUploadView(APIView):
    """Upload MRI images for a case."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, case_id):
        from .models import MRIImage
        from .mri_serializers import MRIImageSerializer
        from rest_framework.parsers import MultiPartParser, FormParser
        
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # Get uploaded file and modality
            file = request.FILES.get('file')
            modality = request.data.get('modality')
            
            if not file:
                return Response(
                    {'error': 'No file provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not modality:
                return Response(
                    {'error': 'Modality not specified'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check if modality already exists for this case
            existing = MRIImage.objects.filter(case=case, modality=modality).first()
            if existing:
                # Update existing
                existing.file_path = file
                existing.file_size = file.size
                existing.original_filename = file.name
                existing.save()
                mri_image = existing
            else:
                # Create new
                mri_image = MRIImage.objects.create(
                    case=case,
                    modality=modality,
                    file_path=file,
                    file_size=file.size,
                    original_filename=file.name
                )
            
            serializer = MRIImageSerializer(mri_image, context={'request': request})
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class MRIImageListView(APIView):
    """List MRI images for a case."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, case_id):
        from .models import MRIImage
        from .mri_serializers import MRIImageSerializer
        
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            images = MRIImage.objects.filter(case=case)
            serializer = MRIImageSerializer(images, many=True, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class SegmentationResultView(APIView):
    """Get segmentation results for a case."""
    permission_classes = [IsAuthenticated]

    def get(self, request, case_id):
        from .models import SegmentationResult
        from django.conf import settings

        try:
            case = Case.objects.get(case_id=case_id)
        except Case.DoesNotExist:
            return Response({'error': 'Case not found'}, status=status.HTTP_404_NOT_FOUND)

        # Permission: owner or admin
        if not request.user.is_staff and case.created_by != request.user:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        try:
            seg = SegmentationResult.objects.get(case=case)
        except SegmentationResult.DoesNotExist:
            return Response(
                {'error': 'No segmentation result found. Run inference first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        def mask_url(field_value):
            """Convert a file path to a media URL."""
            if not field_value:
                return None
            path_str = str(field_value)
            if path_str.startswith('/') or path_str[1:3] == ':\\':
                # Absolute path — make relative to MEDIA_ROOT
                try:
                    from pathlib import Path
                    rel = Path(path_str).relative_to(settings.MEDIA_ROOT)
                    return request.build_absolute_uri(f'{settings.MEDIA_URL}{rel}')
                except ValueError:
                    return None
            return request.build_absolute_uri(f'{settings.MEDIA_URL}{path_str}')

        structured = seg.structured_findings or {}

        return Response({
            'volumes': {
                'whole_tumor':     float(seg.whole_tumor_volume or 0),
                'tumor_core':      float(seg.tumor_core_volume or 0),
                'enhancing_tumor': float(seg.enhancing_tumor_volume or 0),
            },
            'confidence_scores': {
                'whole_tumor':     float(seg.whole_tumor_confidence or 0),
                'tumor_core':      float(seg.tumor_core_confidence or 0),
                'enhancing_tumor': float(seg.enhancing_tumor_confidence or 0),
            },
            'mask_files': {
                'whole_tumor':       mask_url(seg.whole_tumor_mask),
                'tumor_core':        mask_url(seg.tumor_core_mask),
                'enhancing_tumor':   mask_url(seg.enhancing_tumor_mask),
            },
            'structured_findings': structured,
            'gating_weights': structured.get('gating_weights', {}),
            'available_modalities': structured.get('available_modalities', []),
            'model_version': structured.get('model_version', 'MoME+ v1.0'),
            'device': structured.get('device', 'unknown'),
            'created_at': seg.created_at.isoformat() if hasattr(seg, 'created_at') and seg.created_at else None,
        })


class VisualizationListView(APIView):
    """List 2D visualizations for a case."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, case_id):
        return Response({"message": f"Visualizations for {case_id} - To be implemented"})


class FeedbackCreateView(APIView):
    """Submit feedback for a case."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, case_id):
        return Response({"message": f"Feedback for case {case_id} - To be implemented"})


class FeedbackListView(APIView):
    """List all feedback."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        return Response({"message": "Feedback list - To be implemented"})
