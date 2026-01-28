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
        """Get all cases (all users can see all cases for now)."""
        # Show all cases to all users
        cases = Case.objects.all().order_by('-created_at')
        
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
        return Response({"message": f"Segmentation result for {case_id} - To be implemented"})


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
