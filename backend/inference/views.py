"""
ML inference views for brain tumor segmentation.
Handles prediction requests and ground truth uploads.
"""

import traceback
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.conf import settings

from cases.models import Case, SegmentationResult
from .inference_utils import InferenceEngine


class PredictSegmentationView(APIView):
    """Run segmentation prediction on a case."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, case_id):
        """
        POST /api/inference/predict/<case_id>/
        
        Starts segmentation prediction for the specified case.
        """
        try:
            # Verify case exists and belongs to user
            try:
                case = Case.objects.get(case_id=case_id)
            except Case.DoesNotExist:
                return Response(
                    {'error': 'Case not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Check if user has permission (either owner or admin)
            if case.created_by != request.user and not request.user.is_staff:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # Update case status
            case.status = 'processing'
            case.save()
            
            # Run inference
            engine = InferenceEngine()
            result = engine.run_inference(str(case_id))
            
            return Response({
                'message': 'Segmentation completed successfully',
                'result': result
            }, status=status.HTTP_200_OK)
            
        except FileNotFoundError as e:
            case.status = 'failed'
            case.error_message = str(e)
            case.save()
            
            # Log full traceback for debugging
            print("FileNotFoundError during inference:", traceback.format_exc())
            
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except ValueError as e:
            case.status = 'failed'
            case.error_message = str(e)
            case.save()
            
            # Log full traceback for debugging
            print("ValueError during inference:", traceback.format_exc())
            
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            case.status = 'failed'
            case.error_message = str(e)
            case.save()
            
            # Log full traceback for debugging
            print("Inference error:", traceback.format_exc())
            
            return Response(
                {'error': f'Inference failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GetSegmentationResultView(APIView):
    """Get segmentation result for a case."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, case_id):
        """
        GET /api/inference/result/<case_id>/
        
        Retrieves segmentation results for the specified case.
        """
        try:
            case = Case.objects.get(case_id=case_id)
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check permission
        if case.created_by != request.user and not request.user.is_staff:
            return Response(
                {'error': 'Permission denied'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        try:
            result = SegmentationResult.objects.get(case=case)
        except SegmentationResult.DoesNotExist:
            return Response(
                {'error': 'No segmentation result found for this case'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Build response data
        response_data = {
            'case_id': str(case.case_id),
            'patient_id': case.patient_id,
            'volumes': {
                'whole_tumor': result.whole_tumor_volume,
                'tumor_core': result.tumor_core_volume,
                'enhancing_tumor': result.enhancing_tumor_volume
            },
            'confidence_scores': {
                'whole_tumor': result.whole_tumor_confidence,
                'tumor_core': result.tumor_core_confidence,
                'enhancing_tumor': result.enhancing_tumor_confidence
            },
            'mask_files': {
                'whole_tumor': result.whole_tumor_mask.url if result.whole_tumor_mask else None,
                'tumor_core': result.tumor_core_mask.url if result.tumor_core_mask else None,
                'enhancing_tumor': result.enhancing_tumor_mask.url if result.enhancing_tumor_mask else None
            },
            'structured_findings': result.structured_findings,
            'created_at': result.created_at.isoformat(),
            'updated_at': result.updated_at.isoformat()
        }
        
        return Response(response_data, status=status.HTTP_200_OK)


class UploadGroundTruthView(APIView):
    """Upload ground truth segmentation mask for comparison."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, case_id):
        """
        POST /api/inference/upload-ground-truth/<case_id>/
        
        Uploads a ground truth segmentation mask for comparison.
        Expects a NIfTI file in the request.
        """
        try:
            case = Case.objects.get(case_id=case_id)
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check permission
        if case.created_by != request.user and not request.user.is_staff:
            return Response(
                {'error': 'Permission denied'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Check if file was uploaded
        if 'file' not in request.FILES:
            return Response(
                {'error': 'No file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        
        # Validate file extension
        if not uploaded_file.name.endswith(('.nii', '.nii.gz')):
            return Response(
                {'error': 'File must be a NIfTI file (.nii or .nii.gz)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save file to case directory
        case_dir = Path(settings.MEDIA_ROOT) / 'cases' / str(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        
        gt_file_path = case_dir / f'ground_truth_{uploaded_file.name}'
        
        with open(gt_file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Update segmentation result with ground truth path
        try:
            result = SegmentationResult.objects.get(case=case)
            
            # Store ground truth path in structured_findings
            if not result.structured_findings:
                result.structured_findings = {}
            
            result.structured_findings['ground_truth_mask'] = str(gt_file_path.relative_to(settings.MEDIA_ROOT))
            result.save()
            
            return Response({
                'message': 'Ground truth uploaded successfully',
                'file_path': str(gt_file_path.relative_to(settings.MEDIA_ROOT))
            }, status=status.HTTP_200_OK)
            
        except SegmentationResult.DoesNotExist:
            return Response(
                {'error': 'No segmentation result found. Please run prediction first.'},
                status=status.HTTP_400_BAD_REQUEST
            )


# Legacy views (placeholders for future functionality)
class StartSegmentationView(APIView):
    """Start segmentation task for a case."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        return Response({"message": "Use /api/inference/predict/<case_id>/ instead"})


class TaskListView(APIView):
    """List all inference tasks."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        return Response({"message": "Task list - To be implemented"})


class TaskDetailView(APIView):
    """Get task details and status."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, task_id):
        return Response({"message": f"Task detail for {task_id} - To be implemented"})


class CancelTaskView(APIView):
    """Cancel a running task."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, task_id):
        return Response({"message": f"Cancel task {task_id} - To be implemented"})


class ModelVersionListView(APIView):
    """List all model versions."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        return Response({"message": "Model version list - To be implemented"})


class ModelVersionDetailView(APIView):
    """Get model version details."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, model_id):
        return Response({"message": f"Model detail for {model_id} - To be implemented"})


class ActivateModelView(APIView):
    """Activate a model version."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, model_id):
        return Response({"message": f"Activate model {model_id} - To be implemented"})


class ContinualLearningListView(APIView):
    """List continual learning tasks."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        return Response({"message": "CL task list - To be implemented"})


class StartContinualLearningView(APIView):
    """Start a continual learning task."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        return Response({"message": "Start CL task - To be implemented"})


class ContinualLearningDetailView(APIView):
    """Get continual learning task details."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, task_id):
        return Response({"message": f"CL task detail for {task_id} - To be implemented"})
