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
            
            # Resolve case_dir from the uploaded MRI images
            from cases.models import MRIImage
            mri_images = MRIImage.objects.filter(case=case)
            if not mri_images.exists():
                return Response(
                    {'error': 'No MRI images uploaded for this case. Please upload scans first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # case_dir is the parent of the first uploaded file
            media_root = Path(settings.MEDIA_ROOT)
            first_image = mri_images.first()
            # file_path is a Django FileField — use .name to get the string path
            first_path_str = first_image.file_path.name  # relative to MEDIA_ROOT
            first_abs = media_root / first_path_str
            case_dir = first_abs.parent

            if not case_dir.exists():
                return Response(
                    {'error': f'MRI files not found on disk at {case_dir}. Please re-upload.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Run the MoME+ inference engine
            engine = InferenceEngine()
            result = engine.run_inference(str(case_id), case_dir)

            return Response({
                'message': 'Segmentation completed successfully',
                'result': {
                    'volumes': result['volumes'],
                    'confidence_scores': result['confidence_scores'],
                    'gating_weights': result.get('gating_weights', {}),
                    'available_modalities': list(result.get('mask_files', {}).keys()),
                }
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
        
        # Include 2D slice visualizations if available
        from cases.models import Slice2DVisualization
        pred_slice_vizs = Slice2DVisualization.objects.filter(segmentation_result=result, is_ground_truth=False)
        if pred_slice_vizs.exists():
            response_data['slice_images'] = [
                {
                    'plane': sv.plane,
                    'slice_index': sv.slice_index,
                    'url': sv.image_file.url if sv.image_file else None,
                    'has_overlay': sv.has_overlay,
                }
                for sv in pred_slice_vizs
            ]
            
        gt_slice_vizs = Slice2DVisualization.objects.filter(segmentation_result=result, is_ground_truth=True)
        if gt_slice_vizs.exists():
            response_data['gt_slice_images'] = [
                {
                    'plane': sv.plane,
                    'slice_index': sv.slice_index,
                    'url': sv.image_file.url if sv.image_file else None,
                    'has_overlay': sv.has_overlay,
                }
                for sv in gt_slice_vizs
            ]
        
        return Response(response_data, status=status.HTTP_200_OK)


import nibabel as nib
import numpy as np

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
            
            rel_path = str(gt_file_path.relative_to(settings.MEDIA_ROOT))
            result.structured_findings['ground_truth_mask'] = rel_path

            # Load GT and compute Dice/IoU if predicted masks exist
            try:
                gt_img = nib.load(str(gt_file_path))
                gt_data = gt_img.get_fdata()
                
                # Voxel spacing to calculate mm3
                voxel_spacing = gt_img.header.get_zooms()[:3]
                voxel_vol_mm3 = float(np.prod(voxel_spacing))

                def calc_metrics(pred_path, gt_mask_bin):
                    if not pred_path:
                        return None
                    try:
                        abs_pred = Path(settings.MEDIA_ROOT) / str(pred_path)
                        pred_data = nib.load(str(abs_pred)).get_fdata() > 0
                        
                        pred_data = pred_data.astype(bool)
                        gt_mask_bin = gt_mask_bin.astype(bool)
                        
                        if pred_data.shape != gt_mask_bin.shape:
                            if set(pred_data.shape) == set(gt_mask_bin.shape):
                                for ax_perm in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                                    if np.transpose(gt_mask_bin, axes=ax_perm).shape == pred_data.shape:
                                        gt_mask_bin = np.transpose(gt_mask_bin, axes=ax_perm)
                                        break
                        
                        intersection = np.logical_and(pred_data, gt_mask_bin).sum()
                        union = np.logical_or(pred_data, gt_mask_bin).sum()
                        pred_sum = pred_data.sum()
                        gt_sum = gt_mask_bin.sum()
                        
                        dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
                        iou = intersection / union if union > 0 else 0.0
                        
                        return {
                            'dice': float(dice),
                            'iou': float(iou),
                            'pred_volume': float(pred_sum * voxel_vol_mm3),
                            'gt_volume': float(gt_sum * voxel_vol_mm3)
                        }
                    except Exception as e:
                        print(f"Error calculating metrics: {e}")
                        return None

                # BraTS Convention: WT = >0, TC = 1 | 4, ET = 4
                gt_wt = gt_data > 0
                gt_tc = np.logical_or(gt_data == 1, gt_data == 4)
                gt_et = gt_data == 4

                comparison = {
                    'whole_tumor': calc_metrics(result.whole_tumor_mask.name, gt_wt),
                    'tumor_core': calc_metrics(result.tumor_core_mask.name, gt_tc),
                    'enhancing_tumor': calc_metrics(result.enhancing_tumor_mask.name, gt_et)
                }
                
                result.structured_findings['ground_truth_comparison'] = comparison
                
                # Add 2D Slice Visualizations for GT
                from cases.models import MRIImage, Slice2DVisualization
                from src.inference.slice_visualizer import SliceVisualizer
                
                # Get T1ce or any available MRI
                mri_img = MRIImage.objects.filter(case=case, modality='t1ce').first()
                if not mri_img:
                    mri_img = MRIImage.objects.filter(case=case).first()
                
                if mri_img:
                    mri_path = Path(settings.MEDIA_ROOT) / mri_img.file_path.name
                    mri_nii = nib.load(str(mri_path))
                    mri_vol = mri_nii.get_fdata().astype(np.float32)
                    mri_vol = np.transpose(mri_vol, (2, 0, 1)) # to (D, H, W)
                    
                    gt_vol = np.transpose(gt_data, (2, 0, 1)) # to (D, H, W)
                    
                    viz = SliceVisualizer()
                    slice_dir = case_dir / 'slices'
                    slice_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate overlay composite
                    overlay_files = viz.generate_from_arrays(
                        mri_volume=mri_vol,
                        brats_mask=gt_vol,
                        output_dir=str(slice_dir),
                        plane='axial',
                        prefix=f'{case.case_id}_gt_overlay',
                        save_individual=False,
                        save_composite=True,
                        overlay_mode=True,
                    )
                    
                    # Delete existing GT slices if any
                    Slice2DVisualization.objects.filter(segmentation_result=result, is_ground_truth=True).delete()
                    
                    if 'composite' in overlay_files:
                        best_slice = viz.find_best_slice(gt_vol, 'axial')
                        rel_path_img = str(Path(overlay_files['composite']).relative_to(settings.MEDIA_ROOT))
                        
                        Slice2DVisualization.objects.create(
                            segmentation_result=result,
                            plane='axial',
                            slice_index=best_slice,
                            image_file=rel_path_img,
                            modality='t1ce' if mri_img.modality == 't1ce' else mri_img.modality,
                            has_overlay=True,
                            is_ground_truth=True
                        )
                        
                    # Generate standalone composite
                    standalone_files = viz.generate_from_arrays(
                        mri_volume=mri_vol,
                        brats_mask=gt_vol,
                        output_dir=str(slice_dir),
                        plane='axial',
                        prefix=f'{case.case_id}_gt_standalone',
                        save_individual=False,
                        save_composite=True,
                        overlay_mode=False,
                    )
                    
                    if 'composite' in standalone_files:
                        best_slice = viz.find_best_slice(gt_vol, 'axial')
                        rel_path_img = str(Path(standalone_files['composite']).relative_to(settings.MEDIA_ROOT))
                        
                        Slice2DVisualization.objects.create(
                            segmentation_result=result,
                            plane='axial',
                            slice_index=best_slice,
                            image_file=rel_path_img,
                            modality='t1ce' if mri_img.modality == 't1ce' else mri_img.modality,
                            has_overlay=False,
                            is_ground_truth=True
                        )
            except Exception as e:
                print(f"Failed to calculate ground truth metrics and visualizations: {e}")

            result.save()

            media_url = f"{settings.MEDIA_URL}{rel_path}"

            return Response({
                'message': 'Ground truth uploaded successfully',
                'file_path': rel_path,
                'url': media_url,
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
