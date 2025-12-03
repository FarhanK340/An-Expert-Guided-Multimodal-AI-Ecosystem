"""
Tests for API endpoints and functionality.

Tests the Django REST Framework API endpoints, authentication,
and request/response handling.
"""

import pytest
import json
from django.test import TestCase, Client
from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock

from src.api.routes.inference_routes import InferenceViewSet
from src.api.routes.report_routes import ReportViewSet
from src.api.routes.feedback_routes import FeedbackViewSet
from src.api.services.segmentation_service import SegmentationService
from src.api.services.report_service import ReportService
from src.api.services.feedback_service import FeedbackService


class TestInferenceAPI(APITestCase):
    """Test inference API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        self.inference_url = reverse('inference-list')
    
    def test_inference_status_endpoint(self):
        """Test inference status endpoint."""
        response = self.client.get(f"{self.inference_url}status/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('model_info', data)
        self.assertIn('supported_formats', data)
    
    def test_supported_formats_endpoint(self):
        """Test supported formats endpoint."""
        response = self.client.get(f"{self.inference_url}supported_formats/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('input_formats', data)
        self.assertIn('output_formats', data)
        self.assertIn('modalities', data)
        self.assertIn('max_file_size', data)
    
    @patch('src.api.routes.inference_routes.SegmentationService')
    def test_segment_endpoint(self, mock_service):
        """Test segmentation endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.segment_images.return_value = {
            'status': 'success',
            'segmentation_summary': {'total_voxels': 1000}
        }
        
        # Create test data
        test_data = {
            'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
            'output_format': 'json'
        }
        
        # Create test files
        from django.core.files.uploadedfile import SimpleUploadedFile
        test_files = {
            'T1': SimpleUploadedFile('T1.nii.gz', b'fake nifti data'),
            'T1ce': SimpleUploadedFile('T1ce.nii.gz', b'fake nifti data'),
            'T2': SimpleUploadedFile('T2.nii.gz', b'fake nifti data'),
            'FLAIR': SimpleUploadedFile('FLAIR.nii.gz', b'fake nifti data')
        }
        
        # Make request
        response = self.client.post(f"{self.inference_url}segment/", {
            **test_data,
            **test_files
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'success')
    
    def test_segment_endpoint_no_files(self):
        """Test segmentation endpoint with no files."""
        response = self.client.post(f"{self.inference_url}segment/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No files uploaded')
    
    @patch('src.api.routes.inference_routes.SegmentationService')
    def test_batch_segment_endpoint(self, mock_service):
        """Test batch segmentation endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.segment_images.return_value = {
            'status': 'success',
            'segmentation_summary': {'total_voxels': 1000}
        }
        
        # Create test files
        from django.core.files.uploadedfile import SimpleUploadedFile
        test_files = [
            SimpleUploadedFile('case1_T1.nii.gz', b'fake nifti data'),
            SimpleUploadedFile('case1_T1ce.nii.gz', b'fake nifti data'),
            SimpleUploadedFile('case2_T1.nii.gz', b'fake nifti data'),
            SimpleUploadedFile('case2_T1ce.nii.gz', b'fake nifti data')
        ]
        
        # Make request
        response = self.client.post(f"{self.inference_url}batch_segment/", {
            'modalities': ['T1', 'T1ce'],
            'output_format': 'json',
            'files': test_files
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('batch_results', data)
        self.assertIn('total_cases', data)
        self.assertIn('successful_cases', data)
    
    def test_validate_files_endpoint(self):
        """Test file validation endpoint."""
        from django.core.files.uploadedfile import SimpleUploadedFile
        
        # Create test files
        test_files = [
            SimpleUploadedFile('valid.nii.gz', b'fake nifti data'),
            SimpleUploadedFile('invalid.txt', b'fake text data'),
            SimpleUploadedFile('large.nii.gz', b'x' * (101 * 1024 * 1024))  # 101MB
        ]
        
        # Make request
        response = self.client.post(f"{self.inference_url}validate_files/", {
            'files': test_files
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('validation_results', data)
        self.assertIn('total_files', data)
        self.assertIn('valid_files', data)
        
        # Check validation results
        results = data['validation_results']
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]['valid'])  # Valid file
        self.assertFalse(results[1]['valid'])  # Invalid extension
        self.assertFalse(results[2]['valid'])  # Too large


class TestReportAPI(APITestCase):
    """Test report API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        self.report_url = reverse('reports-list')
    
    def test_templates_endpoint(self):
        """Test templates endpoint."""
        response = self.client.get(f"{self.report_url}templates/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('templates', data)
    
    @patch('src.api.routes.report_routes.ReportService')
    def test_generate_report_endpoint(self, mock_service):
        """Test report generation endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.generate_report.return_value = {
            'status': 'success',
            'report': 'Generated clinical report',
            'template_used': 'standard'
        }
        
        # Create test data
        test_data = {
            'segmentation_data': {
                'segmentation_summary': {'total_voxels': 1000},
                'region_counts': {'1': 500, '2': 300, '3': 200}
            },
            'report_type': 'standard',
            'template': 'default',
            'patient_info': {
                'patient_id': 'TEST001',
                'age': 45,
                'gender': 'M'
            }
        }
        
        # Make request
        response = self.client.post(f"{self.report_url}generate/", test_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('report', data)
    
    def test_generate_report_endpoint_no_data(self):
        """Test report generation endpoint with no data."""
        response = self.client.post(f"{self.report_url}generate/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No segmentation data provided')
    
    @patch('src.api.routes.report_routes.ReportService')
    def test_customize_template_endpoint(self, mock_service):
        """Test template customization endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.customize_template.return_value = {
            'status': 'success',
            'template': 'Customized template',
            'template_name': 'standard'
        }
        
        # Create test data
        test_data = {
            'template_name': 'standard',
            'customizations': {
                'sections': ['patient_info', 'imaging_findings', 'clinical_impression'],
                'style': 'detailed'
            }
        }
        
        # Make request
        response = self.client.post(f"{self.report_url}customize_template/", test_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('template', data)
    
    def test_customize_template_endpoint_no_name(self):
        """Test template customization endpoint with no template name."""
        response = self.client.post(f"{self.report_url}customize_template/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Template name required')
    
    @patch('src.api.routes.report_routes.ReportService')
    def test_export_report_endpoint(self, mock_service):
        """Test report export endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.export_report.return_value = {
            'content': b'PDF content',
            'content_type': 'application/pdf',
            'filename': 'report.pdf'
        }
        
        # Create test data
        test_data = {
            'report_data': {
                'report': 'Test report content',
                'metadata': {'generated_at': '2023-01-01T00:00:00Z'}
            },
            'format': 'pdf'
        }
        
        # Make request
        response = self.client.post(f"{self.report_url}export_report/", test_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertIn('attachment', response['Content-Disposition'])
    
    def test_export_report_endpoint_no_data(self):
        """Test report export endpoint with no data."""
        response = self.client.post(f"{self.report_url}export_report/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No report data provided')
    
    def test_report_history_endpoint(self):
        """Test report history endpoint."""
        response = self.client.get(f"{self.report_url}report_history/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('reports', data)
        self.assertIn('total_count', data)


class TestFeedbackAPI(APITestCase):
    """Test feedback API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        self.feedback_url = reverse('feedback-list')
    
    def test_feedback_categories_endpoint(self):
        """Test feedback categories endpoint."""
        response = self.client.get(f"{self.feedback_url}feedback_categories/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('categories', data)
        self.assertIn('descriptions', data)
    
    @patch('src.api.routes.feedback_routes.FeedbackService')
    def test_submit_feedback_endpoint(self, mock_service):
        """Test feedback submission endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.submit_feedback.return_value = {
            'status': 'success',
            'feedback_id': 'feedback_001',
            'message': 'Feedback submitted successfully'
        }
        
        # Create test data
        test_data = {
            'segmentation_id': 'seg_001',
            'feedback_type': 'rating',
            'feedback_data': {
                'quality_rating': 4,
                'comments': 'Good segmentation quality'
            },
            'clinician_info': {
                'clinician_id': 'clin_001',
                'experience_years': 5
            }
        }
        
        # Make request
        response = self.client.post(f"{self.feedback_url}submit/", test_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('feedback_id', data)
    
    def test_submit_feedback_endpoint_no_id(self):
        """Test feedback submission endpoint with no segmentation ID."""
        response = self.client.post(f"{self.feedback_url}submit/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Segmentation ID required')
    
    @patch('src.api.routes.feedback_routes.FeedbackService')
    def test_correct_segmentation_endpoint(self, mock_service):
        """Test segmentation correction endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.submit_correction.return_value = {
            'status': 'success',
            'correction_id': 'correction_001',
            'message': 'Correction submitted successfully'
        }
        
        # Create test data
        test_data = {
            'segmentation_id': 'seg_001',
            'corrected_segmentation': {
                'segmentation_data': np.random.randint(0, 4, (128, 128, 128)).tolist()
            },
            'correction_notes': 'Corrected tumor boundaries',
            'clinician_info': {
                'clinician_id': 'clin_001',
                'experience_years': 5
            }
        }
        
        # Make request
        response = self.client.post(f"{self.feedback_url}correct_segmentation/", test_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('correction_id', data)
    
    def test_correct_segmentation_endpoint_no_data(self):
        """Test segmentation correction endpoint with no data."""
        response = self.client.post(f"{self.feedback_url}correct_segmentation/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Segmentation ID and corrected segmentation required')
    
    @patch('src.api.routes.feedback_routes.FeedbackService')
    def test_rate_quality_endpoint(self, mock_service):
        """Test quality rating endpoint."""
        # Mock the service
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.submit_rating.return_value = {
            'status': 'success',
            'rating_id': 'rating_001',
            'message': 'Rating submitted successfully'
        }
        
        # Create test data
        test_data = {
            'segmentation_id': 'seg_001',
            'quality_rating': 4,
            'rating_criteria': {
                'accuracy': 4,
                'completeness': 3,
                'consistency': 5
            },
            'clinician_info': {
                'clinician_id': 'clin_001',
                'experience_years': 5
            }
        }
        
        # Make request
        response = self.client.post(f"{self.feedback_url}rate_quality/", test_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('rating_id', data)
    
    def test_rate_quality_endpoint_invalid_rating(self):
        """Test quality rating endpoint with invalid rating."""
        test_data = {
            'segmentation_id': 'seg_001',
            'quality_rating': 6  # Invalid rating (should be 1-5)
        }
        
        response = self.client.post(f"{self.feedback_url}rate_quality/", test_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Quality rating must be between 1 and 5')
    
    def test_feedback_history_endpoint(self):
        """Test feedback history endpoint."""
        response = self.client.get(f"{self.feedback_url}feedback_history/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('feedback', data)
        self.assertIn('total_count', data)
    
    def test_feedback_statistics_endpoint(self):
        """Test feedback statistics endpoint."""
        response = self.client.get(f"{self.feedback_url}feedback_statistics/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('time_period', data)
        self.assertIn('total_feedback', data)
        self.assertIn('feedback_by_type', data)


class TestAPIHealth(APITestCase):
    """Test API health and status endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/api/v1/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('message', data)
        self.assertIn('version', data)
    
    def test_api_status_endpoint(self):
        """Test API status endpoint."""
        response = self.client.get('/api/v1/status/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('api_name', data)
        self.assertIn('version', data)
        self.assertIn('status', data)
        self.assertIn('endpoints', data)
        self.assertIn('features', data)
    
    def test_api_docs_endpoint(self):
        """Test API documentation endpoint."""
        response = self.client.get('/api/v1/docs/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn('title', data)
        self.assertIn('version', data)
        self.assertIn('description', data)
        self.assertIn('endpoints', data)
        self.assertIn('authentication', data)


if __name__ == "__main__":
    pytest.main([__file__])


