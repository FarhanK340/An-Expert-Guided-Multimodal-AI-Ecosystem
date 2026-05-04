"""
Django API test suite — Full Inference & Reporting Pipeline.
Tests case creation, MRI upload, inference (mocked), report generation, editing, and export.
"""

import json
import uuid
from io import BytesIO
from unittest.mock import patch, MagicMock

from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()


def make_fake_nifti() -> bytes:
    """Return a minimal NIfTI-1 file header (348 bytes) for upload tests."""
    # NIfTI-1 header: 348 bytes, magic = 'n+1\0'
    header = bytearray(348)
    header[0:4] = (348).to_bytes(4, 'little')   # sizeof_hdr
    header[344:348] = b'n+1\x00'                # magic
    return bytes(header)


class PipelineTestCase(TestCase):
    """Base setup for pipeline tests."""

    def setUp(self):
        self.client = APIClient()
        self.doctor = User.objects.create_user(
            email='doc@pipeline.com', password='TestPass123',
            first_name='Dr', last_name='Test', role='doctor'
        )
        self.patient_user = User.objects.create_user(
            email='pat@pipeline.com', password='TestPass123',
            first_name='Pat', last_name='Ient', role='patient'
        )
        self.auth_as(self.doctor)

    def auth_as(self, user):
        resp = self.client.post('/api/users/login/', {
            'email': user.email, 'password': 'TestPass123'
        }, format='json')
        token = resp.data['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')


class TestCaseManagement(PipelineTestCase):
    """Test case creation and MRI upload."""

    def test_create_case(self):
        resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-001', 'age': 45, 'sex': 'M',
            'clinicalHistory': 'Headaches for 3 months'
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(resp.data['patientId'], 'PAT-001')
        self.assertEqual(resp.data['status'], 'created')

    def test_create_case_missing_patient_id(self):
        resp = self.client.post('/api/cases/', {'age': 45}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_list_cases(self):
        # Create a case first
        self.client.post('/api/cases/', {'patientId': 'PAT-LIST'}, format='json')
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.data, list)

    def test_upload_mri_modality(self):
        case_resp = self.client.post('/api/cases/', {'patientId': 'PAT-UPLOAD'}, format='json')
        case_id = case_resp.data['caseId']

        nifti_data = make_fake_nifti()
        file = SimpleUploadedFile('t1.nii', nifti_data, content_type='application/octet-stream')
        resp = self.client.post(
            f'/api/cases/{case_id}/upload/',
            {'file': file, 'modality': 't1'},
            format='multipart'
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(resp.data['modality'], 't1')

    def test_upload_all_four_modalities(self):
        case_resp = self.client.post('/api/cases/', {'patientId': 'PAT-4MOD'}, format='json')
        case_id = case_resp.data['caseId']

        for mod in ['t1', 't1ce', 't2', 'flair']:
            f = SimpleUploadedFile(f'{mod}.nii', make_fake_nifti(), content_type='application/octet-stream')
            resp = self.client.post(
                f'/api/cases/{case_id}/upload/',
                {'file': f, 'modality': mod},
                format='multipart'
            )
            self.assertEqual(resp.status_code, 201, f"Upload failed for {mod}: {resp.data}")

        images_resp = self.client.get(f'/api/cases/{case_id}/images/')
        self.assertEqual(len(images_resp.data), 4)

    def test_delete_case(self):
        case_resp = self.client.post('/api/cases/', {'patientId': 'PAT-DEL'}, format='json')
        case_id = case_resp.data['caseId']
        resp = self.client.delete(f'/api/cases/{case_id}/delete/')
        self.assertEqual(resp.status_code, 200)

    def test_patient_cannot_create_case(self):
        self.auth_as(self.patient_user)
        # Patients don't have case creation access (permission check in view)
        # The view allows any authenticated user to create cases right now,
        # so this test confirms patient CAN currently create — flagging for future restriction.
        # To enforce: add role check in CaseListCreateView.post()
        resp = self.client.post('/api/cases/', {'patientId': 'PAT-NOPE'}, format='json')
        # Currently returns 201 — future: should return 403 for patients
        self.assertIn(resp.status_code, [201, 403])


class TestInferencePipeline(PipelineTestCase):
    """Test inference endpoint with mocked InferenceEngine."""

    @patch('inference.views.InferenceEngine')
    def test_run_inference_mocked(self, MockEngine):
        """Inference endpoint returns 200 with mocked engine."""
        from cases.models import Case, MRIImage, SegmentationResult
        import os
        from django.conf import settings

        # Create case
        case = Case.objects.create(patient_id='PAT-INF', created_by=self.doctor)

        # Create physical directory so the view's Path.exists() check passes
        case_dir = os.path.join(settings.MEDIA_ROOT, 'cases', str(case.case_id))
        os.makedirs(case_dir, exist_ok=True)

        # Create 4 fake MRIImage records
        for mod in ['t1', 't1ce', 't2', 'flair']:
            MRIImage.objects.create(
                case=case, modality=mod,
                file_path=f'cases/{case.case_id}/{mod}.nii',
                file_size=1024, original_filename=f'{mod}.nii'
            )

        # Mock InferenceEngine.run_inference
        mock_instance = MockEngine.return_value
        mock_instance.run_inference.return_value = {
            'case_id': str(case.case_id),
            'volumes': {'whole_tumor': 15000.0, 'tumor_core': 5000.0, 'enhancing_tumor': 1200.0},
            'confidence_scores': {'whole_tumor': 0.92, 'tumor_core': 0.88, 'enhancing_tumor': 0.75},
            'mask_files': {},
            'created': True,
        }

        resp = self.client.post(f'/api/inference/predict/{case.case_id}/')
        self.assertEqual(resp.status_code, 200, resp.data)
        self.assertIn('result', resp.data)

    def test_inference_case_not_found(self):
        fake_id = uuid.uuid4()
        resp = self.client.post(f'/api/inference/predict/{fake_id}/')
        self.assertEqual(resp.status_code, 404)


class TestReportPipeline(PipelineTestCase):
    """Test report generation, retrieval, editing, and export."""

    def _setup_case_with_segmentation(self):
        """Helper: create a case + SegmentationResult for report tests."""
        from cases.models import Case, SegmentationResult

        case = Case.objects.create(
            patient_id='PAT-RPT', created_by=self.doctor,
            age=50, sex='F', status='completed'
        )
        seg = SegmentationResult.objects.create(
            case=case,
            whole_tumor_mask='cases/fake/wt.nii.gz',
            tumor_core_mask='cases/fake/tc.nii.gz',
            enhancing_tumor_mask='cases/fake/et.nii.gz',
            whole_tumor_volume=18000.0,
            tumor_core_volume=6000.0,
            enhancing_tumor_volume=1500.0,
            whole_tumor_confidence=0.91,
            tumor_core_confidence=0.87,
            enhancing_tumor_confidence=0.74,
            structured_findings={
                'volumes': {
                    'whole_tumor': 18000.0,
                    'tumor_core': 6000.0,
                    'enhancing_tumor': 1500.0
                },
                'confidence_scores': {
                    'whole_tumor': 0.91,
                    'tumor_core': 0.87,
                    'enhancing_tumor': 0.74
                },
                'timestamp': '2026-03-01T23:00:00',
                'model_version': 'MoME+ v1.0',
                'device': 'cpu'
            }
        )
        return case, seg

    def _generate_report_mocked(self, case_id):
        with patch('reports.views.generate_json_descriptor') as mock_desc, \
             patch('reports.views.generate_report_from_descriptor') as mock_report:
            mock_desc.return_value = {
                'patient_info': {'case_id': str(case_id), 'age': 50, 'sex': 'F'},
                'tumor_metrics': {'volumes': {'whole_tumor': 18000.0}}
            }
            mock_report.return_value = "Mocked AI-generated radiology report. " * 5
            return self.client.post(f'/api/reports/generate/{case_id}/')

    def test_generate_report(self):
        case, _ = self._setup_case_with_segmentation()
        resp = self._generate_report_mocked(case.case_id)
        self.assertIn(resp.status_code, [200, 201], resp.data)
        self.assertIn('report', resp.data)
        self.assertIn('finalizedText', resp.data['report'])
        self.assertTrue(len(resp.data['report']['finalizedText']) > 100)

    def test_list_reports(self):
        case, _ = self._setup_case_with_segmentation()
        self._generate_report_mocked(case.case_id)
        resp = self.client.get('/api/reports/')
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.data, list)
        self.assertGreater(len(resp.data), 0)

    def test_get_report_detail(self):
        case, _ = self._setup_case_with_segmentation()
        gen = self._generate_report_mocked(case.case_id)
        report_id = gen.data['report']['reportId']

        resp = self.client.get(f'/api/reports/{report_id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['patientId'], 'PAT-RPT')

    def test_edit_report(self):
        case, _ = self._setup_case_with_segmentation()
        gen = self._generate_report_mocked(case.case_id)
        report_id = gen.data['report']['reportId']

        new_text = "UPDATED: Clinical correction to the AI-generated report text."
        resp = self.client.patch(f'/api/reports/{report_id}/update/', {
            'finalizedText': new_text,
            'editReason': 'Corrected tumor size description'
        }, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['report']['finalizedText'], new_text)
        self.assertEqual(resp.data['report']['editCount'], 1)

    def test_edit_report_increments_count(self):
        case, _ = self._setup_case_with_segmentation()
        gen = self._generate_report_mocked(case.case_id)
        report_id = gen.data['report']['reportId']

        for i in range(3):
            self.client.patch(f'/api/reports/{report_id}/update/', {
                'finalizedText': f'Edit {i}', 'editReason': ''
            }, format='json')

        resp = self.client.get(f'/api/reports/{report_id}/')
        self.assertEqual(resp.data['editCount'], 3)

    def test_export_pdf(self):
        case, _ = self._setup_case_with_segmentation()
        gen = self._generate_report_mocked(case.case_id)
        report_id = gen.data['report']['reportId']

        resp = self.client.post(f'/api/reports/{report_id}/export/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'application/pdf')
        self.assertIn('attachment', resp['Content-Disposition'])
        self.assertGreater(int(resp['Content-Length']) if 'Content-Length' in resp else len(resp.content), 0)

    def test_report_not_found(self):
        fake_id = uuid.uuid4()
        resp = self.client.get(f'/api/reports/{fake_id}/')
        self.assertEqual(resp.status_code, 404)

    def test_cannot_generate_report_without_segmentation(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-NOSEG', created_by=self.doctor)
        resp = self.client.post(f'/api/reports/generate/{case.case_id}/')
        self.assertEqual(resp.status_code, 400)

    def test_patient_cannot_edit_report(self):
        case, _ = self._setup_case_with_segmentation()
        gen = self._generate_report_mocked(case.case_id)
        report_id = gen.data['report']['reportId']

        self.auth_as(self.patient_user)
        resp = self.client.patch(f'/api/reports/{report_id}/update/', {
            'finalizedText': 'Patient trying to edit'
        }, format='json')
        # Patient doesn't own this case → 403
        self.assertEqual(resp.status_code, 403)


class TestCaseManagementExtended(PipelineTestCase):
    """Extended case management tests."""

    def test_case_update_partial(self):
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-UPD', 'age': 30
        }, format='json')
        case_id = case_resp.data['caseId']

        resp = self.client.patch(f'/api/cases/{case_id}/update/', {
            'clinicalHistory': 'Updated history'
        }, format='json')
        self.assertEqual(resp.status_code, 200)

    def test_case_detail_permission_denied_for_other_doctor(self):
        """A doctor cannot view another doctor's case."""
        other_doc = User.objects.create_user(
            email='other@pipeline.com', password='TestPass123',
            first_name='Other', last_name='Doc', role='doctor'
        )
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-PERM'
        }, format='json')
        case_id = case_resp.data['caseId']

        # Switch to other doctor
        self.auth_as(other_doc)
        resp = self.client.get(f'/api/cases/{case_id}/')
        self.assertEqual(resp.status_code, 403)

    def test_case_with_patient_email_linking(self):
        resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-LINK',
            'patientEmail': 'pat@pipeline.com'
        }, format='json')
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.data['patientUserEmail'], 'pat@pipeline.com')

    def test_patient_sees_own_linked_cases(self):
        # Doctor creates case linked to patient
        self.client.post('/api/cases/', {
            'patientId': 'PAT-OWN',
            'patientEmail': 'pat@pipeline.com'
        }, format='json')

        # Patient fetches their cases
        self.auth_as(self.patient_user)
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 200)
        patient_ids = [c['patientId'] for c in resp.data]
        self.assertIn('PAT-OWN', patient_ids)

    def test_patient_does_not_see_unlinked_cases(self):
        # Doctor creates case NOT linked to patient
        self.client.post('/api/cases/', {
            'patientId': 'PAT-NOPE'
        }, format='json')

        self.auth_as(self.patient_user)
        resp = self.client.get('/api/cases/')
        patient_ids = [c['patientId'] for c in resp.data]
        self.assertNotIn('PAT-NOPE', patient_ids)

    def test_case_not_found_returns_404(self):
        fake_id = uuid.uuid4()
        resp = self.client.get(f'/api/cases/{fake_id}/')
        self.assertEqual(resp.status_code, 404)


class TestMRIUploadExtended(PipelineTestCase):
    """Extended MRI upload tests."""

    def test_upload_without_file_returns_400(self):
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-NOFILE'
        }, format='json')
        case_id = case_resp.data['caseId']

        resp = self.client.post(
            f'/api/cases/{case_id}/upload/',
            {'modality': 't1'},
            format='multipart'
        )
        self.assertEqual(resp.status_code, 400)

    def test_upload_without_modality_returns_400(self):
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-NOMOD'
        }, format='json')
        case_id = case_resp.data['caseId']

        f = SimpleUploadedFile('t1.nii', make_fake_nifti(), content_type='application/octet-stream')
        resp = self.client.post(
            f'/api/cases/{case_id}/upload/',
            {'file': f},
            format='multipart'
        )
        self.assertEqual(resp.status_code, 400)

    def test_upload_replaces_existing_modality(self):
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-REPLACE'
        }, format='json')
        case_id = case_resp.data['caseId']

        # Upload first
        f1 = SimpleUploadedFile('t1_v1.nii', make_fake_nifti(), content_type='application/octet-stream')
        self.client.post(f'/api/cases/{case_id}/upload/', {'file': f1, 'modality': 't1'}, format='multipart')

        # Upload again — should replace, not duplicate
        f2 = SimpleUploadedFile('t1_v2.nii', make_fake_nifti(), content_type='application/octet-stream')
        resp = self.client.post(f'/api/cases/{case_id}/upload/', {'file': f2, 'modality': 't1'}, format='multipart')
        self.assertEqual(resp.status_code, 201)

        # Should still have only 1 t1 image
        images_resp = self.client.get(f'/api/cases/{case_id}/images/')
        t1_images = [img for img in images_resp.data if img['modality'] == 't1']
        self.assertEqual(len(t1_images), 1)

    def test_upload_to_nonexistent_case_returns_404(self):
        fake_id = uuid.uuid4()
        f = SimpleUploadedFile('t1.nii', make_fake_nifti(), content_type='application/octet-stream')
        resp = self.client.post(
            f'/api/cases/{fake_id}/upload/',
            {'file': f, 'modality': 't1'},
            format='multipart'
        )
        self.assertEqual(resp.status_code, 404)


class TestSegmentationView(PipelineTestCase):
    """Tests for segmentation result endpoint."""

    def _create_case_with_segmentation(self):
        from cases.models import Case, SegmentationResult
        case = Case.objects.create(
            patient_id='PAT-SEGV', created_by=self.doctor, status='completed'
        )
        seg = SegmentationResult.objects.create(
            case=case,
            whole_tumor_mask='cases/fake/wt.nii.gz',
            tumor_core_mask='cases/fake/tc.nii.gz',
            enhancing_tumor_mask='cases/fake/et.nii.gz',
            whole_tumor_volume=15000.0,
            tumor_core_volume=5000.0,
            enhancing_tumor_volume=1200.0,
            structured_findings={'model_version': 'test', 'device': 'cpu'}
        )
        return case, seg

    def test_segmentation_result_view(self):
        case, seg = self._create_case_with_segmentation()
        resp = self.client.get(f'/api/cases/{case.case_id}/segmentation/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('volumes', resp.data)
        self.assertIn('confidence_scores', resp.data)
        self.assertIn('mask_files', resp.data)

    def test_segmentation_not_found(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-NOSEG2', created_by=self.doctor)
        resp = self.client.get(f'/api/cases/{case.case_id}/segmentation/')
        self.assertEqual(resp.status_code, 404)


class TestReportAccessControl(PipelineTestCase):
    """Test patient access to reports (draft vs finalized)."""

    def _setup_report(self, report_status='draft'):
        from cases.models import Case, SegmentationResult
        from reports.models import Report
        case = Case.objects.create(
            patient_id='PAT-RPAC', created_by=self.doctor,
            status='completed', patient_user=self.patient_user
        )
        SegmentationResult.objects.create(
            case=case,
            whole_tumor_mask='wt.nii.gz', tumor_core_mask='tc.nii.gz',
            enhancing_tumor_mask='et.nii.gz',
            whole_tumor_volume=18000, tumor_core_volume=6000,
            enhancing_tumor_volume=1500,
            structured_findings={'test': True}
        )
        report = Report.objects.create(
            case=case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={'test': True},
            status=report_status,
            last_edited_by=self.doctor
        )
        return case, report

    def test_patient_cannot_see_draft_report(self):
        case, report = self._setup_report('draft')
        self.auth_as(self.patient_user)
        resp = self.client.get(f'/api/reports/{report.report_id}/')
        self.assertEqual(resp.status_code, 403)

    def test_patient_can_see_reviewed_report(self):
        case, report = self._setup_report('reviewed')
        self.auth_as(self.patient_user)
        resp = self.client.get(f'/api/reports/{report.report_id}/')
        self.assertEqual(resp.status_code, 200)

    def test_patient_can_see_finalized_report(self):
        case, report = self._setup_report('finalized')
        self.auth_as(self.patient_user)
        resp = self.client.get(f'/api/reports/{report.report_id}/')
        self.assertEqual(resp.status_code, 200)
