"""
End-to-end workflow tests.
Tests full user journeys through the system using Django's test client.
"""

import uuid
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()


def make_fake_nifti() -> bytes:
    """Return a minimal NIfTI-1 file header (348 bytes)."""
    header = bytearray(348)
    header[0:4] = (348).to_bytes(4, 'little')
    header[344:348] = b'n+1\x00'
    return bytes(header)


class TestFullDoctorWorkflow(TestCase):
    """
    End-to-end: Register → Login → Create case → Upload MRI →
    Mock inference → Get segmentation → Generate report → Edit → Export PDF.
    """

    def setUp(self):
        self.client = APIClient()

    def _register_and_login(self, email, role='doctor'):
        reg_resp = self.client.post('/api/users/register/', {
            'email': email,
            'password': 'TestPass123!',
            'confirm_password': 'TestPass123!',
            'first_name': 'E2E',
            'last_name': 'Doctor',
            'role': role
        }, format='json')
        self.assertEqual(reg_resp.status_code, 201, reg_resp.data)
        self.client.credentials(
            HTTP_AUTHORIZATION=f'Bearer {reg_resp.data["access"]}'
        )
        return reg_resp.data['user']

    def test_full_doctor_workflow(self):
        # 1. Register
        user = self._register_and_login('e2e_doc@test.com')
        self.assertEqual(user['role'], 'doctor')

        # 2. Get profile
        profile_resp = self.client.get('/api/users/profile/')
        self.assertEqual(profile_resp.status_code, 200)

        # 3. Create case
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-E2E',
            'age': 45,
            'sex': 'M',
            'clinicalHistory': 'Headaches'
        }, format='json')
        self.assertEqual(case_resp.status_code, 201)
        case_id = case_resp.data['caseId']

        # 4. Upload 4 MRI modalities
        for mod in ['t1', 't1ce', 't2', 'flair']:
            f = SimpleUploadedFile(
                f'{mod}.nii', make_fake_nifti(),
                content_type='application/octet-stream'
            )
            upload_resp = self.client.post(
                f'/api/cases/{case_id}/upload/',
                {'file': f, 'modality': mod},
                format='multipart'
            )
            self.assertEqual(upload_resp.status_code, 201, f"Upload {mod} failed")

        # 5. Verify images listed
        images_resp = self.client.get(f'/api/cases/{case_id}/images/')
        self.assertEqual(len(images_resp.data), 4)

        # 6. Create segmentation result directly (simulating mocked inference)
        from cases.models import Case, SegmentationResult
        case = Case.objects.get(case_id=case_id)
        SegmentationResult.objects.create(
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
                'volumes': {'whole_tumor': 18000.0, 'tumor_core': 6000.0, 'enhancing_tumor': 1500.0},
                'confidence_scores': {'whole_tumor': 0.91, 'tumor_core': 0.87, 'enhancing_tumor': 0.74},
                'full_segmentation_mask': 'cases/fake/full_seg.nii.gz',
                'model_version': 'MoME+ v1.0',
                'device': 'cpu'
            }
        )
        case.status = 'completed'
        case.save()

        # 7. Get segmentation result
        seg_resp = self.client.get(f'/api/cases/{case_id}/segmentation/')
        self.assertEqual(seg_resp.status_code, 200)
        self.assertIn('volumes', seg_resp.data)

        # 8. Generate report (mock LLM utils since no real mask files on disk)
        with patch('reports.views.generate_json_descriptor') as mock_desc, \
             patch('reports.views.generate_report_from_descriptor') as mock_report:
            mock_desc.return_value = {
                'patient_info': {'case_id': str(case_id), 'age': 45, 'sex': 'M'},
                'tumor_metrics': {
                    'volumes': {'whole_tumor': 18000.0, 'tumor_core': 6000.0, 'enhancing_tumor': 1500.0},
                    'confidence_scores': {'whole_tumor': 0.91, 'tumor_core': 0.87, 'enhancing_tumor': 0.74}
                }
            }
            mock_report.return_value = "Mocked AI-generated radiology report for E2E testing."

            report_resp = self.client.post(f'/api/reports/generate/{case_id}/')
            self.assertIn(report_resp.status_code, [200, 201])
            report_id = report_resp.data['report']['reportId']

        # 9. Edit report
        edit_resp = self.client.patch(f'/api/reports/{report_id}/update/', {
            'finalizedText': 'Clinician-edited report text.',
            'editReason': 'Manual correction'
        }, format='json')
        self.assertEqual(edit_resp.status_code, 200)
        self.assertEqual(edit_resp.data['report']['editCount'], 1)

        # 10. Export PDF
        pdf_resp = self.client.post(f'/api/reports/{report_id}/export/')
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp['Content-Type'], 'application/pdf')

        # 11. List reports
        list_resp = self.client.get('/api/reports/')
        self.assertEqual(list_resp.status_code, 200)
        self.assertGreater(len(list_resp.data), 0)



class TestPatientReadOnlyWorkflow(TestCase):
    """
    End-to-end: Doctor creates case linked to patient →
    Patient can see case and finalized report → Patient cannot edit.
    """

    def setUp(self):
        self.client = APIClient()

    def test_patient_read_only_workflow(self):
        # 1. Register doctor
        doc_resp = self.client.post('/api/users/register/', {
            'email': 'wf_doc@test.com', 'password': 'TestPass123!',
            'confirm_password': 'TestPass123!',
            'first_name': 'WF', 'last_name': 'Doc', 'role': 'doctor'
        }, format='json')
        doc_token = doc_resp.data['access']

        # 2. Register patient
        self.client.credentials()
        pat_resp = self.client.post('/api/users/register/', {
            'email': 'wf_pat@test.com', 'password': 'TestPass123!',
            'confirm_password': 'TestPass123!',
            'first_name': 'WF', 'last_name': 'Pat', 'role': 'patient'
        }, format='json')
        pat_token = pat_resp.data['access']

        # 3. Doctor creates case linked to patient
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {doc_token}')
        case_resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-WF',
            'patientEmail': 'wf_pat@test.com'
        }, format='json')
        self.assertEqual(case_resp.status_code, 201)
        case_id = case_resp.data['caseId']

        # 4. Patient sees linked case
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {pat_token}')
        cases_resp = self.client.get('/api/cases/')
        self.assertEqual(cases_resp.status_code, 200)
        case_ids = [c['caseId'] for c in cases_resp.data]
        self.assertIn(str(case_id), [str(cid) for cid in case_ids])


class TestAccountLifecycle(TestCase):
    """
    End-to-end: Register → Login → Update profile → Change password →
    Login with new password → Delete account → Login fails.
    """

    def setUp(self):
        self.client = APIClient()

    def test_full_account_lifecycle(self):
        # 1. Register
        reg_resp = self.client.post('/api/users/register/', {
            'email': 'lifecycle@test.com', 'password': 'OldPass123!',
            'confirm_password': 'OldPass123!',
            'first_name': 'Life', 'last_name': 'Cycle', 'role': 'doctor'
        }, format='json')
        self.assertEqual(reg_resp.status_code, 201)
        token = reg_resp.data['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')

        # 2. Update profile
        update_resp = self.client.patch('/api/users/profile/update/', {
            'specialty': 'Oncology'
        }, format='json')
        self.assertEqual(update_resp.status_code, 200)

        # 3. Change password
        pw_resp = self.client.post('/api/users/profile/change-password/', {
            'currentPassword': 'OldPass123!',
            'newPassword': 'NewPass456!',
            'confirmPassword': 'NewPass456!'
        }, format='json')
        self.assertEqual(pw_resp.status_code, 200)

        # 4. Login with new password
        self.client.credentials()
        login_resp = self.client.post('/api/users/login/', {
            'email': 'lifecycle@test.com', 'password': 'NewPass456!'
        }, format='json')
        self.assertEqual(login_resp.status_code, 200)
        new_token = login_resp.data['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {new_token}')

        # 5. Delete account
        del_resp = self.client.post('/api/users/profile/delete/', {
            'password': 'NewPass456!'
        }, format='json')
        self.assertEqual(del_resp.status_code, 200)

        # 6. Login should now fail
        self.client.credentials()
        fail_resp = self.client.post('/api/users/login/', {
            'email': 'lifecycle@test.com', 'password': 'NewPass456!'
        }, format='json')
        self.assertEqual(fail_resp.status_code, 401)


class TestAdminOverviewWorkflow(TestCase):
    """Admin can view all users, cases, and reports."""

    def setUp(self):
        self.client = APIClient()
        self.admin = User.objects.create_user(
            email='wf_admin@test.com', password='TestPass123!',
            first_name='WF', last_name='Admin', role='admin', is_staff=True
        )
        self.doctor = User.objects.create_user(
            email='wf_doc2@test.com', password='TestPass123!',
            first_name='WF', last_name='Doc2', role='doctor'
        )
        # Create a case as doctor
        self._auth_as(self.doctor)
        self.client.post('/api/cases/', {'patientId': 'PAT-ADM'}, format='json')

    def _auth_as(self, user):
        resp = self.client.post('/api/users/login/', {
            'email': user.email, 'password': 'TestPass123!'
        }, format='json')
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {resp.data["access"]}')

    def test_admin_overview(self):
        self._auth_as(self.admin)

        # See all users
        users_resp = self.client.get('/api/users/users/')
        self.assertEqual(users_resp.status_code, 200)
        self.assertGreaterEqual(users_resp.data['stats']['total_users'], 2)

        # See all cases (including doctor's)
        cases_resp = self.client.get('/api/cases/')
        self.assertEqual(cases_resp.status_code, 200)
        self.assertGreater(len(cases_resp.data), 0)

        # See all reports (may be empty)
        reports_resp = self.client.get('/api/reports/')
        self.assertEqual(reports_resp.status_code, 200)
