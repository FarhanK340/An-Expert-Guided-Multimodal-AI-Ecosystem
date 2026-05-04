"""
Role-Based Access Control (RBAC) matrix tests.
Systematically tests every major endpoint against every role
(doctor, patient, admin, researcher, unauthenticated).
"""

import uuid
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()


class RBACTestCase(TestCase):
    """Base setup with users for every role."""

    def setUp(self):
        self.client = APIClient()
        self.doctor = User.objects.create_user(
            email='doc@rbac.com', password='TestPass123',
            first_name='Dr', last_name='RBAC', role='doctor'
        )
        self.patient = User.objects.create_user(
            email='pat@rbac.com', password='TestPass123',
            first_name='Pat', last_name='RBAC', role='patient'
        )
        self.admin = User.objects.create_user(
            email='admin@rbac.com', password='TestPass123',
            first_name='Admin', last_name='RBAC', role='admin',
            is_staff=True
        )
        self.researcher = User.objects.create_user(
            email='res@rbac.com', password='TestPass123',
            first_name='Res', last_name='RBAC', role='researcher'
        )
        # Pre-create a case owned by the doctor
        self._auth_as(self.doctor)
        resp = self.client.post('/api/cases/', {
            'patientId': 'PAT-RBAC'
        }, format='json')
        self.case_id = resp.data['caseId']
        self.client.credentials()  # reset

    def _auth_as(self, user):
        resp = self.client.post('/api/users/login/', {
            'email': user.email, 'password': 'TestPass123'
        }, format='json')
        self.assertEqual(resp.status_code, 200, f"Login failed: {resp.data}")
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {resp.data["access"]}')

    def _clear_auth(self):
        self.client.credentials()


# ═══════════════════════════════════════════════════════════════════════
# Unauthenticated access
# ═══════════════════════════════════════════════════════════════════════

class TestUnauthenticatedAccess(RBACTestCase):
    """All protected endpoints must return 401 for unauthenticated requests."""

    def test_cases_list_unauthenticated(self):
        self._clear_auth()
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 401)

    def test_case_detail_unauthenticated(self):
        self._clear_auth()
        resp = self.client.get(f'/api/cases/{self.case_id}/')
        self.assertEqual(resp.status_code, 401)

    def test_profile_unauthenticated(self):
        self._clear_auth()
        resp = self.client.get('/api/users/profile/')
        self.assertEqual(resp.status_code, 401)

    def test_reports_list_unauthenticated(self):
        self._clear_auth()
        resp = self.client.get('/api/reports/')
        self.assertEqual(resp.status_code, 401)

    def test_inference_predict_unauthenticated(self):
        self._clear_auth()
        resp = self.client.post(f'/api/inference/predict/{self.case_id}/')
        self.assertEqual(resp.status_code, 401)

    def test_user_list_unauthenticated(self):
        self._clear_auth()
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 401)


# ═══════════════════════════════════════════════════════════════════════
# Doctor access
# ═══════════════════════════════════════════════════════════════════════

class TestDoctorAccess(RBACTestCase):
    """Doctor: full access to own cases, no admin endpoints."""

    def test_doctor_can_list_own_cases(self):
        self._auth_as(self.doctor)
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 200)

    def test_doctor_can_view_own_case(self):
        self._auth_as(self.doctor)
        resp = self.client.get(f'/api/cases/{self.case_id}/')
        self.assertEqual(resp.status_code, 200)

    def test_doctor_can_create_case(self):
        self._auth_as(self.doctor)
        resp = self.client.post('/api/cases/', {'patientId': 'PAT-NEW'}, format='json')
        self.assertEqual(resp.status_code, 201)

    def test_doctor_can_delete_own_case(self):
        self._auth_as(self.doctor)
        resp = self.client.delete(f'/api/cases/{self.case_id}/delete/')
        self.assertEqual(resp.status_code, 200)

    def test_doctor_cannot_access_user_list(self):
        self._auth_as(self.doctor)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 403)

    def test_doctor_can_view_reports(self):
        self._auth_as(self.doctor)
        resp = self.client.get('/api/reports/')
        self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════════════════════
# Patient access
# ═══════════════════════════════════════════════════════════════════════

class TestPatientAccess(RBACTestCase):
    """Patient: read-only access to linked cases/reports."""

    def test_patient_can_list_cases(self):
        self._auth_as(self.patient)
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 200)

    def test_patient_cannot_view_unlinked_case(self):
        self._auth_as(self.patient)
        resp = self.client.get(f'/api/cases/{self.case_id}/')
        self.assertEqual(resp.status_code, 403)

    def test_patient_cannot_delete_case(self):
        self._auth_as(self.patient)
        resp = self.client.delete(f'/api/cases/{self.case_id}/delete/')
        self.assertEqual(resp.status_code, 403)

    def test_patient_cannot_access_user_list(self):
        self._auth_as(self.patient)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 403)

    def test_patient_can_list_reports(self):
        self._auth_as(self.patient)
        resp = self.client.get('/api/reports/')
        self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════════════════════
# Admin access
# ═══════════════════════════════════════════════════════════════════════

class TestAdminAccess(RBACTestCase):
    """Admin: full access to everything."""

    def test_admin_can_list_all_cases(self):
        self._auth_as(self.admin)
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_can_view_any_case(self):
        self._auth_as(self.admin)
        resp = self.client.get(f'/api/cases/{self.case_id}/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_can_access_user_list(self):
        self._auth_as(self.admin)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_can_list_all_reports(self):
        self._auth_as(self.admin)
        resp = self.client.get('/api/reports/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_can_delete_any_user(self):
        self._auth_as(self.admin)
        resp = self.client.delete(f'/api/users/users/{self.researcher.pk}/')
        self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════════════════════
# Researcher access
# ═══════════════════════════════════════════════════════════════════════

class TestResearcherAccess(RBACTestCase):
    """Researcher: similar to doctor but for research workflows."""

    def test_researcher_can_list_own_cases(self):
        self._auth_as(self.researcher)
        resp = self.client.get('/api/cases/')
        self.assertEqual(resp.status_code, 200)

    def test_researcher_can_create_case(self):
        self._auth_as(self.researcher)
        resp = self.client.post('/api/cases/', {'patientId': 'PAT-RES'}, format='json')
        self.assertEqual(resp.status_code, 201)

    def test_researcher_cannot_access_user_list(self):
        self._auth_as(self.researcher)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 403)

    def test_researcher_cannot_view_other_doctor_case(self):
        self._auth_as(self.researcher)
        resp = self.client.get(f'/api/cases/{self.case_id}/')
        self.assertEqual(resp.status_code, 403)
