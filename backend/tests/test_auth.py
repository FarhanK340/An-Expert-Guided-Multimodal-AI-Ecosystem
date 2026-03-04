"""
Django API test suite — Authentication & Role-Based Access.
Tests registration, login, JWT tokens, and role-based permissions.
"""

import uuid
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()


class AuthTestCase(TestCase):
    """Base helper that creates common test users."""

    def setUp(self):
        self.client = APIClient()

        # Create users for each role
        self.doctor = User.objects.create_user(
            email='doctor@test.com', password='TestPass123',
            first_name='Alice', last_name='Smith', role='doctor'
        )
        self.patient = User.objects.create_user(
            email='patient@test.com', password='TestPass123',
            first_name='Bob', last_name='Jones', role='patient'
        )
        self.admin = User.objects.create_user(
            email='admin@test.com', password='TestPass123',
            first_name='Carol', last_name='Admin', role='admin',
            is_staff=True
        )
        self.researcher = User.objects.create_user(
            email='researcher@test.com', password='TestPass123',
            first_name='Dave', last_name='Researcher', role='researcher'
        )

    def auth_as(self, user):
        """Authenticate client as the given user via JWT login."""
        resp = self.client.post('/api/users/login/', {
            'email': user.email, 'password': 'TestPass123'
        }, format='json')
        self.assertEqual(resp.status_code, 200, f"Login failed for {user.email}: {resp.data}")
        token = resp.data['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        return token


class TestRegistration(AuthTestCase):
    """Test user registration for all roles."""

    def test_doctor_registration(self):
        resp = self.client.post('/api/users/register/', {
            'email': 'newdoc@test.com', 'password': 'TestPass123',
            'confirm_password': 'TestPass123',
            'first_name': 'New', 'last_name': 'Doctor', 'role': 'doctor',
            'specialty': 'Neurology', 'institution': 'City Hospital'
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertIn('access', resp.data)
        self.assertEqual(resp.data['user']['role'], 'doctor')

    def test_patient_registration(self):
        resp = self.client.post('/api/users/register/', {
            'email': 'newpatient@test.com', 'password': 'TestPass123',
            'confirm_password': 'TestPass123',
            'first_name': 'New', 'last_name': 'Patient', 'role': 'patient'
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(resp.data['user']['role'], 'patient')

    def test_registration_password_mismatch(self):
        resp = self.client.post('/api/users/register/', {
            'email': 'x@test.com', 'password': 'TestPass123',
            'confirm_password': 'WrongPass',
            'first_name': 'X', 'last_name': 'Y', 'role': 'doctor'
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_registration_duplicate_email(self):
        resp = self.client.post('/api/users/register/', {
            'email': 'doctor@test.com', 'password': 'TestPass123',
            'confirm_password': 'TestPass123',
            'first_name': 'Dup', 'last_name': 'User', 'role': 'doctor'
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class TestLogin(AuthTestCase):
    """Test login and JWT token flow."""

    def test_doctor_login(self):
        resp = self.client.post('/api/users/login/', {
            'email': 'doctor@test.com', 'password': 'TestPass123'
        }, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('access', resp.data)
        self.assertIn('refresh', resp.data)
        self.assertEqual(resp.data['user']['role'], 'doctor')

    def test_patient_login(self):
        resp = self.client.post('/api/users/login/', {
            'email': 'patient@test.com', 'password': 'TestPass123'
        }, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['user']['role'], 'patient')

    def test_wrong_password(self):
        resp = self.client.post('/api/users/login/', {
            'email': 'doctor@test.com', 'password': 'WrongPassword'
        }, format='json')
        self.assertEqual(resp.status_code, 401)

    def test_nonexistent_user(self):
        resp = self.client.post('/api/users/login/', {
            'email': 'nobody@test.com', 'password': 'TestPass123'
        }, format='json')
        self.assertEqual(resp.status_code, 401)

    def test_token_refresh(self):
        login = self.client.post('/api/users/login/', {
            'email': 'doctor@test.com', 'password': 'TestPass123'
        }, format='json')
        refresh = login.data['refresh']
        resp = self.client.post('/api/users/refresh/', {'refresh': refresh}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('access', resp.data)


class TestProfileAccess(AuthTestCase):
    """Test authenticated profile access."""

    def test_get_profile_authenticated(self):
        self.auth_as(self.doctor)
        resp = self.client.get('/api/users/profile/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['role'], 'doctor')

    def test_get_profile_unauthenticated(self):
        resp = self.client.get('/api/users/profile/')
        self.assertEqual(resp.status_code, 401)

    def test_update_profile(self):
        self.auth_as(self.doctor)
        resp = self.client.patch('/api/users/profile/update/', {
            'specialty': 'Neuro-Oncology', 'institution': 'General Hospital'
        }, format='json')
        self.assertEqual(resp.status_code, 200)


class TestRoleBasedAccess(AuthTestCase):
    """Test that admin-only endpoints reject non-admin users."""

    def test_admin_user_list(self):
        self.auth_as(self.admin)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('users', resp.data)

    def test_doctor_cannot_access_user_list(self):
        self.auth_as(self.doctor)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 403)

    def test_patient_cannot_access_user_list(self):
        self.auth_as(self.patient)
        resp = self.client.get('/api/users/users/')
        self.assertEqual(resp.status_code, 403)
