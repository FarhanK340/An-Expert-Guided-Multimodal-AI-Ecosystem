"""
Stress tests using Python threading.
Tests concurrent API access and response time validation.
Marked with @pytest.mark.stress so they can be excluded from normal runs.
"""

import time
import threading
import pytest
from django.test import TransactionTestCase
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model

User = get_user_model()


@pytest.mark.stress
class TestConcurrentRegistration(TransactionTestCase):
    """Test concurrent user registrations don't corrupt data."""

    def test_concurrent_registrations(self):
        results = []
        errors = []

        def register(idx):
            client = APIClient()
            try:
                resp = client.post('/api/users/register/', {
                    'email': f'stress_reg_{idx}@test.com',
                    'password': 'StressTest123!',
                    'confirm_password': 'StressTest123!',
                    'first_name': f'Stress{idx}',
                    'last_name': 'Reg',
                    'role': 'doctor'
                }, format='json')
                results.append(resp.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors during concurrent registration: {errors}")
        self.assertEqual(results.count(201), 10, f"Not all registrations succeeded: {results}")
        self.assertEqual(User.objects.filter(email__startswith='stress_reg_').count(), 10)


@pytest.mark.stress
class TestConcurrentCaseCreation(TransactionTestCase):
    """Test concurrent case creation by a single user."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='stress_doc@test.com', password='TestPass123!',
            first_name='Stress', last_name='Doc', role='doctor'
        )

    def _get_token(self):
        client = APIClient()
        resp = client.post('/api/users/login/', {
            'email': 'stress_doc@test.com', 'password': 'TestPass123!'
        }, format='json')
        return resp.data['access']

    def test_concurrent_case_creation(self):
        token = self._get_token()
        results = []
        errors = []

        def create_case(idx):
            client = APIClient()
            client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
            try:
                resp = client.post('/api/cases/', {
                    'patientId': f'PAT-STRESS-{idx}'
                }, format='json')
                results.append(resp.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create_case, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(results.count(201), 10)


@pytest.mark.stress
class TestConcurrentCaseListing(TransactionTestCase):
    """Test concurrent case listing requests."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='stress_list@test.com', password='TestPass123!',
            first_name='Stress', last_name='List', role='doctor'
        )
        # Create some cases
        client = APIClient()
        resp = client.post('/api/users/login/', {
            'email': 'stress_list@test.com', 'password': 'TestPass123!'
        }, format='json')
        self.token = resp.data['access']
        client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.token}')
        for i in range(5):
            client.post('/api/cases/', {'patientId': f'PAT-LIST-{i}'}, format='json')

    def test_concurrent_listing(self):
        results = []

        def list_cases():
            client = APIClient()
            client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.token}')
            resp = client.get('/api/cases/')
            results.append(resp.status_code)

        threads = [threading.Thread(target=list_cases) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # All should succeed
        self.assertEqual(results.count(200), 20)


@pytest.mark.stress
class TestAPIResponseTime(TransactionTestCase):
    """Validate API response times are within acceptable limits."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='stress_time@test.com', password='TestPass123!',
            first_name='Time', last_name='Test', role='doctor'
        )
        self.client = APIClient()
        resp = self.client.post('/api/users/login/', {
            'email': 'stress_time@test.com', 'password': 'TestPass123!'
        }, format='json')
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {resp.data["access"]}')

    def test_login_response_time(self):
        client = APIClient()
        start = time.time()
        client.post('/api/users/login/', {
            'email': 'stress_time@test.com', 'password': 'TestPass123!'
        }, format='json')
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, f"Login took {elapsed:.2f}s — too slow")

    def test_case_list_response_time(self):
        # Create 20 cases first
        for i in range(20):
            self.client.post('/api/cases/', {'patientId': f'PAT-TIME-{i}'}, format='json')

        start = time.time()
        resp = self.client.get('/api/cases/')
        elapsed = time.time() - start
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 1.0, f"Case list took {elapsed:.2f}s — too slow")

    def test_health_check_response_time(self):
        from django.test import Client
        client = Client()
        start = time.time()
        resp = client.get('/api/health/')
        elapsed = time.time() - start
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 0.5, f"Health check took {elapsed:.2f}s — too slow")
