"""
Unit tests for health check views.
"""

from django.test import TestCase, Client


class TestHealthCheck(TestCase):
    """Tests for health check endpoints."""

    def setUp(self):
        self.client = Client()

    def test_health_endpoint_returns_healthy(self):
        resp = self.client.get('/api/health/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'Medical AI Backend')

    def test_health_endpoint_includes_version(self):
        resp = self.client.get('/api/health/')
        data = resp.json()
        self.assertIn('version', data)

    def test_system_status_endpoint(self):
        resp = self.client.get('/api/health/status/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('status', data)
        self.assertIn('components', data)
        self.assertIn('database', data['components'])
        # Database should be available since tests use sqlite
        self.assertTrue(data['components']['database'])
