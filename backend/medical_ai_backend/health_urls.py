"""
Health check URLs for system monitoring.
"""

from django.urls import path
from . import health_views

urlpatterns = [
    path('', health_views.health_check, name='health-check'),
    path('status/', health_views.system_status, name='system-status'),
]
