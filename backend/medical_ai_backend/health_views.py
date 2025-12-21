"""
Health check views for monitoring system status.
"""

from django.http import JsonResponse
from django.db import connection
from django.conf import settings
import redis


def health_check(request):
    """Basic health check endpoint."""
    return JsonResponse({
        'status': 'healthy',
        'service': 'Medical AI Backend',
        'version': settings.SPECTACULAR_SETTINGS.get('VERSION', 'v1')
    })


def system_status(request):
    """Detailed system status including database and Redis."""
    status = {
        'database': False,
        'redis': False,
        'debug_mode': settings.DEBUG,
    }
    
    # Check database connection
    try:
        connection.ensure_connection()
        status['database'] = True
    except Exception:
        pass
    
    # Check Redis connection
    try:
        r = redis.from_url(settings.CELERY_BROKER_URL)
        r.ping()
        status['redis'] = True
    except Exception:
        pass
    
    overall_status = 'healthy' if (status['database'] and status['redis']) else 'degraded'
    
    return JsonResponse({
        'status': overall_status,
        'components': status
    })
