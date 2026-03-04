"""
Custom exception handler for DRF.
"""

from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    """
    Custom exception handler that provides consistent error responses.
    """
    # Call DRF's default exception handler first
    response = exception_handler(exc, context)
    
    # If DRF handled it, format the response
    if response is not None:
        error_data = {
            'success': False,
            'error': {
                'message': str(exc),
                'type': exc.__class__.__name__,
            }
        }
        
        if isinstance(response.data, dict):
            error_data['error']['details'] = response.data
        else:
            error_data['error']['details'] = response.data
        
        return Response(error_data, status=response.status_code)
    
    # Handle unexpected exceptions
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    
    return Response({
        'success': False,
        'error': {
            'message ': 'An unexpected error occurred.',
            'type': 'InternalServerError'
        }
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
