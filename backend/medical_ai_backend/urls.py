"""
Main URL configuration for Medical AI Backend.
Routes to API endpoints for all features.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
]

admin.site.site_header = "Medical AI Admin"
admin.site.site_title = "Medical AI"
admin.site.index_title = "Expert-Guided Multimodal AI System"

urlpatterns += [
    # API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    
    # API Endpoints (matching frontend /api/users/, /api/cases/, etc.)
    path('api/users/', include('users.urls')),
    path('api/cases/', include('cases.urls')),
    path('api/reports/', include('reports.urls')),
    path('api/inference/', include('inference.urls')),
    
    # Health Check
    path('api/health/', include('medical_ai_backend.health_urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
