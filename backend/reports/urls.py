"""
Report generation and management URLs.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Report Management
    path('', views.ReportListView.as_view(), name='report_list'),
    path('<uuid:report_id>/', views.ReportDetailView.as_view(), name='report_detail'),
    path('<uuid:report_id>/update/', views.ReportUpdateView.as_view(), name='report_update'),
    
    # Report Generation
    path('generate/<uuid:case_id>/', views.GenerateReportView.as_view(), name='generate_report'),
    
    # PDF Export
    path('<uuid:report_id>/export/', views.ExportPDFView.as_view(), name='export_pdf'),
    path('<uuid:report_id>/pdfs/', views.PDFListView.as_view(), name='pdf_list'),
    
    # Traceability
    path('<uuid:report_id>/traceability/', views.TraceabilityView.as_view(), name='traceability'),
    
    # Templates (Admin)
    path('templates/', views.TemplateListView.as_view(), name='template_list'),
    path('templates/<int:pk>/', views.TemplateDetailView.as_view(), name='template_detail'),
]
