"""
Case management URLs.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Case CRUD
    path('', views.CaseListCreateView.as_view(), name='case_list_create'),
    path('<uuid:case_id>/', views.CaseDetailView.as_view(), name='case_detail'),
    path('<uuid:case_id>/update/', views.CaseUpdateView.as_view(), name='case_update'),
    path('<uuid:case_id>/delete/', views.CaseDeleteView.as_view(), name='case_delete'),

    # MRI Image Upload
    path('<uuid:case_id>/upload/', views.MRIImageUploadView.as_view(), name='mri_upload'),
    path('<uuid:case_id>/images/', views.MRIImageListView.as_view(), name='mri_list'),
    
    # Segmentation Results
    path('<uuid:case_id>/segmentation/', views.SegmentationResultView.as_view(), name='segmentation_result'),
    path('<uuid:case_id>/visualizations/', views.VisualizationListView.as_view(), name='visualization_list'),
    
    # Feedback
    path('<uuid:case_id>/feedback/', views.FeedbackCreateView.as_view(), name='feedback_create'),
    path('feedback/', views.FeedbackListView.as_view(), name='feedback_list'),
]
