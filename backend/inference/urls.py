"""
ML inference and continual learning URLs.
"""

from django.urls import path
from . import views

urlpatterns = [
    # New Inference Endpoints
    path('predict/<uuid:case_id>/', views.PredictSegmentationView.as_view(), name='predict_segmentation'),
    path('result/<uuid:case_id>/', views.GetSegmentationResultView.as_view(), name='get_result'),
    path('upload-ground-truth/<uuid:case_id>/', views.UploadGroundTruthView.as_view(), name='upload_ground_truth'),
    
    # Inference Tasks
    path('segment/', views.StartSegmentationView.as_view(), name='start_segmentation'),
    path('tasks/', views.TaskListView.as_view(), name='task_list'),
    path('tasks/<uuid:task_id>/', views.TaskDetailView.as_view(), name='task_detail'),
    path('tasks/<uuid:task_id>/cancel/', views.CancelTaskView.as_view(), name='cancel_task'),
    
    # Model Management
    path('models/', views.ModelVersionListView.as_view(), name='model_list'),
    path('models/<uuid:model_id>/', views.ModelVersionDetailView.as_view(), name='model_detail'),
    path('models/<uuid:model_id>/activate/', views.ActivateModelView.as_view(), name='activate_model'),
    
    # Continual Learning (Admin)
    path('continual-learning/', views.ContinualLearningListView.as_view(), name='cl_list'),
    path('continual-learning/start/', views.StartContinualLearningView.as_view(), name='start_cl'),
    path('continual-learning/<uuid:task_id>/', views.ContinualLearningDetailView.as_view(), name='cl_detail'),
]
