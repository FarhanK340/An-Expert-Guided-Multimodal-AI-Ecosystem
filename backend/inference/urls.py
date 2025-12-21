"""
ML inference and continual learning URLs.
"""

from django.urls import path
from . import views

urlpatterns = [
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
