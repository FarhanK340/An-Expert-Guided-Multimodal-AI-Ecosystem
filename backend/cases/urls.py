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
]
