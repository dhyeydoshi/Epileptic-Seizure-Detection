"""
URL configuration for API endpoints.
Version 2.0 - API routes.
"""
from django.urls import path
from . import api

app_name = 'api'

urlpatterns = [
    path('predict/', api.predict_api, name='predict'),
    path('models/', api.models_list_api, name='models_list'),
    path('health/', api.health_check, name='health_check'),
]

