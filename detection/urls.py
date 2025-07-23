from django.urls import path
from .views import detect_objects

urlpatterns = [
    path('detect/', detect_objects),
]
