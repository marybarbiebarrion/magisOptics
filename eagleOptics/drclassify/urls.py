from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Root path
    path('upload/', views.predict_dr, name='image_upload'),
]
