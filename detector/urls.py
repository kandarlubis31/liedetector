# detector/urls.py
from django.urls import path
from . import views  

urlpatterns = [
    path('', views.detector_view, name='detector'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('start_analysis/', views.start_analysis, name='start_analysis'),
    path('get_results/', views.get_results, name='get_results'),
]