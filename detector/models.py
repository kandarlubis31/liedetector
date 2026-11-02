from django.db import models
from django.contrib.auth.models import User  # Add this import

class FaceAnalysisResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    blink_count = models.IntegerField(default=0)
    smile_intensity = models.FloatField(default=0)
    gaze_direction = models.JSONField()  # {'x': float, 'y': float}
    video_path = models.FilePathField(path='media/videos/')
    lie_probability = models.FloatField(default=0)