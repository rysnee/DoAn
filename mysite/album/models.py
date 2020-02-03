from django.db import models


class Photo(models.Model):
    file = models.ImageField()
    description = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'photo'
        verbose_name_plural = 'photos'

class ResultPhoto(models.Model):
    index = models.CharField(max_length=255, blank=True)
    score = models.FloatField(blank=True, default=0)
    
    class Meta:
        verbose_name = 'result_photo'
        verbose_name_plural = 'result_photos'