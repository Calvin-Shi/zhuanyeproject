from django.db import models

# Create your models here.
class cal(models.Model):
    value1 = models.CharField(max_length = 10)
    value2 = models.CharField(max_length = 10)
    result = models.CharField(max_length = 10)

