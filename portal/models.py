from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100, unique=True)
    sid = models.CharField(max_length=20, unique=True)
    image_file = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.sid})"

