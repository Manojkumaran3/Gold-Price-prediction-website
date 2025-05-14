from django.db import models

# Create your models here.
from django.db import models

class Prediction(models.Model):
    prediction_date = models.DateField()
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.prediction_date}: ${self.predicted_price}"

class AboutUs(models.Model):
    # For example
    description = models.TextField()




class User(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    phone = models.CharField(max_length=15)

    def __str__(self):
        return self.email
    
class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15, blank=True)
    service = models.CharField(max_length=50, blank=True)
    email = models.EmailField()
    message = models.TextField()

    def __str__(self):
        return f"{self.name} - {self.email}"
