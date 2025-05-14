from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import AboutUs, ContactMessage

admin.site.register(AboutUs)#  Register your models here.
# admin.py

admin.site.register(ContactMessage)