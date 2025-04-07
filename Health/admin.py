from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

# Register your models here.
class CustomUserAdmin(admin.ModelAdmin):
    model = CustomUser
    list_display = ('username', 'email', 'age', 'gender')  # Add fields you want to display
    search_fields = ('username', 'email')  # Search by username or email
    list_filter = ('age', 'gender')  # You can filter by age and gender if required

admin.site.register(CustomUser, CustomUserAdmin)