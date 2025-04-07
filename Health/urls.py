# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('mental_health/', views.mental_health, name='mental_health'),
    path('login/', views.login_view, name='login'),  
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('analysis/', views.analysis, name='analysis'),
    path('process-assessment/', views.process_assessment, name='process_assessment'),
    
]