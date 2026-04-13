from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_page, name='login'),
    path('verify/', views.verify_face, name='verify_face'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_page, name='register'),
    path('add-student/', views.add_student, name='add_student'),
]
