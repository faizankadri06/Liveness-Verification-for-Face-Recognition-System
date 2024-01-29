from django.urls import path
from . import views, face_recognition_liveness_app

urlpatterns = [
    path('',views.user_login, name="user_login"),
    path('Dashboard/',views.signin, name="signin"),
    path('admin_dash/',views.admin, name="admin"),
    path('video_feed/', views.Live, name='video_feed'),
    path('punch/', views.index, name='stream_video'),
]