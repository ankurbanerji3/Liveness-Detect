from django.conf.urls import url, include
from django.urls import path
from .views import process_and_show, FileView

urlpatterns = [
    path('eye_blink', FileView.as_view())
]
