from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

# Url condig
urlpatterns = [
    path('predict_model/', views.predict_model),
    path('upload/', views.upload),
    path('', views.home),
    path('tendencia_infeccion_pais/', views.predict_model),
]
