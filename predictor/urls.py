from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

# Url condig
urlpatterns = [
    path('upload/', views.upload),
    path('', views.home),
    path('tendencia_infeccion_pais/', views.tendencia_infeccion_pais),
    path('tendencia_infecxdia_pais/', views.tendencia_infecxdia_pais),
    path('tendencia_vacuancion_pais/', views.tendencia_vacuancion_pais),
    path('tendencia_casos_depa/', views.tendencia_casos_depa),


]
