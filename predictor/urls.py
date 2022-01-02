from django.urls import path
from . import views
from . import viewsP
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
    path('pred_infectados_pais/', views.prediccion),
    path('pred_mortalidad_pais/', views.pred_mortalidad_pais),
    path('pred_casos_pais/', views.pred_casos_pais),
    path('pred_ultimo_primero/', views.pred_ultimo_primero),
    path('pred_mortalidad_depa/', views.pred_mortalidad_depa),
    path('pred_casos_muertes/', views.pred_casos_muertes),
    path('pred_casos_dia/', views.pred_casos_dia),





]
