from django.contrib import admin
from django.urls import path
from . import views # .은 같은 디렉토리 의미 여기선 foods

urlpatterns = [
    #path('index/', views.index), #foods안에 views에서 index모델을 가져오라는 뜻
    #path('menu/<str:food>/', views.food_detail), #동적url

    path('menu/', views.index),
    path('menu/<int:pk>/', views.baekho_detail),
]