from django.urls import path
from . import views

app_name = 'movie_frontend'

urlpatterns = [
    path('', views.home, name='home'),                     # 首页
    path('movies/', views.movie_list, name='movie_list'),  # 电影列表
    path('movies/<int:movie_id>/', views.movie_detail, name='movie_detail'),  # 详情页
    path('recommendations/', views.recommendations, name='recommendations'), # 推荐页
]
