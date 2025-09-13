from django.urls import path
from . import views

app_name = 'movie_frontend'

urlpatterns = [
    path('accounts/signup/', views.signup, name='signup'),  #新增注册路由

    path('accounts/signup/', views.signup, name='signup'),
    path('', views.home, name='home'),
    path('movies/', views.movie_list, name='movie_list'),
    path('movies/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('choose-favorites/', views.choose_favorites, name='choose_favorites'),
    path('movies/<int:movie_id>/toggle_favorite/', views.toggle_favorite, name='toggle_favorite'),
]