from django.urls import path
from . import views

app_name = 'movie_frontend'

urlpatterns = [
    path('accounts/signup/', views.signup, name='signup'),
    path('', views.home, name='home'),
    path('movies/', views.movie_list, name='movie_list'),
    path('movies/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('choose-favorites/', views.choose_favorites, name='choose_favorites'),
    path('movies/<int:movie_id>/toggle_favorite/', views.toggle_favorite, name='toggle_favorite'),
    path('movies/<int:movie_id>/toggle_watchlist/', views.toggle_watchlist, name='toggle_watchlist'),
    path('profile/', views.profile, name='profile'),

    # --- 新增评论相关URL ---
    path('review/<int:review_id>/edit/', views.edit_review, name='edit_review'),
    path('review/<int:review_id>/delete/', views.delete_review, name='delete_review'),

    # --- 新增点赞URL ---
    path('review/<int:review_id>/like/', views.like_review, name='like_review'),

    # --- 新增：统一的、处理偏好切换的AJAX接口 ---
    path('favorites/toggle_entity/', views.toggle_favorite_entity, name='toggle_favorite_entity'),

    # --- 新增：用于AJAX刷新的URL ---
    path('refresh-popular-movies/', views.refresh_popular_movies, name='refresh_popular_movies'),
]