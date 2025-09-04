from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from films_recommender_system.views import RegisterView

urlpatterns = [
    # 管理后台
    path('admin/', admin.site.urls),

    # 后端 API
    path('api/', include('films_recommender_system.urls')),
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # 前端页面（首页、电影列表等）
    path('', include(('movie_frontend.urls', 'movie_frontend'), namespace='movie_frontend')),

    # Django 自带登录/退出
    path('accounts/', include('django.contrib.auth.urls')),
]
