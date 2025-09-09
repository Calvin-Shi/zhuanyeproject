"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

from films_recommender_system.views import (
    RegisterView,
    movie_list_view,
    movie_detail_view,
    register_page_view,
)

from films_recommender_system.views import RegisterView
from testWeb import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # 测试APP的URL
    path('index/', views.index),
    path('calPage', views.calPage),
    path('cal',views.calculate),
    path('list',views.calList),
    path('del',views.delData),

    # --- DRF-API路由 ---
    # 将电影推荐系统APP的URL包含到主路由中，并为其分配一个命名空间
    path('api/', include('films_recommender_system.urls')),
    # 认证API路由
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # --- Web Page URLs ---
    path('',movie_list_view, name='movie_list'),
    path('movies/<int:movie_id>/', movie_detail_view, name='movie_detail'),

    # --- 用户认证 URLs ---
    # 使用Django内置的LoginView
    path('login/', auth_views.LoginView.as_view(template_name='films_recommender_system/login.html'), name='login'),
    # 使用Django内置的LogioutView
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    # 注册页面
    path('register/', register_page_view, name='register_page'),
]
