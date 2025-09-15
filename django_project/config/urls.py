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
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from films_recommender_system.views import RegisterView

from films_recommender_system.views import RegisterView
from testWeb import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # 前端应用的URL
    path('', include(('movie_frontend.urls', 'movie_frontend'), namespace='movie_frontend')),
    path('accounts/', include('django.contrib.auth.urls')),

    # 测试APP的URL
    path('index/', views.index),
    path('calPage', views.calPage),
    path('cal',views.calculate),
    path('list',views.calList),
    path('del',views.delData),

    # 将电影推荐系统APP的URL包含到主路由中，并为其分配一个命名空间
    path('api/', include('films_recommender_system.urls')),
    # 认证API路由
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]

# 仅在开发模式下提供媒体文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)