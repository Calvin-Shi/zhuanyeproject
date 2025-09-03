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
from testWeb import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # 测试APP的URL
    path('index/', views.index),
    path('calPage', views.calPage),
    path('cal',views.calculate),
    path('list',views.calList),
    path('del',views.delData),
    # 将电影推荐系统APP的URL包含到主路由中，并为其分配一个命名空间
    path('api/', include('films_recommender_system.urls'))
]
