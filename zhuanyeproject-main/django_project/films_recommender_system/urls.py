from rest_framework.routers import DefaultRouter
from .views import MovieViewSet, UserReviewViewSet

# 创建一个路由器
router = DefaultRouter()

# 注册ViewSet
router.register(r'movies', MovieViewSet, basename='movies')
router.register(r'reviews', UserReviewViewSet, basename='userreview')

# 包含由router生成的所有URL
urlpatterns = router.urls