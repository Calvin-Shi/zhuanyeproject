from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import MovieViewSet, UserReviewViewSet, RealtimeRecommendationView, BatchMovieDetailView # 导入新视图

# 创建一个路由器
router = DefaultRouter()

# 注册ViewSet
router.register(r'movies', MovieViewSet, basename='movies')
router.register(r'reviews', UserReviewViewSet, basename='userreview')

# 首先定义我们自己的URL，然后包含由router生成的URL
urlpatterns = [
    # 为实时推荐API添加的路由
    # 将 URL 'recommendations/realtime/<user_id>/' 映射到 RealtimeRecommendationView 视图
    path('recommendations/realtime/<str:user_id>/', RealtimeRecommendationView.as_view(), name='realtime-recommendations'),
    # MODIFIED: 使用新的批量获取URL，替换掉旧的单个获取URL
    path('movies/batch-details/', BatchMovieDetailView.as_view(), name='batch-movie-details'),
]

urlpatterns += router.urls
