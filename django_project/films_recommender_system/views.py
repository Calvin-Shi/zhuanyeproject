from django.contrib.auth.models import User
from django.shortcuts import render
from rest_framework import  viewsets, permissions
from django_filters.rest_framework import  DjangoFilterBackend
from .models import Movie, UserReview
from .serializers import MovieListSerializer, MovieDetailSerializer,UserReviewSerializer
from rest_framework import generics
from rest_framework.permissions import AllowAny
from .serializers import RegisterSerializer

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]

class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Movie.objects.all().order_by('-release_year')
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['release_year', 'genres__name', 'language']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return MovieDetailSerializer
        return MovieListSerializer

class UserReviewViewSet(viewsets.ModelViewSet):
    serializer_class = UserReviewSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return UserReview.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
from django.core.cache import cache
import sys
import logging
import pandas as pd

# --- 实时推荐API实现 ---

# 动态地将算法脚本所在的目录添加到系统路径中
# 这是为了能够从该脚本中导入 HybridRecModel 类
project_root = settings.BASE_DIR.parent
algorithm_path = project_root / 'django_project' / 'films_recommender_system' / 'management' / 'commands'
if str(algorithm_path) not in sys.path:
    sys.path.insert(0, str(algorithm_path))

from generate_recommendations import HybridRecModel

logger = logging.getLogger(__name__)

# 核心函数：获取或训练模型
def get_or_train_model():
    """
    处理模型训练和缓存的核心逻辑。
    1. 首先尝试从缓存获取模型。
    2. 如果缓存中没有，则从数据库查询实时数据，训练一个新模型。
    3. 将训练好的新模型存入缓存，并设置24小时有效期。
    """
    model = cache.get('recommendation_model')
    if model is None:
        logger.info("缓存中未找到推荐模型，开始使用实时数据库数据进行训练...")
        try:
            # 步骤 1: 使用Django ORM查询实时数据
            logger.info("正在查询数据库中的用户评分和电影信息...")
            
            # 查询所有用户评分，并获取所需字段
            reviews_qs = UserReview.objects.filter(movie__imdb_id__isnull=False).select_related('user', 'movie')
            reviews_list = list(reviews_qs.values('user__username', 'movie__imdb_id', 'rating'))
            if not reviews_list:
                raise ValueError("数据库中没有用户评分数据，无法训练模型。")
            # 将查询结果转换为Pandas DataFrame，并重命名列以匹配算法输入格式
            reviews_df = pd.DataFrame(reviews_list).rename(columns={
                'user__username': 'user_id',
                'movie__imdb_id': 'item_id'
            })

            # 查询所有电影及其类型信息
            movies_qs = Movie.objects.prefetch_related('genres')
            movies_data = []
            for movie in movies_qs:
                # 算法需要电影类型是单个字符串，格式如: "科幻 / 犯罪"
                genre_str = " / ".join([g.name for g in movie.genres.all()])
                movies_data.append({'imdb_id': movie.imdb_id, 'genres': genre_str})
            if not movies_data:
                raise ValueError("数据库中没有电影数据。")
            movies_df = pd.DataFrame(movies_data)

            # 步骤 2: 实例化模型，并使用实时数据进行训练
            model = HybridRecModel()
            
            # 创建一个模拟的日志记录器，以捕获算法类中的训练进度信息
            class MockLogger:
                def write(self, msg):
                    logger.info(msg.strip())
                @property
                def style(self):
                    return self
                def SUCCESS(self, msg):
                    return msg

            # 调用为实时数据训练而设计的新方法
            model.train_from_dataframes(reviews_df, movies_df, stdout=MockLogger(), stderr=MockLogger())
            
            # 步骤 3: 将新训练好的模型存入缓存，有效期24小时
            logger.info("模型训练完成。正在将模型存入缓存，有效期24小时。")
            cache.set('recommendation_model', model, timeout=86400) # 86400 秒 = 24 小时
        except Exception as e:
            logger.error(f"训练或加载推荐模型失败: {e}", exc_info=True)
            return None
    else:
        logger.info("从缓存中成功加载推荐模型。")
    return model

# API视图：通过HTTP暴露推荐功能
class RealtimeRecommendationView(APIView):
    """
    用于为特定用户获取实时电影推荐的API视图。
    """
    permission_classes = [AllowAny] # 允许任何客户端访问此接口

    def get(self, request, user_id, *args, **kwargs):
        """处理对 /api/recommendations/realtime/<user_id>/ 的GET请求"""
        # 获取训练好的模型实例（来自缓存或新训练）
        model = get_or_train_model()
        if model is None:
            return Response({"error": "因训练错误，推荐模型当前不可用。"}, status=503)

        logger.info(f"收到为用户 {user_id} 的推荐请求")
        
        try:
            # 使用模型为指定用户生成推荐
            recommendations = model.get_recommendations_for_user(user_id, top_n=20)
            
            if not recommendations:
                return Response({"user_id": user_id, "recommendations": [], "message": "用户不存在或暂无推荐"}, status=404)

            return Response({"user_id": user_id, "recommendations": recommendations})
        except Exception as e:
            logger.error(f"为用户 {user_id} 生成推荐时出错: {e}", exc_info=True)
            return Response({"error": "生成推荐时发生内部错误。"}, status=500)
''
