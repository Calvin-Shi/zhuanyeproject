# films_recommender_system/views.py

from django.contrib.auth.models import User
from rest_framework import viewsets, permissions
from django_filters.rest_framework import DjangoFilterBackend
from .models import Movie, UserReview, Recommendation, UserProfile
from .serializers import MovieListSerializer, MovieDetailSerializer, UserReviewSerializer, RecommendationMovieSerializer
from rest_framework import generics
from rest_framework.permissions import AllowAny
from .serializers import RegisterSerializer
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
import logging
import numpy as np
from django.core.cache import cache
# --- 新增：导入余弦相似度计算工具 ---
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ... 其他视图 (RegisterView, MovieViewSet, UserReviewViewSet) 保持不变 ...
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]


class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Movie.objects.all().order_by('-release_year')
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['release_year', 'genres__name', 'language']

    def get_serializer_class(self):
        if self.action == 'retrieve': return MovieDetailSerializer
        return MovieListSerializer


class UserReviewViewSet(viewsets.ModelViewSet):
    serializer_class = UserReviewSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self): return UserReview.objects.filter(user=self.request.user)

    def perform_create(self, serializer): serializer.save(user=self.request.user)


# --- 混合式实时推荐 API 视图 (最终三层健壮版) ---
class RealtimeRecommendationView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, user_id, *args, **kwargs):
        logger.info(f"收到用户 '{user_id}' 的实时推荐请求")
        try:
            user = User.objects.get(username=user_id)
        except User.DoesNotExist:
            user = None

        model_assets = cache.get('recommendation_model_assets')
        if model_assets and user:
            try:
                response = self.get_content_based_recommendations(user, model_assets)
                return response
            except Exception as e:
                logger.warning(f"内容模型实时推断失败: {e}。将尝试备用方案。")

        if user:
            response = self.get_profile_based_recommendations(user)
            if response:
                return response

        return self.get_global_fallback_recommendations()

    def get_content_based_recommendations(self, user, model_assets):
        item_vectors = model_assets['item_vectors']
        item_map = model_assets['item_map']

        rec_profile = Recommendation.objects.get(user=user)
        favorite_movie_pks = list(rec_profile.favorite_movies.values_list('id', flat=True))
        if not favorite_movie_pks: raise ValueError("用户无喜好电影")

        favorite_movies_imdb_ids = list(
            Movie.objects.filter(pk__in=favorite_movie_pks).values_list('imdb_id', flat=True))
        favorite_indices = [item_map[imdb_id] for imdb_id in favorite_movies_imdb_ids if imdb_id in item_map]
        if not favorite_indices: raise ValueError("喜好电影均不在模型中")

        # 1. 计算用户的平均兴趣向量
        user_vector = item_vectors[favorite_indices].mean(axis=0).reshape(1, -1)

        # 2. 计算用户向量与所有电影向量的余弦相似度
        scores = cosine_similarity(user_vector, item_vectors)[0]

        # 3. 获取分数最高的电影的索引
        # argsort返回的是从小到大的索引，所以需要反转
        top_indices = np.argsort(scores)[::-1]

        # 4. 结果过滤
        favorite_imdb_set = set(favorite_movies_imdb_ids)
        # 反向映射现在可以直接从 item_map 创建
        reverse_item_map = {i: imdb_id for imdb_id, i in item_map.items()}
        recommended_imdb_ids = []
        for index in top_indices:
            imdb_id = reverse_item_map.get(index)
            if imdb_id and imdb_id not in favorite_imdb_set:
                recommended_imdb_ids.append(imdb_id)
            if len(recommended_imdb_ids) >= 50: break

        logger.info(f"成功为用户 '{user.username}' 生成基于内容的推荐。")
        return Response(
            {"user_id": user.username, "source": "content_based_inference", "recommendations": recommended_imdb_ids})

    def get_profile_based_recommendations(self, user):
        try:
            profile = UserProfile.objects.get(user=user)
            f_genres = profile.favorite_genres.all()
            if f_genres.exists():
                movies_qs = Movie.objects.filter(genres__in=f_genres).distinct().order_by('-truth_score')
                # 排除用户已经“喜欢”的电影
                liked_movies = Recommendation.objects.get(user=user).favorite_movies.all()
                movies_qs = movies_qs.exclude(pk__in=liked_movies)

                imdb_ids = list(movies_qs.values_list('imdb_id', flat=True)[:50])
                if imdb_ids:
                    logger.info(f"为用户 '{user.username}' 生成基于偏好类型的备用推荐。")
                    return Response({"user_id": user.username, "source": "cold_start_profile",
                                     "recommendations": [mid for mid in imdb_ids if mid]})
        except (UserProfile.DoesNotExist, Recommendation.DoesNotExist):
            pass
        return None

    def get_global_fallback_recommendations(self):
        logger.info("执行全局回退策略，返回真值分数最高的电影。")
        movies_qs = Movie.objects.order_by('-truth_score').values_list('imdb_id', flat=True).distinct()
        imdb_ids = list(movies_qs[:50])
        return Response({"user_id": "anonymous", "source": "cold_start_global",
                         "recommendations": [mid for mid in imdb_ids if mid]})


class BatchMovieDetailView(APIView):
    # ... (无修改) ...
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        imdb_ids_str = request.query_params.get('ids', '')
        if not imdb_ids_str: return Response({"error": "缺少 'ids' 查询参数。"}, status=status.HTTP_400_BAD_REQUEST)
        imdb_ids = imdb_ids_str.split(',')
        movies = Movie.objects.filter(imdb_id__in=imdb_ids).prefetch_related('titles')
        movies_dict = {movie.imdb_id: movie for movie in movies}
        sorted_movies = [movies_dict[imdb_id] for imdb_id in imdb_ids if imdb_id in movies_dict]
        serializer = RecommendationMovieSerializer(sorted_movies, many=True)
        return Response(serializer.data)