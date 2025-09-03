from django.shortcuts import render
from rest_framework import  viewsets, permissions
from django_filters.rest_framework import  DjangoFilterBackend
from .models import Movie, UserReview
from .serializers import MovieListSerializer, MovieDetailSerializer,UserReviewSerializer

class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    """
    一个只读的ViewSet，用于展示电影列表和详情
    只允许GET请求
    """
    queryset = Movie.objects.all().order_by('-release_year')    # 按上映年份降序排列

    # 过滤器、搜索、分页配置
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['release_year', 'genres__name', 'language']

    def get_serializer_class(self):
        # 根据不同动作返回不同的Serializer
        if self.action == 'retrieve':
            return MovieDetailSerializer
        return MovieListSerializer

class UserReviewViewSet(viewsets.ModelViewSet):
    """
    一个完整的ViewSet，允许用户创建、读取、更新、删除自己的评论
    """
    serializer_class = UserReviewSerializer
    # 权限设置，必须是已登录的用户才能访问此接口
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # 确保用户只能看到和操作自己的评论
        return UserReview.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # 在创建新的评论时，自动将当前登录用户关联上
        serializer.save(user=self.request.user)
