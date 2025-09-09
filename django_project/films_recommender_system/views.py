from urllib import request

from django.contrib.auth.models import User
from django.shortcuts import render
from rest_framework import  viewsets, permissions
from rest_framework.authentication import SessionAuthentication
from django_filters.rest_framework import  DjangoFilterBackend
from .models import Movie, UserReview
from .serializers import MovieListSerializer, MovieDetailSerializer,UserReviewSerializer
from rest_framework import generics
from rest_framework.permissions import AllowAny
from .serializers import RegisterSerializer
from django.shortcuts import render, get_object_or_404

# ------------ DRF-API ------------

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    # 允许任何人访问注册接口
    permission_classes = [AllowAny]

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

# 访问权限类
class IsOwnerOrReadOnly(permissions.BasePermission):
    """只允许对象的所有者进行编辑"""

    def has_object_permission(self, request, view, obj):
        # 读取权限对所有人开放(GET, HEAD, OPTIONS 请求)
        if request.method in permissions.SAFE_METHODS:
            return True

        # 写入权限值开放给该评论的所有者
        return obj.user == request.user

class UserReviewViewSet(viewsets.ModelViewSet):
    """
    一个完整的ViewSet，允许用户创建、读取、更新、删除自己的评论
    - 读取(GET)某部电影的所有评论
    - 创建(POST)自己的新评论
    - 更新(PUT/PATCH)和删除(DELETE)自己的评论
    """

    authentication_classes = [SessionAuthentication]

    serializer_class = UserReviewSerializer
    # 权限设置，必须是已登录的用户才能访问此接口
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]

    def get_queryset(self):
        """返回与特定电影相关的所有评论"""
        # 从 URL 查询参数中获取 movie_id
        movie_id = self.request.query_params.get('movie_id')
        movie = Movie.objects.get(id=movie_id)
        if movie:
            print("movie here:", movie.original_title, movie.release_year)
            for item in UserReview.objects.filter(movie=movie).order_by('-timestamp'):
                print(item.user,item.timestamp)

            return UserReview.objects.filter(movie=movie).order_by('-timestamp')

        # 若没有提供 movie，返回一个空 queryset
        return UserReview.objects.none()

    def perform_create(self, serializer):
        # 在创建新的评论时，自动将当前登录用户关联上
        serializer.save(user=self.request.user)


# ------------ Django模板渲染 ------------

def movie_list_view(request):
    """渲染电影列表页"""

    movies = Movie.objects.all().order_by('-release_year')
    context = {
        'movies': movies,
    }

    return render(request, 'films_recommender_system/movie_list.html', context)

def movie_detail_view(request, movie_id):
    """
    渲染电影详情页
    'movie_id'是从 URL 捕获的电影主键 (ID)
    """

    movie = get_object_or_404(Movie, pk=movie_id)
    # 获取该电影的所有用户评论
    reviews = UserReview.objects.filter(movie=movie).order_by('-timestamp')

    context = {
        'movie': movie,
        'review': reviews
    }

    return render(request, 'films_recommender_system/movie_detail.html', context)

def register_page_view(request):
    """返回注册页面的HTML模板"""
    return render(request, 'films_recommender_system/register.html')