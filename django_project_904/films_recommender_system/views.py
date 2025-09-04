
from django.contrib.auth.models import User
from rest_framework import viewsets, permissions, generics
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.permissions import AllowAny

from .models import Movie, UserReview
from .serializers import (
    RegisterSerializer,
    MovieListSerializer, MovieDetailSerializer,
    UserReviewSerializer
)

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]

class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    """
    /api/movies/ 列表；/api/movies/{id}/ 详情
    """
    queryset = Movie.objects.all().prefetch_related(
        'titles', 'genres', 'directors', 'actors', 'scriptwriters', 'reviews'
    ).order_by('-release_year')

    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['release_year', 'genres__name', 'directors__name']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return MovieDetailSerializer
        return MovieListSerializer

class UserReviewViewSet(viewsets.ModelViewSet):
    """已登录用户的评分/评论 CRUD"""
    serializer_class = UserReviewSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return UserReview.objects.filter(user=self.request.user).select_related('movie')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
