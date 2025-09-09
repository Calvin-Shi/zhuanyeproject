from rest_framework import serializers
from rest_framework.exceptions import ValidationError
from .models import Movie, MovieTitle, Genre, Person, Source, Review, UserReview
from django.contrib.auth.models import User

class RegisterSerializer(serializers.ModelSerializer):
    """用于用户注册的Serializer"""
    class Meta:
        model = User
        fields = ('username', 'password', 'email')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            email=validated_data.get('email', '')
        )
        return user


# ------ 辅助的、用于嵌套的 Serializer ------

class GenreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Genre
        fields = ['name']

class PersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Person
        fields = ['name']

class MovieTitleSerializer(serializers.ModelSerializer):
    class Meta:
        model = MovieTitle
        fields = ['title_text', 'language', 'is_primary']

class ReviewSerializer(serializers.ModelSerializer):
    source = serializers.CharField(source='source.name', read_only=True)
    score_max = serializers.FloatField(source='source.score_max', read_only=True)

    class Meta:
        model = Review
        fields = ['source', 'author', 'content', 'score', 'score_max']

# ------ 核心的、用于API端点的Serializer ------

class MovieListSerializer(serializers.ModelSerializer):
    """用于电影列表页的、轻量级的Serializer"""
    primary_cn_title = serializers.SerializerMethodField()

    class Meta:
        model = Movie
        fields = ['imdb_id', 'primary_cn_title', 'release_year']

    def get_primary_cn_title(self,obj):
        primary_cn_title_obj = obj.titles.filter(is_primary=True).first()
        if primary_cn_title_obj:
            return primary_cn_title_obj.title_text
        any_title_obj = obj.titles.filter().first()
        return any_title_obj.title_text if any_title_obj else obj.original_title

class MovieDetailSerializer(MovieListSerializer):
    """用于电影详情页的Serializer，继承自列表Serializer"""
    # 添加关联字段
    genres = GenreSerializer(many=True, read_only=True)
    directors = PersonSerializer(many=True, read_only=True)
    actors = PersonSerializer(many=True, read_only=True)
    scriptwriters = PersonSerializer(many=True, read_only=True)

    titles = MovieTitleSerializer(many=True, read_only=True)
    reviews = ReviewSerializer(many=True, read_only=True, source='reviews_set')

    class Meta(MovieListSerializer.Meta):
        fields = MovieListSerializer.Meta.fields + [
            'titles', 'genres', 'language', 'length',
            'directors', 'actors', 'scriptwriters',
            'summary', 'reviews'
        ]

class UserReviewSerializer(serializers.ModelSerializer):
    """用于处理用户评分和评论的Serializer"""
    user = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = UserReview
        fields = ['user', 'movie', 'rating', 'review', 'timestamp']

    def validate(self, data):
        user = self.context['request'].user
        movie = data.get('movie')

        if self.instance:
            return data

        if user.is_authenticated and UserReview.objects.filter(user=user, movie=movie).exists():
            raise ValidationError("您已经对这部电影进行过评价，请勿重复提交。")

        return data
