from rest_framework import serializers
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
    # MODIFIED: 重命名为 display_title 以明确其通用性
    display_title = serializers.SerializerMethodField()

    class Meta:
        model = Movie
        fields = ['id', 'imdb_id', 'display_title', 'release_year', 'poster_url']

    # MODIFIED: 优化标题获取的降级逻辑
    def get_display_title(self, obj):
        # 预取相关的标题可以提升性能，但这需要在视图层完成
        # 如果没有预取，这将导致N+1查询
        titles = getattr(obj, '_prefetched_objects_cache', {}).get('titles', obj.titles.all())

        # 标记了中文主标题以及标题对应语言类型后，可启用下面的逻辑
        # primary_cn_title = next((t.title_text for t in titles if t.is_primary and 'zh' in t.language.lower()), None)
        # if primary_cn_title: return primary_cn_title
        #
        # any_primary_title = next((t.title_text for t in titles if t.is_primary), None)
        # if any_primary_title: return any_primary_title
        #
        # any_cn_title = next((t.title_text for t in titles if 'zh' in t.language.lower()), None)
        # if any_cn_title: return any_cn_title

        return obj.original_title


# NEW: 为推荐页面卡片创建的、极度轻量级的Serializer
class RecommendationMovieSerializer(MovieListSerializer):
    """只包含推荐页电影卡片所需的最少信息"""

    class Meta:
        model = Movie
        # 继承 MovieListSerializer 的字段，确保标题等核心信息存在
        fields = ['id', 'imdb_id', 'display_title', 'release_year', 'poster_url']


class MovieDetailSerializer(MovieListSerializer):
    """用于电影详情页的Serializer，继承自列表Serializer"""
    genres = GenreSerializer(many=True, read_only=True)
    directors = PersonSerializer(many=True, read_only=True)
    actors = PersonSerializer(many=True, read_only=True)
    scriptwriters = PersonSerializer(many=True, read_only=True)
    titles = MovieTitleSerializer(many=True, read_only=True)
    # Django默认的反向关系名是 'review_set'
    reviews = ReviewSerializer(many=True, read_only=True, source='review_set')

    class Meta(MovieListSerializer.Meta):
        # 继承父类的字段，并添加详情页专属字段
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