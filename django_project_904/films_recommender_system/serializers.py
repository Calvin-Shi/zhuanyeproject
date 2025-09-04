
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Movie, MovieTitle, Genre, Person, Source, Review, UserReview

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('username', 'password', 'email')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            email=validated_data.get('email', '')
        )

class GenreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Genre
        fields = ['id', 'name']

class PersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Person
        fields = ['id', 'name']

class MovieTitleSerializer(serializers.ModelSerializer):
    class Meta:
        model = MovieTitle
        fields = ['id', 'title_text', 'language', 'is_primary']

class ReviewSerializer(serializers.ModelSerializer):
    source = serializers.StringRelatedField()
    class Meta:
        model = Review
        fields = ['id', 'source', 'score', 'content', 'created_at']

class MovieListSerializer(serializers.ModelSerializer):
    display_title = serializers.CharField(read_only=True)
    release_year = serializers.IntegerField()
    class Meta:
        model = Movie
        fields = ['id', 'display_title', 'release_year']

class MovieDetailSerializer(serializers.ModelSerializer):
    display_title = serializers.CharField(read_only=True)
    titles = MovieTitleSerializer(many=True, read_only=True)
    genres = GenreSerializer(many=True, read_only=True)
    directors = PersonSerializer(many=True, read_only=True)
    actors = PersonSerializer(many=True, read_only=True)
    scriptwriters = PersonSerializer(many=True, read_only=True)
    reviews = ReviewSerializer(many=True, read_only=True)

    class Meta:
        model = Movie
        fields = [
            'id', 'display_title', 'original_title', 'release_year', 'language', 'length',
            'summary', 'titles', 'genres', 'directors', 'actors', 'scriptwriters', 'reviews'
        ]

class UserReviewSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')
    class Meta:
        model = UserReview
        fields = ['user', 'movie', 'rating', 'review', 'timestamp']
