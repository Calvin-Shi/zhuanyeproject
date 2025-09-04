
from django.db import models
from django.contrib.auth.models import User

class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text='电影类型')
    def __str__(self): return self.name

class Person(models.Model):
    name = models.CharField(max_length=200, unique=True, help_text='导演/编剧/演员姓名')
    def __str__(self): return self.name

class Movie(models.Model):
    original_title = models.CharField(max_length=200)
    release_year = models.IntegerField(null=True, blank=True)
    language = models.CharField(max_length=20, default='en')
    length = models.IntegerField(null=True, blank=True, help_text='片长(分钟)')
    summary = models.TextField(default="", blank=True)

    genres = models.ManyToManyField(Genre, blank=True, related_name='movies')
    directors = models.ManyToManyField(Person, blank=True, related_name='directed_movies')
    actors = models.ManyToManyField(Person, blank=True, related_name='acted_movies')
    scriptwriters = models.ManyToManyField(Person, blank=True, related_name='written_movies')

    @property
    def display_title(self):
        primary = self.titles.filter(is_primary=True).first()
        if primary:
            return primary.title_text
        any_title = self.titles.first()
        return any_title.title_text if any_title else self.original_title

    def __str__(self):
        return f"{self.display_title} ({self.release_year or ''})".strip()

class MovieTitle(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='titles')
    title_text = models.CharField(max_length=200)
    language = models.CharField(max_length=10, default='zh-CN')
    is_primary = models.BooleanField(default=False)
    def __str__(self): return f"{self.movie_id}: {self.title_text} ({self.language})"

class Source(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text='数据来源网站')
    url = models.URLField(blank=True)
    def __str__(self): return self.name

class Review(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='reviews')
    source = models.ForeignKey(Source, on_delete=models.CASCADE, related_name='reviews')
    score = models.FloatField(null=True, blank=True)
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.source.name}→{self.movie.display_title}: {self.score}"

class UserReview(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.IntegerField()
    review = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} → {self.movie.display_title}: {self.rating}"

class Recommendation(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    recommended_movies = models.ManyToManyField(Movie, related_name='recommendation', blank=True)
    last_update = models.DateTimeField(auto_now=True)
    def __str__(self): return f"Recommendation for {self.user.username}"
