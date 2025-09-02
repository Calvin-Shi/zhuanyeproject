from django.db import models
from django.contrib.auth.models import User

# ------ 核心实体模型：存储电影有关信息 ------

# 电影类型
class Genre(models.Model):
    name = models.CharField(max_length=100,unique=True,help_text='电影类型(e.g. 科幻，动作)')
    def __str__(self): return self.name

# 导演/编剧/演员姓名
class Person(models.Model):
    name = models.CharField(max_length=200,unique=True,help_text='导演/编剧/演员姓名')
    def __str__(self): return self.name

# 电影信息
class Movie(models.Model):
    """
    电影的核心实体，描述性信息（标题、评分）通过外键关联到它上面
    """
    imdb_id = models.CharField(max_length=20,unique=True,null=True,blank=True,help_text='IMDb ID')
    original_title = models.CharField(max_length=200, default='', db_index=True, help_text="电影的原始语言标题")
    language = models.CharField(max_length=10,default='zh-CN',help_text='语言代码(e.g. zh-CN, en)')
    release_year = models.IntegerField(null=True, blank=True, help_text='上映年份')
    length = models.IntegerField(null=True, blank=True, help_text='片长')
    summary = models.TextField(null=True, blank=True,help_text='剧情简介')

    class Meta:
        # 为original_title和release_year添加一个联合唯一约束，以此作为电影唯一标识(优先使用IMDb ID)
        unique_together = ('original_title', 'release_year')

    genres = models.ManyToManyField(Genre, blank=True)
    directors = models.ManyToManyField(Person, related_name='directed_movies', blank=True)
    actors = models.ManyToManyField(Person, related_name='acted_in_movies', blank=True)
    scriptwriters = models.ManyToManyField(Person, related_name='script_write_for_movies', blank=True)

    def __str__(self):
        # 尝试获取主流中文译名，如果主流中文译名不存在，则返回一个任意的其他中文译名，都不存在则返回电影原始标题
        primary_title = self.titles.filter(is_primary=True).first()
        if primary_title:
            return f"{primary_title.title_text} ({self.release_year})"

        any_title = self.titles.first()
        if any_title:
            return f"{any_title.title_text} ({self.release_year})"

        return f"Movie: {self.original_title}"

# 电影名称
class MovieTitle(models.Model):
    """
    存储电影可能存在的多个中文译名，与 Movie 模型是一对多的关系
    """
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE,related_name='titles',help_text='关联的电影')
    title_text = models.CharField(max_length=200,help_text='标题内容')
    language = models.CharField(max_length=10,default='zh-CN',help_text='语言代码(e.g. zh-CN, en)')
    is_primary = models.BooleanField(default=False,help_text='是否为用于显示的主标题')

    def __str__(self):
        return f"{self.movie.id}: {self.title_text} ({self.language})"

# ------ 真值评估模型 ------

# 信息源
class Source(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text='数据来源网站(e.g. IMDb，豆瓣)')
    base_url = models.URLField(help_text='网站主页链接')
    credibility_level = models.IntegerField(default=5, help_text='预设的可信度等级(1-10)，用于加权')
    score_max = models.FloatField(default=10, help_text='该评分体系满分值，用于后续进行归一化处理')

    def __str__(self): return self.name

# 评价信息
class Review(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, help_text='关联的电影')
    source = models.ForeignKey(Source, on_delete=models.CASCADE, help_text='评论来源')

    author = models.CharField(max_length=200, null=True, blank=True, help_text='评论作者')
    content = models.TextField(help_text='评论内容')
    score = models.FloatField(null=True, blank=True, help_text='电影评分')
    score_max = models.FloatField(default=10, help_text='该评分体系满分值，用于后续进行归一化处理')

    def __str__(self):
        return f"Review for {self.movie.titles.filter(is_primary=True).first()} from {self.source.name}"

# ------ 用户行为模型 ------

# 用户评价
class UserReview(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, help_text='评分用户')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, help_text='评价的电影')
    rating = models.IntegerField(help_text='用户评分(e.g. 1-5)')
    review = models.TextField(help_text='用户评论')
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 确保一个用户对一部电影只能有一个评分
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username}'s rating for {self.movie.titles.filter(is_primary=True).first()}: {self.rating}"

# ------ 推荐结果模型 ------

# 推荐信息
class Recommendation(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    recommended_movies = models.ManyToManyField(Movie, related_name='recommendation', blank=True)
    last_update = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Recommendation for {self.user.username}"