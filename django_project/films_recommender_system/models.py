# films_recommender_system/models.py

from django.db import models
from django.contrib.auth.models import User


# ------ 核心实体模型：存储电影有关信息 ------

# 电影类型
class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text='电影类型(e.g. 科幻，动作)')

    def __str__(self): return self.name


# 导演/编剧/演员姓名
class Person(models.Model):
    name = models.CharField(max_length=200, unique=True, db_index=True,  help_text='导演/编剧/演员姓名')

    def __str__(self): return self.name


# 电影信息
class Movie(models.Model):
    """
    电影的核心实体，描述性信息（标题、评分）通过外键关联到它上面
    """
    # 电影基础信息
    imdb_id = models.CharField(max_length=20, unique=True, null=True, blank=True, help_text='IMDb ID')
    original_title = models.CharField(max_length=200, default='', db_index=True, help_text="电影的原始语言标题")
    language = models.CharField(max_length=10, default='zh-CN', help_text='语言代码(e.g. zh-CN, en)')
    release_year = models.IntegerField(null=True, blank=True, help_text='上映年份')
    length = models.IntegerField(null=True, blank=True, help_text='片长')
    summary = models.TextField(null=True, blank=True, help_text='剧情简介')
    genres = models.ManyToManyField(Genre, blank=True)
    directors = models.ManyToManyField(Person, related_name='directed_movies', blank=True)
    actors = models.ManyToManyField(Person, related_name='acted_in_movies', blank=True)
    scriptwriters = models.ManyToManyField(Person, related_name='script_write_for_movies', blank=True)

    # 电影海报和剧照图链接
    poster_url = models.URLField(null=True, blank=True, help_text='电影海报链接')
    backdrop_url = models.URLField(null=True, blank=True, help_text='电影剧照图链接')

    # --- 新增：用于冷启动推荐的真值分数 ---
    truth_score = models.FloatField(default=0.0, db_index=True, help_text="根据多源评分计算的全局真值分数")

    # 电影标识
    class Meta:
        # 为original_title和release_year添加一个联合唯一约束，以此作为电影唯一标识(优先使用IMDb ID)
        unique_together = ('original_title', 'release_year')

    def __str__(self):
        # 尝试获取主流中文译名，如果主流中文译名不存在，则返回电影原始标题
        primary_title = self.titles.filter(is_primary=True).first()
        if primary_title:
            return f"{primary_title.title_text} ({self.release_year})"

        return f"Movie: {self.original_title}({self.release_year})"


# 电影名称
class MovieTitle(models.Model):
    """
    存储电影可能存在的多个译名，与 Movie 模型是一对多的关系
    """
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='titles', help_text='关联的电影')
    title_text = models.CharField(max_length=200, db_index=True, help_text='标题内容')
    language = models.CharField(max_length=10, default='zh-CN', help_text='语言代码(e.g. zh-CN, en)')
    is_primary = models.BooleanField(default=False, help_text='是否为用于显示的主标题')

    def __str__(self):
        return f"movie({self.movie.imdb_id}): {self.title_text} ({self.movie.release_year})"


# ------ 真值评估模型 ------

# 信息源
class Source(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text='数据来源网站(e.g. IMDb，豆瓣)')
    base_url = models.URLField(help_text='网站主页链接')
    credibility_level = models.IntegerField(default=5, help_text='预设的可信度等级(1-10)，用于加权')
    score_max = models.FloatField(default=5, help_text='该评分体系满分值，用于后续进行归一化处理')

    def __str__(self): return self.name


# 电影评价信息
class Review(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, help_text='关联的电影')
    source = models.ForeignKey(Source, on_delete=models.CASCADE, help_text='评论来源')

    author = models.CharField(max_length=200, null=True, blank=True, help_text='评论作者')
    content = models.TextField(help_text='评论内容')
    score = models.FloatField(null=True, blank=True, help_text='电影评分')
    score_max = models.FloatField(default=10, help_text='该评分体系满分值，用于后续进行归一化处理')
    content_date = models.DateTimeField(null=True, blank=True, help_text='评论时间')
    approvals_num = models.IntegerField(null=True, blank=True, help_text='该评论被赞同数')

    def __str__(self):
        primary_title = self.movie.titles.filter(is_primary=True).first()
        movie_title = primary_title.title_text if primary_title else "Unknown Movie"
        return f"Review for {movie_title} from {self.source.name}"


# ------ 用户行为模型 ------
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    nickname = models.CharField(max_length=100, blank=True, null=True, help_text='用户昵称')
    bio = models.TextField(blank=True, null=True, help_text='个人简介')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True, help_text='用户头像')
    profile_background = models.ImageField(upload_to='backgrounds/', null=True, blank=True, help_text='个人中心背景图')

    # 收藏/待看列表
    watchlist = models.ManyToManyField(Movie, related_name='watchlisted_by', blank=True)

    # --- 用户的偏好 ---
    favorite_people = models.ManyToManyField(Person, related_name='favorited_by_users', blank=True,
                                             help_text="用户喜欢的演员/导演")
    favorite_genres = models.ManyToManyField(Genre, related_name='favorited_by_users', blank=True,
                                             help_text="用户喜欢的电影类型")

    def __str__(self):
        return f"{self.user.username}的资料"


# 真值电影推荐网站用户评价信息
class UserReview(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews', help_text='评分用户')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='user_reviews', help_text='评价的电影')
    rating = models.FloatField(help_text='用户评分 (e.g. 1.0-10.0)')
    review = models.TextField(help_text='用户评论')
    timestamp = models.DateTimeField(auto_now_add=True)
    likes_count = models.IntegerField(default=0, help_text='该评论被赞同数')

    # 记录哪些用户赞同了这条评论
    liked_by = models.ManyToManyField(User, related_name='liked_reviews', blank=True)

    class Meta:
        # 按时间倒序排列评论
        ordering = ['-timestamp']

    def __str__(self):
        primary_title = self.movie.titles.filter(is_primary=True).first()
        movie_title = primary_title.title_text if primary_title else self.movie.original_title
        return f"{self.user.username}'s rating for {movie_title}: {self.rating}"


# 用户浏览历史模型
class BrowsingHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='browsing_history')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    viewed_on = models.DateTimeField(auto_now=True, help_text="最近浏览时间")

    class Meta:
        unique_together = ('user', 'movie')  # 一个用户对一个电影只有一条历史记录，时间会自动更新
        ordering = ['-viewed_on']

    def __str__(self):
        return f"{self.user.username} viewed {self.movie} on {self.viewed_on}"


# ------ 推荐结果模型 ------

# 推荐信息
class Recommendation(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

    # 用户的“喜欢”列表，作为算法的输入
    favorite_movies = models.ManyToManyField(Movie, related_name='favorited_by', blank=True)

    # 存储由算法生成的推荐电影ID列表
    recommended_movie_ids = models.JSONField(null=True, blank=True, help_text="存储由算法生成的推荐电影ID列表")

    last_update = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Recommendation settings for {self.user.username}"