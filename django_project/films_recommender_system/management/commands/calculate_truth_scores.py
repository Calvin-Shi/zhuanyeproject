# films_recommender_system/management/commands/calculate_truth_scores.py

from django.core.management.base import BaseCommand
from django.db.models import Avg, Count
from films_recommender_system.models import Movie, Review, UserReview
from tqdm import tqdm
import math


class Command(BaseCommand):
    help = 'Calculates and updates a global truth score for each movie.'

    def handle(self, *args, **options):
        self.stdout.write("正在为所有电影计算真值分数...")

        # 获取所有电影以便进行迭代
        all_movies = list(Movie.objects.all())

        for movie in tqdm(all_movies, desc="计算分数"):
            total_score = 0
            total_weight = 0

            # 1. 处理外部源的评论 (Review 模型)
            external_reviews = Review.objects.filter(movie=movie, score__isnull=False).select_related('source')
            for review in external_reviews:
                # 将所有评分归一化到10分制
                normalized_score = (review.score / review.source.score_max) * 10
                # 使用源的可信度作为权重
                weight = review.source.credibility_level
                total_score += normalized_score * weight
                total_weight += weight

            # 2. 处理本站用户的评论 (UserReview 模型)
            user_reviews_stats = UserReview.objects.filter(movie=movie, rating__isnull=False).aggregate(
                avg_rating=Avg('rating'),
                count=Count('id')
            )

            if user_reviews_stats['avg_rating'] is not None and user_reviews_stats['count'] > 0:
                user_score = user_reviews_stats['avg_rating']
                review_count = user_reviews_stats['count']

                # 为本站评论定义一个基础权重和一个流行度因子
                base_weight = 7  # 高度信任我们自己的用户
                # 使用对数函数使评分数量的影响随着增长而减弱
                # 这可以防止一部有数千条评论的电影完全主导结果
                count_factor = min(math.log10(review_count + 1) * 2, 3)  # 权重加成上限为3

                weight = base_weight + count_factor
                total_score += user_score * weight
                total_weight += weight

            # 3. 计算最终的加权平均分
            if total_weight > 0:
                final_score = total_score / total_weight
                movie.truth_score = round(final_score, 2)
            else:
                # 如果没有任何评分，分数为0
                movie.truth_score = 0

        # 4. 为了性能，一次性批量更新所有电影
        Movie.objects.bulk_update(all_movies, ['truth_score'])

        self.stdout.write(self.style.SUCCESS(f"成功为 {len(all_movies)} 部电影更新了真值分数。"))