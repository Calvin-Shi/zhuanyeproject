# films_recommender_system/management/commands/check_rec_data.py

from django.core.management.base import BaseCommand
from films_recommender_system.models import UserReview, Recommendation, BrowsingHistory
from django.contrib.auth.models import User

# 关键：从训练脚本中直接导入标准，确保一致性
try:
    from films_recommender_system.management.commands.generate_recommendations import MIN_INTERACTIONS, \
        MIN_UNIQUE_MOVIES
except ImportError:
    # 如果由于某种原因找不到，提供一个默认值，避免崩溃
    MIN_INTERACTIONS = 50
    MIN_UNIQUE_MOVIES = 20


class Command(BaseCommand):
    help = 'Checks if the interaction data meets the minimum requirements to train the recommendation model.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.HTTP_INFO("--- 推荐模型数据健康检查 ---"))

        # 1. 统计总交互数
        review_count = UserReview.objects.count()
        favorite_count = Recommendation.favorite_movies.through.objects.count()
        history_count = BrowsingHistory.objects.count()
        total_interactions = review_count + favorite_count + history_count

        # 2. 统计被交互过的唯一电影数
        movies_with_reviews = UserReview.objects.values_list('movie_id', flat=True).distinct()
        movies_with_favorites = Recommendation.favorite_movies.through.objects.values_list('movie_id',
                                                                                           flat=True).distinct()
        movies_with_history = BrowsingHistory.objects.values_list('movie_id', flat=True).distinct()

        interacted_movie_ids = set(movies_with_reviews) | set(movies_with_favorites) | set(movies_with_history)
        unique_interacted_movies = len(interacted_movie_ids)

        # 3. 打印报告
        self.stdout.write("\n指标 1: 总交互数量")
        self.stdout.write(
            f"  - 当前总数: {total_interactions} (评分: {review_count}, 喜欢: {favorite_count}, 浏览: {history_count})")
        self.stdout.write(f"  - 最低要求: {MIN_INTERACTIONS}")

        pass_interactions = total_interactions >= MIN_INTERACTIONS
        if pass_interactions:
            self.stdout.write(self.style.SUCCESS("  - 状态: 达标 (PASS)"))
        else:
            self.stdout.write(
                self.style.ERROR(f"  - 状态: 不足 (FAIL) - 还需 {MIN_INTERACTIONS - total_interactions} 条"))

        self.stdout.write("\n指标 2: 唯一交互电影数")
        self.stdout.write(f"  - 当前总数: {unique_interacted_movies}")
        self.stdout.write(f"  - 最低要求: {MIN_UNIQUE_MOVIES}")

        pass_movies = unique_interacted_movies >= MIN_UNIQUE_MOVIES
        if pass_movies:
            self.stdout.write(self.style.SUCCESS("  - 状态: 达标 (PASS)"))
        else:
            self.stdout.write(
                self.style.ERROR(f"  - 状态: 不足 (FAIL) - 还需 {MIN_UNIQUE_MOVIES - unique_interacted_movies} 部"))

        # 4. 最终结论
        self.stdout.write(self.style.HTTP_INFO("\n--- 结论 ---"))
        if pass_interactions and pass_movies:
            self.stdout.write(
                self.style.SUCCESS("恭喜！数据已达到最低要求，可以运行 'generate_recommendations' 来训练核心推荐模型。"))
        else:
            self.stdout.write(self.style.WARNING(
                "数据量仍显不足。建议登录不同测试账号，进行更多的“喜欢”、“评分”和“浏览”操作来丰富数据。"))

        self.stdout.write("-" * 20)