# films_recommender_system/management/commands/generate_recommendations.py

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
from scipy.sparse import coo_matrix
import implicit
from django.core.management.base import BaseCommand
from django.core.cache import cache
from films_recommender_system.models import Movie, UserReview, Recommendation, BrowsingHistory
import numpy as np

# 数据充足性的最低要求
MIN_INTERACTIONS = 50  # 至少需要50条交互记录
MIN_UNIQUE_MOVIES = 20  # 至少需要20部被交互过的电影


class Command(BaseCommand):
    help = 'Trains the core recommendation model and caches the assets for real-time use.'

    def handle(self, *args, **options):
        self.stdout.write("开始训练核心推荐模型...")
        self.stdout.write("[1/3] 从数据库获取交互数据...")

        reviews_qs = UserReview.objects.select_related('user', 'movie')
        reviews_list = list(reviews_qs.values('user__username', 'movie__imdb_id', 'rating'))
        reviews_df = pd.DataFrame(reviews_list).rename(
            columns={'user__username': 'user_id', 'movie__imdb_id': 'item_id'})
        if not reviews_df.empty: reviews_df['weight'] = reviews_df['rating']

        fav_movies_qs = Recommendation.favorite_movies.through.objects.select_related('recommendation__user', 'movie')
        fav_movies_list = list(fav_movies_qs.values('recommendation__user__username', 'movie__imdb_id'))
        fav_movies_df = pd.DataFrame(fav_movies_list).rename(
            columns={'recommendation__user__username': 'user_id', 'movie__imdb_id': 'item_id'})
        if not fav_movies_df.empty: fav_movies_df['weight'] = 5.0

        history_qs = BrowsingHistory.objects.select_related('user', 'movie')
        history_list = list(history_qs.values('user__username', 'movie__imdb_id'))
        history_df = pd.DataFrame(history_list).rename(
            columns={'user__username': 'user_id', 'movie__imdb_id': 'item_id'})
        if not history_df.empty: history_df['weight'] = 0.5

        interactions_df = pd.concat([reviews_df, fav_movies_df, history_df], ignore_index=True)

        # --- 新增：数据充足性检查 ---
        if interactions_df.shape[0] < MIN_INTERACTIONS:
            self.stderr.write(self.style.ERROR(
                f"交互数据过少 ({interactions_df.shape[0]}条)，少于最低要求的 {MIN_INTERACTIONS} 条。任务中止，不会生成模型缓存。"))
            # 同时确保旧的坏缓存被删除
            cache.delete('recommendation_model_assets')
            self.stdout.write(self.style.WARNING("已清除可能存在的旧模型缓存。"))
            return

        interactions_df.sort_values('weight', ascending=False, inplace=True)
        interactions_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first', inplace=True)

        movies_qs = Movie.objects.filter(imdb_id__isnull=False)
        valid_movie_ids = set(movies_qs.values_list('imdb_id', flat=True))
        interactions_df = interactions_df[interactions_df['item_id'].isin(valid_movie_ids)].copy()

        if interactions_df['item_id'].nunique() < MIN_UNIQUE_MOVIES:
            self.stderr.write(self.style.ERROR(
                f"被交互过的电影数量过少 ({interactions_df['item_id'].nunique()}部)，少于最低要求的 {MIN_UNIQUE_MOVIES} 部。任务中止。"))
            cache.delete('recommendation_model_assets')
            self.stdout.write(self.style.WARNING("已清除可能存在的旧模型缓存。"))
            return
        # --- 检查结束 ---

        self.stdout.write("[2/3] 正在训练 ALS 模型...")
        interactions_df['user_id'] = interactions_df['user_id'].astype("category")
        interactions_df['item_id'] = interactions_df['item_id'].astype("category")
        item_map = {name: i for i, name in enumerate(interactions_df['item_id'].cat.categories)}
        reverse_item_map = {i: name for name, i in item_map.items()}
        item_user_matrix = coo_matrix(
            (interactions_df['weight'].astype(np.float32),
             (interactions_df['item_id'].cat.codes, interactions_df['user_id'].cat.codes))
        ).tocsr()
        als_model = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.01, iterations=20)
        als_model.fit(item_user_matrix)

        self.stdout.write("[3/3] 正在缓存模型资产...")
        model_assets = {
            'item_vectors': als_model.item_factors,
            'item_map': item_map,
            'reverse_item_map': reverse_item_map,
        }
        cache.set('recommendation_model_assets', model_assets, timeout=60 * 60 * 24 * 7)
        self.stdout.write(self.style.SUCCESS("核心模型训练完成并已成功缓存！"))