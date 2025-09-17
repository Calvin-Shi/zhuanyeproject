# films_recommender_system/management/commands/generate_recommendations.py

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
from django.core.management.base import BaseCommand
from django.core.cache import cache
from films_recommender_system.models import Movie
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Builds and caches content-based feature vectors for all movies.'

    def handle(self, *args, **options):
        self.stdout.write("开始为所有电影构建内容画像向量...")

        # 1. 获取所有电影及其相关内容
        movies = Movie.objects.prefetch_related('genres', 'directors', 'actors').all()

        if not movies:
            self.stderr.write(self.style.ERROR("数据库中没有电影，任务中止。"))
            return

        documents = []
        movie_ids_in_order = []

        for movie in tqdm(movies, desc="[1/2] 创建特征文档"):
            if not movie.imdb_id:
                continue

            # 为每部电影创建一个由其内容特征（类型、导演、演员）组成的“词袋”
            features = []

            # 为特征添加前缀以避免混淆（例如，类型'Action'和演员'Action Bronson'）
            for genre in movie.genres.all():
                features.append(f"genre_{genre.name.replace(' ', '')}")

            for director in movie.directors.all():
                features.append(f"director_{director.name.replace(' ', '')}")

            for actor in movie.actors.all():
                features.append(f"actor_{actor.name.replace(' ', '')}")

            if features:
                documents.append(" ".join(features))
                movie_ids_in_order.append(movie.imdb_id)

        if not documents:
            self.stderr.write(self.style.ERROR("没有任何电影有关联的内容特征，无法构建模型。"))
            return

        # 2. 使用TF-IDF将“词袋”转换为数学向量
        self.stdout.write("[2/2] 正在使用TF-IDF进行向量化...")
        vectorizer = TfidfVectorizer(max_features=1000)  # 限制特征维度，防止过大
        movie_vectors = vectorizer.fit_transform(documents).toarray()  # 转换为NumPy数组

        # 3. 构建并缓存资产
        # 现在的item_map是imdb_id到向量数组行索引的映射
        item_map = {imdb_id: i for i, imdb_id in enumerate(movie_ids_in_order)}

        model_assets = {
            'item_vectors': movie_vectors,
            'item_map': item_map,
            # 在这个模型中，我们不再需要 reverse_item_map，因为ID本身就是名称
        }

        cache.set('recommendation_model_assets', model_assets, timeout=None)

        self.stdout.write(self.style.SUCCESS("内容画像向量构建完成并已成功缓存！"))
        self.stdout.write(f"  - 向量维度: {movie_vectors.shape}")
        self.stdout.write(f"  - 映射长度: {len(item_map)}")