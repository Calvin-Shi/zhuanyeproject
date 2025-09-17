# films_recommender_system/management/commands/build_search_index.py

import time
from django.core.management.base import BaseCommand
from django.core.cache import cache
from films_recommender_system.models import Movie
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Builds and caches a structured search index for fast, prioritized lookups.'

    def handle(self, *args, **options):
        self.stdout.write("开始构建结构化搜索索引...")
        start_time = time.time()

        movies_qs = Movie.objects.prefetch_related(
            'titles', 'genres', 'directors', 'actors'
        ).all()

        search_index = []

        self.stdout.write(f"正在为 {movies_qs.count()} 部电影创建索引文档...")
        for movie in tqdm(movies_qs):
            # --- 升级：创建结构化的搜索文档 ---
            # 将不同来源的文本分开存储，并全部转为小写
            title_text = " ".join(
                [movie.original_title.lower()] + [t.title_text.lower() for t in movie.titles.all()]
            )

            people_text = " ".join(
                [p.name.lower() for p in movie.directors.all()] + [p.name.lower() for p in movie.actors.all()]
            )

            genre_text = " ".join([g.name.lower() for g in movie.genres.all()])

            search_index.append({
                'id': movie.id,
                'title_text': title_text,
                'people_text': people_text,
                'genre_text': genre_text,
                'truth_score': movie.truth_score,
            })

        # 存入缓存，永不过期
        cache.set('global_search_index', search_index, timeout=None)

        end_time = time.time()
        duration = end_time - start_time
        self.stdout.write(
            self.style.SUCCESS(f"结构化搜索索引构建完成！共处理 {len(search_index)} 个文档，耗时 {duration:.2f} 秒。"))