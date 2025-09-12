import argparse
import csv
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from films_recommender_system.models import Movie, MovieTitle, Genre, Person
from django.conf import settings
import os

class Command(BaseCommand):
    help = '从csv文件中加载电影相关数据'

    def add_arguments(self, parser):
        parser.add_argument('csv_filename', type=str, help='位于data/路径下的包含电影元数据的csv文件名')

    @transaction.atomic # 确保事务操作要么全部成功，要么全部失败，保证数据完整性
    def handle(self, *args, **options):
        csv_filename = options['csv_filename']
        # 获取项目根目录
        PROJECT_ROOT = settings.BASE_DIR.parent
        # 构建csv文件的绝对路径
        csv_file_path = os.path.join(PROJECT_ROOT, 'data', csv_filename)

        self.stdout.write(self.style.SUCCESS(f"开始从{csv_file_path} 导入数据"))

        try:
            with open(csv_file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    original_title = row.get('original_title')
                    release_year = row.get('release_year')

                    if not original_title or not release_year:
                        self.stdout.write(self.style.WARNING(f"跳过不完整的行: {row}"))
                        continue    # 没有原始名称或上映年份，直接跳过这一行，保证数据质量

                    # 创建或获取 Movie 对象
                    movie, created = Movie.objects.get_or_create(
                        original_title=original_title,
                        release_year=int(release_year),
                        defaults={
                            'language': row.get('language',''),
                            'length': int(row['length']),
                            'summary': row.get('summary', ''),
                            'imdb_id': row.get('imdb_id')if row.get('imdb_id') else None
                        }
                    )

                    # 如果是新电影，记录它的所有信息
                    if created:
                        self.stdout.write(f"创建新的电影信息：{original_title} ({release_year})")

                        # 1.创建并关联电影名称
                        # 创建原始标题
                        MovieTitle.objects.create(
                            movie=movie,
                            title_text=original_title,
                            language=movie.language,
                            is_primary=False
                        )
                        # 创建中文名称
                        if row.get('cn_titles'):
                            for cn_title in row.get('cn_titles').split('|'):
                                MovieTitle.objects.create(
                                    movie=movie,
                                    title_text=cn_title.strip(),
                                    language='zh-CN',
                                    is_primary=False    # 是否为主流中文名称由后续的真值算法确定
                                )

                        # 2.处理多对多关系：类型(Genres)、导演(directors)、编剧(scriptwriters)、演员(actors)
                        self._link_many_to_many(movie, row, 'genres', Genre)
                        self._link_many_to_many(movie, row, 'directors', Person)
                        self._link_many_to_many(movie, row, 'scriptwriters', Person)
                        self._link_many_to_many(movie, row, 'actors', Person)

        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"{csv_file_path}路径下未找到文件"))
            return

        self.stdout.write(self.style.SUCCESS('电影元数据导入完成'))


    def _link_many_to_many(self, movie, row, field_name, model):
        """
        辅助函数，用于处理多对多关系链接
        movie: Movie 对象
        row: 当前 CSV 行
        field_name: 字段名 (e.g., 'genres', 'actors')
        model: 关联的模型 (e.g., Genre, Person)
        """
        names_str = row.get(field_name, '')
        if not names_str:
            return

        # Movie 模型上的多对多管理器，(e.g. movie.genres, movie.actors)
        movie_field = getattr(movie, field_name)

        for name in names_str.split('|'):
            name = name.strip()
            if name:
                obj, created = model.objects.get_or_create(name=name)
                # 将该对象添加到电影的多对多关系中
                movie_field.add(obj)
                if created:
                    self.stdout.write(f"创建了新的 {model.__name__}: {name}")