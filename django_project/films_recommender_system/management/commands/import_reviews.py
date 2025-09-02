import csv
from django.core.management.base import BaseCommand
from django.db import transaction
from films_recommender_system.models import Movie, Source, Review
from django.conf import settings
import os


class Command(BaseCommand):
    help = '从csv文件中加载特定来源的电影评论数据'

    def add_arguments(self, parser):
        parser.add_argument('csv_filename', type=str,help='位于/data路径下包含评论数据的csv文件名')
        parser.add_argument(
            '--source',
            type=str,
            required=True,
            help='该评论数据的数据源名称(e.g. "Rotten Tomatoes", "Douban")'
        )

    @transaction.atomic
    def handle(self, *args, **options):
        csv_filename = options['csv_filename']
        # 获取项目根目录
        PROJECT_ROOT = settings.BASE_DIR.parent
        # 构建csv文件的绝对路径
        csv_file_path = os.path.join(PROJECT_ROOT, 'data', csv_filename)
        source_name = options['source']

        self.stdout.write(self.style.SUCCESS(f"开始从{csv_file_path}文件中 导入来自{source_name}的评论数据"))

        try:
            # 1.获取或创建数据源(Source)对象
            source, _ = Source.objects.get_or_create(
                name=source_name,
                defaults={'base_url': f"https://www.{source_name.lower().replace(" ","")}.com"}
            )

            with open(csv_file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                created_count = 0

                for row in reader:
                    # 2.查找评论关联的电影(Movie)对象
                    movie = self._get_movie(row)

                    if movie:
                        # 若找到了关联的电影，则创建评论(Review)对象
                        _, created = Review.objects.get_or_create(
                            movie=movie,
                            source=source,
                            score=float(row['score'])if row.get('score') else None,
                            score_max=float(row['score_max'])if row.get('score_max') else source.score_max,
                            defaults={
                                'author': row.get('author',''),
                                'content': row.get('content',''),
                            }
                        )
                        if created:
                            created_count += 1
                    else:
                        self.stdout.write(self.style.WARNING(
                            f"未找到评论所关联的电影{row.get('original_title')}({row.get('release_year')})，跳过该条评论信息的记录"
                        ))
            self.stdout.write(f"本次运行共创建了{created_count}条新的评论记录")

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"{csv_file_path}路径下未找到对应文件"))
            return

        self.stdout.write(self.style.SUCCESS(f"来源{source_name}的评论数据导入成功"))

    def _get_movie(self, row):
        """
        辅助函数，通过 imdb_id 或自然键 (title+year) 查找电影
        """

        imdb_id = row.get('imdb_id')
        original_title = row.get('original_title')
        release_year = int(row.get('release_year')) if row.get('release_year') else None

        # 优先通过imdb_id进行查询
        if imdb_id:
            try:
                return Movie.objects.get(imdb_id=imdb_id)
            except Movie.DoesNotExist:
                # 如果imdb_id找不到，可以继续尝试使用自然键进行匹配
                pass
        # 回退到使用自然键
        if original_title and release_year:
            try:
                return Movie.objects.get(original_title=original_title, release_year=release_year)
            except Movie.DoesNotExist:
                return None
            except Movie.MultipleObjectsReturned:
                self.stdout.write(self.style.WARNING(
                    f"数据库中存在重复的电影：{original_title}({release_year}),将使用第一条记录"
                ))
                return Movie.objects.filter(original_title=original_title, release_year=release_year).first()
        # 未查询到任何电影
        return None