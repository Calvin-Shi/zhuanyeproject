# films_recommender_system/management/commands/import_reviews.py

import csv
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from django.db import transaction
from django.conf import settings
from films_recommender_system.models import Movie, Source, Review


class Command(BaseCommand):
    help = '从CSV文件加载特定来源的电影评论数据。'

    def add_arguments(self, parser):
        parser.add_argument('csv_filename', type=str, help='位于/data路径下包含评论数据的CSV文件名')
        parser.add_argument(
            '--source',
            type=str,
            required=True,
            help='该评论数据的数据源名称 (e.g., "Letterboxd", "Douban")'
        )

    @transaction.atomic
    def handle(self, *args, **options):
        csv_filename = options['csv_filename']
        source_name = options['source']
        # 构建csv文件的绝对路径 (假设'data'目录在项目根目录)
        csv_file_path = os.path.join(settings.BASE_DIR.parent, 'data', csv_filename)

        self.stdout.write(self.style.SUCCESS(f"===== 开始从 {csv_file_path} 导入来自 {source_name} 的评论... ====="))

        try:
            # 1. 获取或创建数据源(Source)对象
            source, _ = Source.objects.get_or_create(
                name=source_name,
                defaults={'base_url': f"https://www.{source_name.lower().replace(' ', '')}.com"}
            )

            # 使用 'utf-8-sig' 来兼容被Excel编辑过的UTF-8文件
            with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                created_count = 0
                skipped_count = 0

                for i, row in enumerate(reader, 1):
                    # 2. 查找评论关联的电影(Movie)对象
                    movie = self._get_movie(row, i)

                    if not movie:
                        skipped_count += 1
                        continue

                    # 3. 准备评论数据，并进行类型转换和错误处理
                    author = row.get('nickname', '')
                    content = row.get('content', '')

                    try:
                        score = float(row['score']) if row.get('score') else None
                    except (ValueError, TypeError):
                        self.stdout.write(self.style.WARNING(f"行 {i}: 无效的评分值 '{row.get('score')}'，已设为None。"))
                        score = None

                    try:
                        score_max = float(row['score_max']) if row.get('score_max') else source.score_max
                    except (ValueError, TypeError):
                        score_max = source.score_max

                    try:
                        approvals_num = int(row['approvals_num']) if row.get('approvals_num') else None
                    except (ValueError, TypeError):
                        self.stdout.write(
                            self.style.WARNING(f"行 {i}: 无效的赞同数 '{row.get('approvals_num')}'，已设为None。"))
                        approvals_num = None

                    content_date = self._parse_datetime(row.get('content_date'), i)

                    # 4. 创建或获取评论对象
                    # 使用 movie, source, author, 和 content 作为联合唯一标识来避免重复导入
                    _, created = Review.objects.get_or_create(
                        movie=movie,
                        source=source,
                        author=author,
                        content=content,  # 假设相同用户对同一电影的相同内容的评论是唯一的
                        defaults={
                            'score': score,
                            'score_max': score_max,
                            'content_date': content_date,
                            'approvals_num': approvals_num
                        }
                    )

                    if created:
                        created_count += 1

                self.stdout.write(self.style.SUCCESS(f"\n处理完成！"))
                self.stdout.write(f"  > 本次运行共创建了 {created_count} 条新的评论记录。")
                self.stdout.write(f"  > 因未找到关联电影，跳过了 {skipped_count} 条评论。")

        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"错误: 在 {csv_file_path} 路径下未找到文件。"))
            return
        except Exception as e:
            import traceback
            self.stderr.write(self.style.ERROR(f"处理CSV时发生未知错误: {e}"))
            self.stderr.write(traceback.format_exc())
            return

        self.stdout.write(self.style.SUCCESS(f"===== 来源 {source_name} 的评论数据导入成功 ====="))

    def _get_movie(self, row, row_num):
        """辅助函数，通过 imdb_id 或 (title+year) 查找电影"""
        imdb_id = row.get('imdb_id','')
        original_title = row.get('original_title')

        try:
            release_year = int(row.get('release_year')) if row.get('release_year') else None
        except (ValueError, TypeError):
            self.stdout.write(self.style.WARNING(f"行 {row_num}: 无效的年份 '{row.get('release_year')}'，无法查找电影。"))
            return None

        # 优先通过 imdb_id 查询
        if imdb_id:
            movie = Movie.objects.filter(imdb_id=imdb_id).first()
            if movie:
                return movie

        # 回退到使用 original_title 和 release_year
        if original_title and release_year:
            try:
                return Movie.objects.get(original_title=original_title, release_year=release_year)
            except Movie.DoesNotExist:
                self.stdout.write(self.style.WARNING(
                    f"行 {row_num}: 未在数据库中找到电影 '{original_title} ({release_year})'，跳过此评论。"
                ))
                return None
            except Movie.MultipleObjectsReturned:
                self.stdout.write(self.style.WARNING(
                    f"行 {row_num}: 数据库中存在重复的电影 '{original_title} ({release_year})'，将使用第一条记录。"
                ))
                return Movie.objects.filter(original_title=original_title, release_year=release_year).first()

        self.stdout.write(self.style.WARNING(
            f"行 {row_num}: CSV行中缺少足够的电影标识信息 (imdb_id或title+year)，跳过此评论。"
        ))
        return None

    def _parse_datetime(self, date_string, row_num):
        """辅助函数，用于解析多种常见格式的日期时间字符串"""
        if not date_string:
            return None

        # 常见日期时间格式列表
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S',  # e.g., '2023-10-27 15:45:00'
            '%Y-%m-%d',  # e.g., '2023-10-27'
            '%Y/%m/%d',  # e.g., '2023/10/27'
        ]

        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        self.stdout.write(self.style.WARNING(
            f"行 {row_num}: 无法解析日期格式 '{date_string}'，该评论的日期将设为None。"
        ))
        return None