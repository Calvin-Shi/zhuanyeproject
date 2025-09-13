# films_recommender_system/management/commands/import_movies.py

import os
import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Q
from django.conf import settings
from django.utils.text import slugify

from films_recommender_system.models import Movie, MovieTitle, Genre, Person

# 用于网络抓取的请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


class Command(BaseCommand):
    help = '从CSV文件导入电影数据，并可选择性地为缺少图片的电影抓取海报和背景图链接。'

    def add_arguments(self, parser):
        parser.add_argument('csv_filename', type=str, help='位于项目根目录下 data/ 文件夹内的CSV文件名')
        parser.add_argument(
            '--scrape-missing-images',
            action='store_true',
            help='导入数据后，为那些仍然缺少海报或背景图的电影从Letterboxd网站抓取图片链接。'
        )

    def handle(self, *args, **options):
        # --- 步骤 1: 从CSV文件导入核心数据 ---
        self._import_data_from_csv(options['csv_filename'])

        # --- 步骤 2: 如果用户指定，则执行图片抓取 ---
        if options['scrape_missing_images']:
            self._scrape_missing_images()

        self.stdout.write(self.style.SUCCESS('\n所有任务执行完毕。'))

    @transaction.atomic
    def _import_data_from_csv(self, csv_filename):
        """处理从CSV文件导入数据的逻辑 (优化版)"""
        csv_file_path = os.path.join(settings.BASE_DIR.parent, 'data', csv_filename)

        if not os.path.exists(csv_file_path):
            self.stderr.write(self.style.ERROR(f"错误：文件未找到于路径 {csv_file_path}"))
            return

        self.stdout.write(self.style.SUCCESS(f"===== 开始从 {csv_file_path} 导入电影数据... ====="))

        processed_identifiers = set()

        try:
            with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                for i, row in enumerate(reader, 1):
                    imdb_id = row.get('imdb_id') or None  # 确保空字符串转为None
                    original_title = row.get('original_title')
                    release_year_str = row.get('release_year')

                    if not (imdb_id or (original_title and release_year_str)):
                        self.stdout.write(self.style.WARNING(f"第 {i} 行：跳过不完整的行"))
                        continue

                    # --- 文件内去重逻辑 (保持不变) ---
                    identifier = imdb_id if imdb_id else (original_title, release_year_str)
                    if identifier in processed_identifiers:
                        self.stdout.write(self.style.WARNING(f"第 {i} 行：跳过文件内重复的电影条目 -> {identifier}"))
                        continue
                    processed_identifiers.add(identifier)

                    try:
                        release_year = int(release_year_str) if release_year_str else None
                    except (ValueError, TypeError):
                        self.stdout.write(self.style.WARNING(f"第 {i} 行：跳过无效的年份: {release_year_str}"))
                        continue

                    # --- 全新的、更稳健的查找与更新逻辑 ---
                    movie = None
                    created = False

                    # 1. 优先通过IMDb ID查找
                    if imdb_id:
                        movie = Movie.objects.filter(imdb_id=imdb_id).first()

                    # 2. 如果IMDb ID找不到，再通过标题和年份查找，以处理合并场景
                    if not movie and original_title and release_year:
                        movie = Movie.objects.filter(
                            original_title=original_title,
                            release_year=release_year
                        ).first()

                    # 准备要更新或创建的数据
                    data_from_csv = {
                        'summary': row.get('summary', ''),
                        'poster_url': row.get('poster_url') or None,
                        'backdrop_url': row.get('backdrop_url') or None,
                    }
                    if row.get('length'):
                        try:
                            data_from_csv['length'] = int(float(row.get('length')))
                        except (ValueError, TypeError):
                            pass

                    if movie:
                        # --- 更新现有电影 ---
                        self.stdout.write(f"更新已有电影: {movie.original_title} ({movie.release_year})")
                        # 使用循环更新字段，避免手动写一长串
                        for field, value in data_from_csv.items():
                            setattr(movie, field, value)

                        # 特别处理：如果找到的电影没有imdb_id，而CSV里有，就给它补上
                        if imdb_id and not movie.imdb_id:
                            movie.imdb_id = imdb_id
                            self.stdout.write(self.style.SUCCESS(f"  > 为其补充了 IMDb ID: {imdb_id}"))

                        movie.save()

                    else:
                        # --- 创建新电影 ---
                        self.stdout.write(f"创建新电影: {original_title} ({release_year})")

                        # 将所有数据合并用于创建
                        creation_data = data_from_csv.copy()
                        creation_data['original_title'] = original_title
                        creation_data['release_year'] = release_year
                        if imdb_id:
                            creation_data['imdb_id'] = imdb_id

                        movie = Movie.objects.create(**creation_data)
                        created = True

                    # 仅在首次创建电影时处理关联数据
                    if created:
                        MovieTitle.objects.create(movie=movie, title_text=original_title, is_primary=False)
                        if row.get('alternative_titles'):
                            for title in row.get('alternative_titles').split('/'):
                                if title.strip():
                                    MovieTitle.objects.create(
                                        movie=movie, title_text=title.strip(), language='zh-CN', is_primary=False
                                    )
                        self._link_many_to_many(movie, row, 'genres', Genre)
                        self._link_many_to_many(movie, row, 'directors', Person)
                        self._link_many_to_many(movie, row, 'actors', Person)

        except Exception as e:
            import traceback
            self.stderr.write(self.style.ERROR(f"处理CSV时发生未知错误: {e}"))
            self.stderr.write(traceback.format_exc())
            return

        self.stdout.write(self.style.SUCCESS("===== CSV数据导入完成 ====="))

    def _link_many_to_many(self, movie, row, field_name, model):
        """辅助函数，用于处理多对多关系链接"""
        names_str = row.get(field_name, '')
        if not names_str:
            return

        movie_field = getattr(movie, field_name)
        for name in names_str.split('/'):
            name = name.strip()
            if name:
                obj, created = model.objects.get_or_create(name=name)
                movie_field.add(obj)
                if created:
                    self.stdout.write(f"  > 添加了新的 {model.__name__}: {name}")

    def _scrape_missing_images(self):
        """为数据库中缺少图片链接的电影抓取数据"""
        self.stdout.write(self.style.SUCCESS("\n===== 开始为缺少图片的电影抓取链接... ====="))
        movies_to_update = Movie.objects.filter(
            Q(poster_url__isnull=True) | Q(poster_url='') |
            Q(backdrop_url__isnull=True) | Q(backdrop_url='')
        ).distinct()

        count = movies_to_update.count()
        if count == 0:
            self.stdout.write(self.style.SUCCESS('所有电影都已有图片链接，无需抓取。'))
            return

        self.stdout.write(f'发现 {count} 部电影需要更新图片链接。')
        success_count, failure_count = 0, 0

        for movie in movies_to_update:
            self.stdout.write(f"\n处理: {movie.original_title} ({movie.release_year})")
            film_detail_url = self._get_film_detail_url(movie)

            if not film_detail_url:
                self.stdout.write(self.style.ERROR("  > 无法找到详情页URL，跳过。"))
                failure_count += 1
                continue

            try:
                time.sleep(1.5)  # 遵守礼貌抓取原则
                response = requests.get(film_detail_url, headers=HEADERS, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                poster_url, backdrop_url = self._extract_image_urls(soup)

                updated = False
                if poster_url and not movie.poster_url:
                    movie.poster_url = poster_url
                    self.stdout.write(self.style.SUCCESS("  > 找到海报URL。"))
                    updated = True
                if backdrop_url and not movie.backdrop_url:
                    movie.backdrop_url = backdrop_url
                    self.stdout.write(self.style.SUCCESS("  > 找到背景URL。"))
                    updated = True

                if updated:
                    movie.save()
                    success_count += 1
                else:
                    self.stdout.write(self.style.WARNING("  > 未能找到所有缺失的图片链接。"))
                    failure_count += 1

            except requests.RequestException as e:
                self.stdout.write(self.style.ERROR(f"  > 请求详情页时失败: {e}"))
                failure_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  > 发生未知错误: {e}"))
                failure_count += 1

        self.stdout.write(self.style.SUCCESS(f'\n图片抓取完成！成功: {success_count}，失败: {failure_count}。'))

    def _get_film_detail_url(self, movie):
        """尝试通过直接访问和搜索两种方式获取电影在Letterboxd上的详情页URL"""
        film_slug = slugify(movie.original_title)
        direct_url = f"https://letterboxd.com/film/{film_slug}/"
        try:
            # 使用HEAD请求进行快速检查
            res = requests.head(direct_url, headers=HEADERS, timeout=5, allow_redirects=True)
            if res.status_code == 200:
                self.stdout.write(self.style.SUCCESS(f"  > 直接访问成功: {res.url}"))
                return res.url
        except requests.RequestException:
            pass  # 失败是正常的，将回退到搜索

        self.stdout.write(self.style.WARNING("  > 直接访问失败，回退到精确搜索..."))

        # 优先使用中文名搜索，其次是原始名
        primary_title = movie.titles.filter(language='zh-CN').first()
        search_term = primary_title.title_text if primary_title else movie.original_title

        try:
            search_url = f"https://letterboxd.com/search/films/{quote_plus(search_term)}/"
            self.stdout.write(f"  > 正在搜索: '{search_term}'")
            time.sleep(1.5)
            response = requests.get(search_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('ul.results li.film-result')  # 使用更通用的选择器

            for result in results:
                year_element = result.find('small', class_='release-year')
                year_in_result = year_element.text.strip() if year_element else None

                title_element = result.find('h2', class_='headline-2').find('a')
                title_in_result = title_element.text.strip()

                # 进行年份匹配，这是最重要的匹配条件
                if year_in_result == str(movie.release_year):
                    href = title_element['href']
                    found_url = f"https://letterboxd.com{href}"
                    self.stdout.write(self.style.SUCCESS(f"  > 通过年份精确匹配成功: {found_url}"))
                    return found_url

            self.stdout.write(self.style.WARNING("  > 在搜索结果中未找到精确年份匹配项。"))
            return None
        except requests.RequestException as e:
            self.stdout.write(self.style.ERROR(f"  > 搜索时网络错误: {e}"))
            return None

    def _extract_image_urls(self, soup):
        """从详情页soup对象中提取图片URL"""
        poster_url, backdrop_url = None, None

        # 提取海报图URL
        poster_div = soup.select_one('div.poster.film-poster')
        if poster_div:
            poster_img = poster_div.find('img', class_='image')
            if poster_img:
                # 获取srcset中最高清的链接
                poster_url = poster_img.get('srcset', '').split(',')[-1].strip().split(' ')[0] or poster_img.get('src')

        # 提取背景图URL
        backdrop_div = soup.select_one('div.backdrop-wrapper')
        if backdrop_div and 'data-backdrop' in backdrop_div.attrs:
            backdrop_url = backdrop_div['data-backdrop']

        return poster_url, backdrop_url