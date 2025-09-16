# films_recommender_system/management/commands/clear_rec_cache.py

from django.core.management.base import BaseCommand
from django.core.cache import cache


class Command(BaseCommand):
    help = 'Clears the cached recommendation model assets.'

    def handle(self, *args, **options):
        self.stdout.write("正在清除推荐模型缓存...")

        # 删除存储模型资产的特定缓存键
        result = cache.delete('recommendation_model_assets')

        if result:
            self.stdout.write(self.style.SUCCESS("缓存 'recommendation_model_assets' 已成功清除！"))
        else:
            self.stdout.write(self.style.WARNING("缓存键 'recommendation_model_assets' 未找到，可能已经被清除了。"))