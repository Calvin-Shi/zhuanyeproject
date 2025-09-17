# films_recommender_system/management/commands/debug_rec_cache.py

from django.core.management.base import BaseCommand
from django.core.cache import cache
import numpy as np


class Command(BaseCommand):
    help = 'Debugs the contents of the recommendation model cache.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.HTTP_INFO("--- 推荐模型缓存内容诊断 ---"))

        model_assets = cache.get('recommendation_model_assets')

        if not model_assets:
            self.stdout.write(self.style.ERROR("错误: 缓存键 'recommendation_model_assets' 未找到！"))
            self.stdout.write("这说明缓存是空的。请先运行 'generate_recommendations'。")
            return

        self.stdout.write("成功从缓存中加载 'recommendation_model_assets'。")
        self.stdout.write("\n正在检查资产内容...")

        try:
            item_vectors = model_assets['item_vectors']
            item_map = model_assets['item_map']
            reverse_item_map = model_assets['reverse_item_map']

            # 检查类型
            self.stdout.write(f"  - 'item_vectors' 类型: {type(item_vectors)}")
            self.stdout.write(f"  - 'item_map' 类型: {type(item_map)}")
            self.stdout.write(f"  - 'reverse_item_map' 类型: {type(reverse_item_map)}")

            # 检查维度/长度
            vec_len = item_vectors.shape[0] if isinstance(item_vectors, np.ndarray) else 'N/A'
            map_len = len(item_map) if isinstance(item_map, dict) else 'N/A'
            rev_map_len = len(reverse_item_map) if isinstance(reverse_item_map, dict) else 'N/A'

            self.stdout.write(f"  - 'item_vectors' 形状/长度: {vec_len}")
            self.stdout.write(f"  - 'item_map' 长度: {map_len}")
            self.stdout.write(f"  - 'reverse_item_map' 长度: {rev_map_len}")

            # 最终一致性检查
            self.stdout.write(self.style.HTTP_INFO("\n--- 一致性检查 ---"))
            if vec_len == map_len and map_len == rev_map_len:
                self.stdout.write(self.style.SUCCESS(f"PASS: 所有资产长度一致 ({vec_len})。缓存是健康的。"))
            else:
                self.stdout.write(self.style.ERROR("FAIL: 资产长度不一致！这是导致API错误的原因。"))
                self.stdout.write(f"  - 向量长度: {vec_len}")
                self.stdout.write(f"  - 映射长度: {map_len}")

        except (KeyError, AttributeError) as e:
            self.stdout.write(self.style.ERROR(f"错误: 缓存数据结构已损坏！缺少关键键或属性: {e}"))

        self.stdout.write("-" * 25)