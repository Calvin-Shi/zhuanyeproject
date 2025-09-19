import matplotlib.pyplot as plt
import numpy as np

# 1. 定义数据
labels = ['缓存未命中 (首次调用)', '缓存命中 (后续调用)']
times_ms = [142530, 412]

# 2. 创建图表
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

colors = ['#FF6347', '#4682B4'] # Tomato, SteelBlue

bars = ax.bar(labels, times_ms, color=colors)

# 3. 使用对数坐标
# 因为两个数值差距过大，使用对数坐标才能在同一张图上清晰地展示它们
ax.set_yscale('log')

# 4. 添加标签、标题
ax.set_ylabel('响应时间 (毫秒) - 对数坐标', fontsize=14)
ax.set_title('API响应时间对比 (缓存效果分析)', fontsize=18, fontweight='bold', pad=20)
ax.tick_params(axis='x', labelsize=12)

# 在柱子顶端标注数值
def autolabel(rects):
    """在每个柱子上方附加一个文本标签，显示其高度。"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)} ms', # 格式化为整数毫秒
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

autolabel(bars)

# 移除Y轴的次要刻度标签，使其更清晰
ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
# 设置Y轴主要刻度的标签格式
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))


fig.tight_layout()

# 5. 保存图片
plt.savefig('cache_performance_comparison.png', dpi=300)

print("图表 'cache_performance_comparison.png' 已成功生成并保存。")
