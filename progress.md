四、 实验过程

本次实验旨在构建一个从算法研究到工程落地、完整且高效的电影推荐服务。为达成此目标，实验在开始前便确立了清晰的系统架构设计：采用“离线+在线”相结合的模式，将耗时的模型训练与快速的API响应分离。整个系统被划分为数据层、算法层、服务层和缓存层四个低耦合模块，数据从数据库出发，经由算法层训练、缓存层存储，最终通过服务层对外提供推荐能力。这一顶层设计指导了后续全部的开发与优化工作。

整个实验过程严格遵循了从理论研究、算法选型到工程实践、系统优化的标准流程，具体可分为三个核心阶段。

第一阶段：算法探索、评估与选型

本阶段的核心目标是，通过对多种主流推荐算法进行严谨的实现、对比和深度调试，为项目找到一个性能最优、效果最好的模型。本阶段不仅仅是实现算法，更对算法的每个环节进行了细致的考量。

1.1 基准模型：Item-based协同过滤

实验首先实现了经典的Item-based协同过滤算法作为性能基准。其核心思想是“喜欢某物品的用户也可能喜欢与该物品相似的其他物品”。

特征工程：实验将用户的评分（rating）作为其偏好强度的weight（权重），这比简单的“0/1”交互更能体现用户的喜好程度。

关键代码：计算物品相似度矩阵
```python
 说明：首先，构建一个用户-物品加权交互矩阵 (weighted_user_item_matrix)，
 然后将其转置为 (物品, 用户) 矩阵，最后使用余弦相似度计算出物品与物品之间的相似度。
from sklearn.metrics.pairwise import cosine_similarity

item_item_sim_matrix = cosine_similarity(
    weighted_user_item_matrix.T, 
    dense_output=False
).tocsr()
```

1.2 核心模型：ALS矩阵分解

接着，实验实现了更先进的ALS（交替最小二乘法）矩阵分解算法。该算法能学习用户和物品的隐向量表示，捕捉更深层次的关联信息。

参数调优：为确保模型性能，实验对ALS模型的关键超参数进行了初步调优，最终选定了如factors=64（隐向量维度）、regularization=0.01（正则化系数）等一组表现较优的参数组合。

关键代码：训练ALS模型
```python
 说明：本实验使用了业界广泛应用的`implicit`库来高效地实现ALS算法。
import implicit

als_model = implicit.als.AlternatingLeastSquares(
    factors=64, regularization=0.01, iterations=20
)
als_model.fit(weighted_item_user_matrix)
```
遇到的主要挑战及解决方案：
1.  环境问题：初次安装`implicit`库时，因缺少C++编译器而失败。解决方案：改为使用`conda`从`conda-forge`渠道安装预编译好的二进制包，成功解决了环境依赖问题。
2.  数据类型问题：在构建用户ID与索引的映射时，因`user_id`列存在数字和字符串混合类型，导致程序在排序时抛出`TypeError`。解决方案：在数据加载后，立即使用`.astype(str)`将所有ID列强制转换为字符串类型，保证了数据一致性。
3.  库的特性问题：在预测阶段，频繁出现`IndexError`。经深度调试后发现，`implicit`库的`model.user_factors`和`model.item_factors`两个属性的命名与其内容直观上是相反的。解决方案：在代码中交换使用这两个属性，从而获取了正确的用户和物品隐向量，解决了索引越界问题。

1.3 深度学习方法的探索与反思

为了探索更前沿的方案，实验同样尝试了NCF（神经协同过滤）等深度学习模型。这个过程虽未带来性能提升，但其揭示的“模型并非越复杂越好”的原则，是本次实验最有价值的结论之一。

探索过程：
1.  初次尝试NCF，性能极差（HR@10仅为0.24），远低于传统模型。
2.  初步怀疑是训练不充分，增加训练轮数和负采样比例后，性能反而进一步下降，证明模型已陷入严重过拟合。
3.  为对抗过拟合，我们简化模型，仅使用其GMF部分，但效果依然不佳。
4.  最后，转变思路，将学习任务从“分类”调整为“排序”，并实现了BPR损失函数，但最终性能仍无法与传统模型媲美。

结论与反思：这一系列的尝试有力地证明，对于本项目当前的数据规模和稀疏度，复杂的深度学习模型反而难以学习到有效的特征表示。相比之下，ALS等传统模型因其内置的归纳偏置，表现得更为稳定和出色。

1.4 最终方案：加权混合推荐模型

基于上述探索，实验最终决定采用一个加权混合模型，融合表现最佳的ALS协同信号和基于电影类型的内容信号。

特征工程：对于内容信号，实验提取了电影的genres（类型）字段，并使用TF-IDF（词频-逆文档频率） 技术将其文本信息转换为数值型特征向量，这是处理文本类特征的工业标准方法。

关键代码：分数归一化与加权融合
```python
 说明：为消除不同算法分数范围的差异，使用 MinMaxScaler 将它们各自归一化到 [0, 1] 区间，
 然后再进行加权求和，得到最终的混合推荐分。
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
norm_als_scores = scaler.fit_transform(als_scores_array)
norm_content_scores = scaler.fit_transform(content_scores_array)

final_scores = (0.7 * norm_als_scores) + (0.3 * norm_content_scores)
```
遇到的主要挑战及解决方案：在融合分数时，因依赖了Python字典不稳定的`.values()`顺序，导致数据错位并引发`IndexError`。解决方案：重构代码，改为通过显式遍历候选物品ID列表来构建分数数组，确保了数据顺序的严格一致，修复了此Bug。

第二阶段：系统集成与API服务化

本阶段的目标是将选出的最优混合模型，从离线脚本转变为一个健壮、高效、可用的在线服务。

2.1 实时推荐API开发

实验遵循RESTful设计原则，使用DRF创建了`RealtimeRecommendationView`，实现了资源的清晰定位和无状态通信。

边界情况处理：实验对API的输入进行了校验，能够优雅地处理各种边界情况。例如，当请求一个没有评分历史的新用户时，API会返回一个空的推荐列表及提示信息，而不是程序崩溃。

关键代码：API视图的GET方法
```python
 说明：此方法是API的入口，负责解析URL中的user_id，调用推荐逻辑，并返回JSON结果。
class RealtimeRecommendationView(APIView):
    def get(self, request, user_id, *args, **kwargs):
        model = get_or_train_model()
         ... (此处省略了对 model 是否为 None 的判断等鲁棒性代码)
        recommendations = model.get_recommendations_for_user(user_id, top_n=20)
        return Response({"user_id": user_id, "recommendations": recommendations})
```

2.2 性能优化：引入缓存机制

为解决模型训练耗时（长达数分钟）的性能瓶颈，实验设计并实现了“一次训练，周期复用”的缓存策略。

缓存策略权衡：实验最终选择了文件缓存（FileBasedCache） 而非内存缓存，其核心优势在于持久化——即使服务重启，训练好的模型也不会丢失。缓存有效期设置为24小时，这是在“模型时效性”与“服务器计算资源消耗”之间做出的合理权衡。

关键代码：模型缓存与加载逻辑
```python
 说明：此函数实现了缓存的核心逻辑。它首先检查缓存，若模型不存在，
 则触发一次训练，并将结果存入文件缓存，有效期设为24小时。
from django.core.cache import cache

def get_or_train_model():
    model = cache.get('recommendation_model')
    if model is None:
        model = HybridRecModel()
        model.train(...)
        cache.set('recommendation_model', model, timeout=86400)  86400秒 = 24小时
    return model
```

第三阶段：在线数据管道重构

在完成API开发后，一个关键问题是模型训练的数据源仍是静态CSV文件，无法服务于网站的真实用户。本阶段的目标就是打通算法与实时数据库的连接。

3.1 数据管道重构

实验重构了`get_or_train_model`函数，使其使用Django ORM直接从数据库中查询最新的用户评分和电影数据，并实时转换为算法所需的Pandas DataFrame格式。

关键代码：从数据库查询并转换数据
```python
 说明：使用 Django ORM 高效地查询所有评分记录，并通过.values()和rename()
 直接生成了符合算法输入格式的DataFrame，打通了Web框架与算法模块的数据链路。
import pandas as pd
from .models import UserReview

reviews_qs = UserReview.objects.values('user__username', 'movie__imdb_id', 'rating')
reviews_df = pd.DataFrame(list(reviews_qs)).rename(columns={
    'user__username': 'user_id', 'movie__imdb_id': 'item_id'
})
```

3.2 算法模块解耦

为配合数据管道的重构，`HybridRecModel`类也被重构，新增了`train_from_dataframes`方法。这使得算法核心类不再与数据加载方式绑定，既可以从文件训练（服务于离线命令），也可以从DataFrame训练（服务于在线API），大大提高了代码的灵活性和可维护性。至此，整个推荐系统形成了从实时数据获取、在线训练到API服务的完整闭环。

总结与展望

总结而言，本次实验成功地设计、实现并部署了一个完整且高效的电影推荐系统服务。从多种主流推荐算法的严格评估与选型出发，最终确定并实现了一个表现优异的加权混合推荐模型。更重要的是，通过API封装、缓存优化、数据管道重构等一系列工程实践，成功地将该算法从一个离线模型，转变为能够响应实时请求、服务于真实用户、并具备持久化学习能力的在线服务，完整地走完了从“算法研究”到“系统落地”的全过程。

展望未来，当前系统仍有广阔的优化空间。例如，可以通过引入更丰富的用户与物品特征（如导演、演员、文本简介等）来进一步提升模型精度；在积累更多数据后，可以重新探索如DeepFM、Wide & Deep等更先进的深度学习模型；架构上，可以构建独立的离线训练平台与在线评估体系（A/B测试），以实现更标准的工业级部署和迭代。这些都为后续的研究和开发工作留下了清晰的方向。