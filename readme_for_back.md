# 推荐系统后端交接文档

本文档说明如何使用和集成已经开发完成的实时推荐服务。

---

## 1. API 接口说明

项目现已包含一个即开即用的实时推荐API，你可以直接调用它来为任何用户获取个性化的电影推荐列表。

- **接口地址**: `GET /api/recommendations/realtime/<user_id>/`
- **请求方式**: `GET`
- **参数说明**: 
  - URL路径中的 `<user_id>` 部分，请传入用户的 `username` 字符串。
- **调用示例**:
  ```http
  GET http://127.0.0.1:8000/api/recommendations/realtime/some_username/
  ```

---

## 2. 核心工作机制 (重要)

每24h后第一次调用api会执行一次完整的模型训练

---

## 3. 返回数据格式

API会返回一个标准的JSON对象，格式如下：

#### 成功情况

当用户存在且有推荐结果时，返回一个包含推荐电影IMDb ID列表的JSON。

```json
{
  "user_id": "some_username",
  "recommendations": ["tt0111161", "tt0068646", "tt0110912", "..."] 
}
```

#### 用户不存在或无推荐的情况

当用户不存在于模型中，或该用户没有任何评分历史导致无法生成推荐时，返回一个空的推荐列表。

```json
{
  "user_id": "new_user_without_reviews",
  "recommendations": [],
  "message": "User not found or no recommendations available."
}
```

---

## 4. 后端集成要求

正常启动Django项目，然后在需要推荐功能的地方（例如用户个人主页），调用上述API即可。

```shell
# 正常启动服务
python manage.py runserver
```

---

## 5. 相关代码文件说明

为了方便您理解和后续维护，我们已在本次集成所涉及的核心代码文件中添加了详细的中文注释。

主要涉及以下4个文件：

1.  `main/django_project/config/settings.py`
    - **修改内容**: 添加了 `CACHES` 配置，启用文件缓存以持久化训练好的推荐模型。

2.  `main/django_project/films_recommender_system/urls.py`
    - **修改内容**: 添加了实时推荐API (`/api/recommendations/realtime/<user_id>/`) 的路由。

3.  `main/django_project/films_recommender_system/views.py`
    - **修改内容**: 实现了API的核心视图 `RealtimeRecommendationView`，以及模型缓存和加载逻辑 `get_or_train_model`，并重构了数据管道，使其从数据库读取实时数据。

4.  `main/django_project/films_recommender_system/management/commands/generate_recommendations.py`
    - **修改内容**: 重构了 `HybridRecModel` 类，使其能够支持从数据库数据（DataFrame）进行训练，以适应在线服务需求。
