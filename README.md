# Recommender_System (电影推荐系统)

## 项目目录结构
```
Recommender_System/
├── .git/ # Git 版本控制
├── .idea/ # IDE 配置 (已忽略)
├── .venv/ # 虚拟环境 (已忽略)
├── data/ # 存放爬虫生成的CSV文件
│ ├── movies.csv
│ ├── reviews_douban.csv
│ └── ...
├── scrapers/ # 爬虫脚本
│ ├── new_crawler.py # 豆瓣热门电影 & 评论爬虫
│ ├── test_new_crawler.py # 爬虫测试脚本
│ └── ...
├── algorithm/ # 算法
│ └── Truth_value.py # 数据清洗 & 真值生成
├── django_project/ # Django Web 项目
│ ├── config/ # Django 配置文件
│ │ ├── init.py
│ │ ├── asgi.py
│ │ ├── settings.py
│ │ ├── urls.py
│ │ └── wsgi.py
│ ├── manage.py # Django 管理脚本
│ └── films_recommender_system # 推荐系统 app
├── truth_value_out/ # 真值算法输出 
│ ├── item_quality.csv
│ ├── interactions_gt.csv
│ ├── splits.csv
│ └── eval_samples.csv
├── requirements.txt # 项目依赖
├── .gitignore
└── README.md
```



---

## 主要修改说明（基于 `main` 分支）

### `requirements.txt`
已添加 `selenium` 库，作为项目的必要依赖。

### `scrapers` 目录
- **代码精简**: 移除了 `new_crawler.py` 中冗余的代码，使脚本更简洁高效。
- **配置调整**: `new_crawler.py` 中的 `max_attempts`（最大尝试点击次数）从 10 增加到了 15，旨在爬取更多电影。
- **数据存储**: 爬取到的所有数据现在都统一保存到项目根目录下的 **`data/`** 文件夹中。
- **数据量增加**: 为了提升真值算法的准确性，爬虫现在会抓取更多的电影短评。
- **使用须知**: 运行爬虫前，请务必确认 `new_crawler.py` 中 `driver_path` 的值与你的本地 `chromedriver` 路径一致。

### `algorithm` 目录
新增了 `Truth_value.py` 脚本，该脚本的主要功能是对爬取的数据进行**清洗**并生成用于评估的**真值**数据。
