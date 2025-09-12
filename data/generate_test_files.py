import csv
import random
import os

# --- 1. 定义原始数据池 ---

# 电影基础数据
MOVIES_DATA = [
    {
        "imdb_id": "tt1375666",
        "original_title": "Inception",
        "cn_titles": "盗梦空间",
        "release_year": 2010,
        "summary": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "directors": ["Christopher Nolan"],
        "scriptwriters": ["测试1"],
        "actors": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "language": "en",
        "length": 111
    },
    {
        "imdb_id": "tt0111161",
        "original_title": "The Shawshank Redemption",
        "cn_titles": "肖申克的救赎",
        "release_year": 1994,
        "summary": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "genres": ["Drama"],
        "directors": ["Frank Darabont"],
        "scriptwriters": ["测试2"],
        "actors": ["Tim Robbins", "Morgan Freeman", "Bob Gunton"],
        "language": "en",
        "length": 222
    },
    {
        "imdb_id": "tt0468569",
        "original_title": "The Dark Knight",
        "cn_titles": "蝙蝠侠：黑暗骑士",
        "release_year": 2008,
        "summary": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        "genres": ["Action", "Crime", "Drama"],
        "directors": ["Christopher Nolan"],
        "scriptwriters": ["测试3"],
        "actors": ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
        "language": "en",
        "length": 333
    },
    {
        "imdb_id": None,  # 特意留空一个，测试备用关联键
        "original_title": "Forrest Gump",
        "cn_titles": "阿甘正传",
        "release_year": 1994,
        "summary": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.",
        "genres": ["Drama", "Romance"],
        "directors": ["Robert Zemeckis"],
        "scriptwriters": ["测试4"],
        "actors": ["Tom Hanks", "Robin Wright", "Gary Sinise"],
        "language": "en",
        "length": 444
    },
]

# 评论数据池
REVIEW_AUTHORS = ["Cinephile_101", "MovieMaven", "ReelTalker", "FilmFanatic", "ScreenScribe"]
REVIEW_SNIPPETS = [
    "A true masterpiece of modern cinema. The direction is flawless.",
    "A bit overrated in my opinion, but still a solid film.",
    "The lead performance was absolutely phenomenal and deserves all the awards.",
    "An unforgettable story that will stay with you for days.",
    "Visually stunning, but the plot felt a little weak.",
    "A must-see for any fan of the genre.",
]


# --- 2. 定义文件生成函数 ---

def create_movies_csv(filename="movies.csv"):
    """生成电影元数据CSV文件"""
    headers = [
        "imdb_id", "cn_titles", "original_title", "release_year",
        "summary", "genres", "directors", "scriptwriters", "actors","language", "length"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for movie in MOVIES_DATA:
            row = {
                "imdb_id": movie["imdb_id"],
                "cn_titles": movie["cn_titles"],
                "original_title": movie["original_title"],
                "release_year": movie["release_year"],
                "summary": movie["summary"],
                "genres": "|".join(movie["genres"]),  # 使用'|'拼接多值字段
                "directors": "|".join(movie["directors"]),
                "scriptwriters": "|".join(movie["scriptwriters"]),
                "actors": "|".join(movie["actors"]),
                "language": movie["language"],
                "length": movie["length"]
            }
            writer.writerow(row)
    print(f"Successfully created '{filename}'")


def create_reviews_csv(source_name, score_max, score_range, filename):
    """
    为指定的数据源生成评论CSV文件
    :param source_name: 来源名称, e.g., "Rotten Tomatoes"
    :param score_max: 该来源的满分值
    :param score_range: 评分的随机范围 (min, max)
    :param filename: 输出的文件名
    """
    headers = [
        "imdb_id", "original_title", "release_year",
        "author", "content", "score", "score_max"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for movie in MOVIES_DATA:
            # 为每部电影生成2到5条评论
            for _ in range(random.randint(2, 5)):
                # 根据评分范围生成随机分数
                if isinstance(score_range[0], float):
                    score = round(random.uniform(score_range[0], score_range[1]), 1)
                else:
                    score = random.randint(score_range[0], score_range[1])

                row = {
                    "imdb_id": movie["imdb_id"],
                    "original_title": movie["original_title"],
                    "release_year": movie["release_year"],
                    "author": random.choice(REVIEW_AUTHORS),
                    "content": random.choice(REVIEW_SNIPPETS),
                    "score": score,
                    "score_max": score_max,
                }
                writer.writerow(row)
    print(f"Successfully created '{filename}'")


# --- 3. 主执行函数 ---

def main():
    """主函数，调用所有生成器"""
    print("Starting to generate test data CSV files...")

    # 创建一个 'data' 文件夹来存放CSV文件 (如果不存在)
    if not os.path.exists('data'):
        os.makedirs('data')

    # 生成电影主文件
    create_movies_csv("movies_test.csv")

    # 生成烂番茄的评论文件 (百分制)
    create_reviews_csv(
        source_name="Rotten Tomatoes",
        score_max=100,
        score_range=(40, 100),
        filename="reviews_rottentomatoes_test.csv"
    )

    # 生成豆瓣的评论文件 (10分制)
    create_reviews_csv(
        source_name="Douban",
        score_max=10.0,
        score_range=(5.5, 9.8),
        filename="reviews_douban_test.csv"
    )

    print("\nAll test data files have been generated in the 'data' folder.")
    print("You can now use your Django management commands to import them.")


if __name__ == "__main__":
    main()