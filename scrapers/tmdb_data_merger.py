import tmdbsimple as tmdb
import pandas as pd
import time
import random

# 1. 设置 TMDb API Key
tmdb.API_KEY = '2082b05fc60e2b3228de0568499dfb65'

# 2. 定义输入和输出文件名
INPUT_CSV = 'extracted_movies.csv'
OUTPUT_CSV = 'combined_movie_data.csv'


def get_crew_info(crew_list, job_title):
    """从演职员列表中提取指定职位的人员姓名"""
    names = [person['name'] for person in crew_list if person['job'] == job_title]
    return ' / '.join(names)


def get_cast_info(cast_list, limit=3):
    """从演员列表中提取前几位主演的姓名"""
    names = [person['name'] for person in cast_list[:limit]]
    return ' / '.join(names)


def fetch_tmdb_data(imdb_id):
    """通过 IMDb ID 从 TMDb API 获取电影详细信息。"""
    try:
        # 使用 find() 方法通过外部 ID (imdb_id) 查找电影
        find_result = tmdb.Find(imdb_id).info(external_source='imdb_id')

        # 检查是否找到了结果
        if 'movie_results' in find_result and len(find_result['movie_results']) > 0:
            tmdb_movie = find_result['movie_results'][0]
            tmdb_id = tmdb_movie['id']

            # --- 第一次 API 调用：获取基本信息 ---
            details = tmdb.Movies(tmdb_id).info()

            # --- 第二次 API 调用：获取演职员表 ---
            credits = tmdb.Movies(tmdb_id).credits()

            # 提取导演、编剧、主演
            director = get_crew_info(credits.get('crew', []), 'Director')
            writers = get_crew_info(credits.get('crew', []), 'Writer')
            actors = get_cast_info(credits.get('cast', []), 5)  # 默认取前5位主演

            return {
                'id': details.get('id'),
                'imdb_id': details.get('imdb_id'),
                '海报': 'https://image.tmdb.org/t/p/w500' + details.get('poster_path', ''),
                'backdrop_path': details.get('backdrop_path', ''),
                '评分': details.get('vote_average'),
                '上映日期': details.get('release_date'),
                '片长': details.get('runtime'),
                '原片名': details.get('original_title'),
                '中文片名': details.get('title'),
                '剧情简介': details.get('overview'),
                '导演': director,
                '编剧': writers,
                '主演': actors,
                '类型': ', '.join([g['name'] for g in details.get('genres', [])])
            }
        else:
            print(f"未在 TMDb 上找到 IMDb ID 为 '{imdb_id}' 的电影。")
            return None

    except Exception as e:
        print(f"查找 IMDb ID '{imdb_id}' 时发生错误: {e}")
        return None


def main():
    """主函数：读取豆瓣数据，获取 TMDb 数据，然后合并保存。"""
    try:
        # 1. 读取您从豆瓣爬取的 CSV 文件
        douban_df = pd.read_csv(INPUT_CSV)

        # 2. 为每一部电影获取 TMDb 数据
        tmdb_data_list = []
        for index, row in douban_df.iterrows():
            imdb_id = row['imdb_id']

            # 修复: 检查 'original_title' 列是否存在
            movie_title_to_print = row.get('original_title', '未知电影')

            # 跳过没有 IMDb ID 的电影
            if not isinstance(imdb_id, str) or not imdb_id.startswith('tt'):
                print(f"跳过无有效 IMDb ID 的电影：{movie_title_to_print}")
                tmdb_data_list.append({})
                continue

            print(f"正在处理 IMDb ID: {imdb_id}...")
            tmdb_movie_info = fetch_tmdb_data(imdb_id)
            if tmdb_movie_info:
                tmdb_data_list.append(tmdb_movie_info)
            else:
                tmdb_data_list.append({})  # 如果找不到，添加一个空字典以保持行数一致

            # 随机延迟，防止请求过于频繁
            time.sleep(random.uniform(0.5, 1.5))

        # 3. 将 TMDb 数据转换成 DataFrame
        tmdb_df = pd.DataFrame(tmdb_data_list)

        # 4. 将两个 DataFrame 合并
        combined_df = pd.concat([douban_df, tmdb_df], axis=1)

        # 5. 将合并后的数据保存到新的 CSV 文件
        combined_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n成功将合并后的数据保存到 '{OUTPUT_CSV}' 文件中。")

    except FileNotFoundError:
        print(f"错误：未能找到文件 '{INPUT_CSV}'。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()