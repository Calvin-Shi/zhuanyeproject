from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv
import re
from selenium.common.exceptions import NoSuchElementException


def scrape_douban_data():
    """
    爬取豆瓣热门电影数据，包括电影元数据和用户评论，并返回两个列表。
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver_path = "D:\\chromedriver\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    base_url = 'https://movie.douban.com/explore?support_type=movie&is_all=false&category=%E7%83%AD%E9%97%A8&type=%E5%85%A8%E9%83%A8'

    movies = []
    all_reviews = []
    imdb_counter = 1

    try:
        # 第一步：爬取电影列表页，获取基本信息和每个电影的链接
        driver.get(base_url)
        time.sleep(5)
        movie_cards = driver.find_elements(By.CSS_SELECTOR, "ul.subject-list-list li")

        if not movie_cards:
            print("未找到电影卡片")
            return None, None

        for card in movie_cards:
            try:
                movie_link = card.find_element(By.TAG_NAME, "a").get_attribute("href")

                name_span = card.find_element(By.CLASS_NAME, "drc-subject-info-title-text")
                full_name = name_span.text.strip()
                match = re.search(r'([\u4e00-\u9fa5]+)\s+(.+)', full_name)
                if match:
                    cn_title = match.group(1)
                    original_title = match.group(2)
                else:
                    cn_title = full_name
                    original_title = full_name

                subtitle_div = card.find_element(By.CLASS_NAME, "drc-subject-info-subtitle")
                subtitle_text = subtitle_div.text.strip()
                parts = [p.strip() for p in subtitle_text.split('/')]
                release_year = parts[0] if len(parts) > 0 else "未知"
                directors = parts[2] if len(parts) > 2 else "未知"
                actors = parts[3] if len(parts) > 3 else "未知"

                movies.append({
                    "imdb_id": f"tt{imdb_counter:06}",
                    "original_title": original_title,
                    "cn_titles": cn_title,
                    "release_year": release_year,
                    "directors": directors,
                    "actors": actors,
                    "link": movie_link,
                })
                imdb_counter += 1

            except Exception as e:
                print(f"提取列表页信息时发生异常: {e}")
                continue

        # 第二步：访问每个电影的详情页，提取额外信息和评论
        for movie in movies:
            try:
                print(f"正在爬取电影详情页: {movie['original_title']}")
                driver.get(movie['link'])
                time.sleep(3)

                # 提取剧情简介 (summary)
                try:
                    summary_element = driver.find_element(By.CSS_SELECTOR, 'span[property="v:summary"]')
                    movie['summary'] = summary_element.text.strip()
                except NoSuchElementException:
                    movie['summary'] = "无"

                # 提取类型 (genres)
                try:
                    genres_elements = driver.find_elements(By.CSS_SELECTOR, 'span[property="v:genre"]')
                    genres = " / ".join([g.text.strip() for g in genres_elements])
                    movie['genres'] = genres if genres else "未知"
                except NoSuchElementException:
                    movie['genres'] = "未知"

                # 提取编剧 (scriptwriters)
                try:
                    scriptwriters_section = driver.find_element(By.XPATH,
                                                                "//div[@id='info']//span[text()='编剧']/following-sibling::span[1]")
                    scriptwriters_links = scriptwriters_section.find_elements(By.TAG_NAME, "a")
                    scriptwriters = " / ".join([s.text.strip() for s in scriptwriters_links])
                    movie['scriptwriters'] = scriptwriters if scriptwriters else "未知"
                except NoSuchElementException:
                    movie['scriptwriters'] = "未知"

                # 提取片长 (length)
                try:
                    length_element = driver.find_element(By.CSS_SELECTOR, 'span[property="v:runtime"]')
                    movie['length'] = length_element.text.strip()
                except NoSuchElementException:
                    movie['length'] = "未知"

                # 新增：提取豆瓣均分 (Douban Average Score)
                try:
                    score_element = driver.find_element(By.CSS_SELECTOR, 'strong[property="v:average"]')
                    movie['douban_average_score'] = score_element.text.strip()
                except NoSuchElementException:
                    movie['douban_average_score'] = "无评分"

                # 新增：提取评分人数 (Number of Ratings)
                try:
                    votes_element = driver.find_element(By.CSS_SELECTOR, 'span[property="v:votes"]')
                    movie['number_of_ratings'] = votes_element.text.strip()
                except NoSuchElementException:
                    movie['number_of_ratings'] = "未知"

                # 新增：根据均分判断电影星级
                try:
                    score = float(movie['douban_average_score'])
                    movie['douban_star_rating'] = score / 2.0
                except (ValueError, TypeError):
                    movie['douban_star_rating'] = "未知星级"

                # 爬取评论
                comment_items = driver.find_elements(By.CSS_SELECTOR, "#comments-section .comment-item")

                if not comment_items:
                    print(f"未找到评论: {movie['original_title']}")

                for item in comment_items:
                    try:
                        comment_info_text = item.find_element(By.CSS_SELECTOR, ".comment-info").text.strip()
                        user_status = "未知"
                        if "看过" in comment_info_text:
                            user_status = "看过"
                        elif "想看" in comment_info_text:
                            user_status = "想看"

                        comment_author = item.find_element(By.CSS_SELECTOR, ".comment-info a").text.strip()

                    except:
                        comment_author = "未知作者"
                        user_status = "未知"

                    try:
                        rating_element = item.find_element(By.CSS_SELECTOR, ".comment-info [class^='allstar']")
                        rating_class = rating_element.get_attribute("class")
                        score_match = re.search(r'allstar(\d{2})', rating_class)
                        comment_score = float(score_match.group(1)) / 10 if score_match else "无评分"

                    except:
                        comment_score = "无评分"

                    try:
                        comment_time = item.find_element(By.CSS_SELECTOR, ".comment-time").get_attribute("title")
                    except:
                        comment_time = "无时间"

                    try:
                        content_element = item.find_element(By.CSS_SELECTOR, "p.comment-content")
                        comment_content = content_element.text.strip()
                    except:
                        comment_content = "无评论内容"

                    all_reviews.append({
                        "imdb_id": movie["imdb_id"],
                        "original_title": movie["original_title"],
                        "release_year": movie["release_year"],
                        "author": comment_author,
                        "user_status": user_status,
                        "content": comment_content,
                        "score": comment_score,
                        "score_max": 5.0,
                        "comment_time": comment_time
                    })

            except Exception as e:
                print(f"提取电影详情信息时发生异常: {e}")
                continue

        return movies, all_reviews

    except Exception as e:
        print(f"发生异常: {e}")
        return None, None

    finally:
        driver.quit()


if __name__ == '__main__':
    movie_list, review_list = scrape_douban_data()

    if movie_list and review_list:
        movies_csv_filename = "movies.csv"
        movies_fieldnames = ["imdb_id", "original_title", "cn_titles", "release_year", "summary", "genres", "directors",
                             "scriptwriters", "actors", "length", "douban_average_score", "douban_star_rating",
                             "number_of_ratings", "link"]
        with open(movies_csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=movies_fieldnames)
            writer.writeheader()
            writer.writerows(movie_list)

        print(f"已成功爬取电影元数据并保存到 {movies_csv_filename} 文件中。")

        reviews_csv_filename = "reviews_douban.csv"
        reviews_fieldnames = ["imdb_id", "original_title", "release_year", "author", "user_status", "content", "score",
                              "score_max", "comment_time"]
        with open(reviews_csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=reviews_fieldnames)
            writer.writeheader()
            writer.writerows(review_list)

        print(f"已成功爬取电影评论并保存到 {reviews_csv_filename} 文件中。")
    else:
        print("爬取失败。请检查网络连接或网站结构是否已更改。")