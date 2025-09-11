# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv
import re
from selenium.common.exceptions import NoSuchElementException
import random
from pathlib import Path

# 用于推荐（热门短评）配置
MAX_REVIEWS_PER_MOVIE = 40  # 每部电影最多抓多少条（20 的倍数）
MIN_USEFUL_VOTES = 2  # 过滤“有用”太少的短评，建议先设 0~5 观察
PAGE_SIZE = 20  # 豆瓣短评每页 20 条


def scrape_douban_data() :
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_experimental_option('excludeSwitches' , ['enable-logging'])

    driver_path = "D:\\chromedriver\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service , options=chrome_options)

    base_url = 'https://movie.douban.com/explore?support_type=movie&is_all=false&category=%E7%83%AD%E9%97%A8&type=%E5%85%A8%E9%83%A8'

    movies = []
    all_reviews = []
    imdb_counter = 1

    try :
        # 爬取电影列表页，获取基本信息和每个电影的链接
        driver.get(base_url)
        time.sleep(5)  # 初始加载等待

        # 动态加载更多，直到按钮消失或达到最大尝试次数
        max_attempts = 0  # 最大尝试点击次数，避免无限循环
        attempt = 0

        while attempt < max_attempts :
            try :
                # 滚动到页面底部，确保按钮可见
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # 等待滚动完成

                # 查找并点击"加载更多"按钮
                load_more_button = driver.find_element(By.XPATH , "//button[contains(text(), '加载更多')]")
                load_more_button.click()
                time.sleep(random.uniform(6 , 10))  # 随机延时等待新内容加载
                print("点击'加载更多'成功，加载下一批电影...")
                attempt += 1
            except NoSuchElementException :
                print("未找到'加载更多'按钮，已到达列表末尾。")
                break
            except Exception as e :
                print(f"点击'加载更多'时发生异常: {e}")
                time.sleep(5)  # 短暂休息后重试
                continue

        # 所有加载完成后，获取所有电影卡片
        movie_cards = driver.find_elements(By.CSS_SELECTOR , "ul.subject-list-list li")

        if not movie_cards :
            print("未找到电影卡片")
            return None , None

        # —— 新增：打印总数
        total_movies = len(movie_cards)
        print(f"总共找到 {total_movies} 部电影卡片。")

        for card in movie_cards :
            try :
                movie_link = card.find_element(By.TAG_NAME , "a").get_attribute("href")

                name_span = card.find_element(By.CLASS_NAME , "drc-subject-info-title-text")
                full_name = name_span.text.strip()
                match = re.search(r'([\u4e00-\u9fa5]+)\s+(.+)' , full_name)
                if match :
                    cn_title = match.group(1)
                    original_title = match.group(2)
                else :
                    cn_title = full_name
                    original_title = full_name

                subtitle_div = card.find_element(By.CLASS_NAME , "drc-subject-info-subtitle")
                subtitle_text = subtitle_div.text.strip()
                parts = [p.strip() for p in subtitle_text.split('/')]
                release_year = parts[0] if len(parts) > 0 else "未知"
                directors = parts[2] if len(parts) > 2 else "未知"  # 列表页的初步导演信息
                actors = parts[3] if len(parts) > 3 else "未知"

                movies.append({
                    "imdb_id" : f"tt{imdb_counter:06}" ,
                    "original_title" : original_title ,
                    "cn_titles" : cn_title ,
                    "release_year" : release_year ,
                    "directors" : directors ,
                    "actors" : actors ,
                    "link" : movie_link ,
                })
                imdb_counter += 1

            except Exception as e :
                print(f"提取列表页信息时发生异常: {e}")
                continue

        # 访问每个电影的详情页，提取额外信息和评论
        # —— 新增：带进度打印
        for idx , movie in enumerate(movies , start=1) :
            try :
                print(f"[{idx}/{total_movies}] 正在爬取电影详情页: {movie['original_title']}")
                driver.get(movie['link'])
                time.sleep(3)

                # 提取剧情简介
                try :
                    summary_element = driver.find_element(By.CSS_SELECTOR , 'span[property="v:summary"]')
                    movie['summary'] = summary_element.text.strip()
                except NoSuchElementException :
                    movie['summary'] = "无"

                # 提取类型
                try :
                    genres_elements = driver.find_elements(By.CSS_SELECTOR , 'span[property="v:genre"]')
                    genres = " / ".join([g.text.strip() for g in genres_elements])
                    movie['genres'] = genres if genres else "未知"
                except NoSuchElementException :
                    movie['genres'] = "未知"

                # 提取编剧
                try :
                    scriptwriters_section = driver.find_element(By.XPATH ,
                                                                "//div[@id='info']//span[text()='编剧']/following-sibling::span[1]")
                    scriptwriters_links = scriptwriters_section.find_elements(By.TAG_NAME , "a")
                    scriptwriters = " / ".join([s.text.strip() for s in scriptwriters_links])
                    movie['scriptwriters'] = scriptwriters if scriptwriters else "未知"
                except NoSuchElementException :
                    movie['scriptwriters'] = "未知"

                # 提取导演
                try :
                    directors_section = driver.find_element(By.XPATH ,
                                                            "//div[@id='info']//span[text()='导演']/following-sibling::span[1]")
                    directors_links = directors_section.find_elements(By.TAG_NAME , "a")
                    movie['directors'] = " / ".join(
                        [d.text.strip() for d in directors_links]) if directors_links else "未知"
                except NoSuchElementException :
                    movie['directors'] = "未知"

                # 提取片长
                try :
                    length_element = driver.find_element(By.CSS_SELECTOR , 'span[property="v:runtime"]')
                    movie['length'] = length_element.text.strip()
                except NoSuchElementException :
                    movie['length'] = "未知"

                # 提取豆瓣均分
                try :
                    score_element = driver.find_element(By.CSS_SELECTOR , 'strong[property="v:average"]')
                    movie['douban_average_score'] = score_element.text.strip()
                except NoSuchElementException :
                    movie['douban_average_score'] = "无评分"

                # 提取评分人数
                try :
                    votes_element = driver.find_element(By.CSS_SELECTOR , 'span[property="v:votes"]')
                    movie['number_of_ratings'] = votes_element.text.strip()
                except NoSuchElementException :
                    movie['number_of_ratings'] = "未知"

                # 根据均分判断电影星级
                try :
                    score = float(movie['douban_average_score'])
                    movie['douban_star_rating'] = score / 2.0
                except (ValueError , TypeError) :
                    movie['douban_star_rating'] = "未知星级"

                # 爬“全部 + 热门”短评列表页，翻页取前 X 条
                reviews = crawl_hot_comments_for_movie(driver , movie)
                all_reviews.extend(reviews)


            except Exception as e :
                print(f"提取电影详情信息时发生异常: {e}")
                continue

        return movies , all_reviews

    except Exception as e :
        print(f"发生异常: {e}")
        return None , None

    finally :
        driver.quit()


def crawl_hot_comments_for_movie(driver , movie) :
    """
    通过点击“全部短评”链接进入评论页，并翻页抓取热门短评。
    """
    collected = []

    print(f"尝试通过点击链接进入电影 {movie['original_title']} 的评论页面...")

    try :
        # 1. 找到“全部短评”链接并点击
        all_comments_link = driver.find_element(By.CSS_SELECTOR , "#comments-section h2 .pl a")
        all_comments_link.click()

        # 2. 等待页面跳转和加载新内容
        time.sleep(random.uniform(3 , 6))
        print(f"成功点击链接，正在抓取 {movie['original_title']} 的评论...")

    except NoSuchElementException :
        print("未找到“全部短评”链接。")
        return collected
    except Exception as e :
        print(f"点击“全部短评”链接时发生错误: {e}")
        return collected

    # 3. 循环翻页，直到达到最大评论数
    while len(collected) < MAX_REVIEWS_PER_MOVIE :
        try :
            # 获取当前页的评论
            items = driver.find_elements(By.CSS_SELECTOR , "div.comment-item")
            if not items :
                print("当前页面没有找到评论，可能已抓取完毕或页面结构有变。")
                break

            for it in items :
                # 抓取每条评论的详细信息
                try :
                    votes_text_element = it.find_element(By.CSS_SELECTOR , ".votes, .vote-count")
                    votes_text = votes_text_element.text.strip()
                    useful = int(re.findall(r"\d+" , votes_text)[0]) if re.search(r"\d+" , votes_text) else 0
                    if useful < MIN_USEFUL_VOTES :
                        continue
                except NoSuchElementException :
                    useful = 0

                try :
                    info = it.find_element(By.CSS_SELECTOR , ".comment-info")
                    author = info.find_element(By.CSS_SELECTOR , "a").text.strip()
                    user_status = "看过" if "看过" in info.text else "想看" if "想看" in info.text else "未知"
                except NoSuchElementException :
                    author , user_status = "未知作者" , "未知"

                try :
                    rating_el = it.find_element(By.CSS_SELECTOR , ".comment-info [class^='allstar']")
                    cls = rating_el.get_attribute("class")
                    comment_score = float(re.search(r"allstar(\d{2})" , cls).group(1)) / 10
                except (NoSuchElementException , AttributeError) :
                    comment_score = "无评分"

                try :
                    content = it.find_element(By.CSS_SELECTOR , "span.short").text.strip()
                except NoSuchElementException :
                    content = it.find_element(By.CSS_SELECTOR , "p.comment-content").text.strip()

                try :
                    comment_time = it.find_element(By.CSS_SELECTOR , ".comment-time").get_attribute("title")
                except NoSuchElementException :
                    comment_time = "无时间"

                collected.append({
                    "imdb_id" : movie["imdb_id"] ,
                    "original_title" : movie["original_title"] ,
                    "release_year" : movie["release_year"] ,
                    "author" : author ,
                    "user_status" : user_status ,
                    "content" : content ,
                    "score" : comment_score ,
                    "score_max" : 5.0 ,
                    "comment_time" : comment_time
                })
                print(f"已成功抓取第 {len(collected)} 条评论。")

                if len(collected) >= MAX_REVIEWS_PER_MOVIE :
                    print("已达到最大抓取数量，停止抓取。")
                    break

            # 4. 尝试点击“后页”链接进行翻页
            try :
                next_page_link = driver.find_element(By.CSS_SELECTOR , "a.next[data-page='next']")
                print("找到“后页 >”按钮，正在点击...")
                next_page_link.click()
                time.sleep(random.uniform(3 , 6))  # 等待新页面加载
            except NoSuchElementException :
                print("未找到“后页 >”按钮，已到达最后一页或抓取完毕。")
                break
            except Exception as e :
                print(f"点击“后页 >”时发生错误: {e}")
                break

        except Exception as e :
            print(f"在评论页面抓取时发生错误: {e}")
            break

    return collected


if __name__ == '__main__' :
    movie_list , review_list = scrape_douban_data()

    if movie_list and review_list :
        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(exist_ok=True , parents=True)

        # 修改文件名，加上 "test_"
        movies_csv_filename = data_dir / "test_movies.csv"
        movies_fieldnames = ["imdb_id" , "original_title" , "cn_titles" , "release_year" , "summary" , "genres" ,
                             "directors" ,
                             "scriptwriters" , "actors" , "length" , "douban_average_score" , "douban_star_rating" ,
                             "number_of_ratings" , "link"]
        with open(movies_csv_filename , mode='w' , newline='' , encoding='utf-8') as file :
            writer = csv.DictWriter(file , fieldnames=movies_fieldnames)
            writer.writeheader()
            writer.writerows(movie_list)

        print(f"已成功爬取电影元数据并保存到 {movies_csv_filename} 文件中。")

        # 修改文件名，加上 "test_"
        reviews_csv_filename = data_dir / "test_reviews_douban.csv"
        reviews_fieldnames = ["imdb_id" , "original_title" , "release_year" , "author" , "user_status" , "content" ,
                              "score" ,
                              "score_max" , "comment_time"]
        with open(reviews_csv_filename , mode='w' , newline='' , encoding='utf-8') as file :
            writer = csv.DictWriter(file , fieldnames=reviews_fieldnames)
            writer.writeheader()
            writer.writerows(review_list)

        print(f"已成功爬取电影评论并保存到 {reviews_csv_filename} 文件中。")
    else :
        print("爬取失败。请检查网络连接或网站结构是否已更改。")