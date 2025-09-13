import csv
import re
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth

# --- 配置区域 ---
INPUT_CSV_FILE = 'movies_letterdoxd_details_v19.csv'  # 确保这是输入文件名
OUTPUT_CSV_FILE = 'reviews_letterdoxd_v14.csv'
MOVIES_TO_PROCESS = 5  # 设置为 -1 来处理文件中的所有电影
PAGES_PER_MOVIE = 2


# --- 配置结束 ---


def setup_driver():
    """配置并返回一个经过伪装和优化的WebDriver实例"""
    print("正在设置浏览器驱动 (无头模式)...")
    chrome_options = Options()

    # 启用无头模式，实现后台运行
    chrome_options.add_argument("--headless")

    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("log-level=3")
    chrome_options.page_load_strategy = 'eager'

    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32",
            webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)

    driver.set_page_load_timeout(45)
    print("驱动设置完成。")
    return driver


def convert_score(score_text):
    """将星级评分文本 (例如, '★★★★½') 转换为数字分数"""
    if not score_text:
        return ""
    score = 0.0
    score += score_text.count('★') * 1.0
    score += score_text.count('½') * 0.5
    return str(score) if score > 0 else ""


def parse_likes(likes_text):
    """从 'likes' 文本中提取数字 (例如, '703 likes')"""
    if not likes_text:
        return "0"
    match = re.search(r'([\d,]+)', likes_text)
    return match.group(1).replace(',', '') if match else "0"


def scrape_movie_reviews():
    """主函数，用于组织和执行评论抓取过程"""
    print(f"正在从 '{INPUT_CSV_FILE}' 读取电影列表...")
    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as f:
            movies_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_CSV_FILE}' 未找到。请检查文件路径。")
        return

    total_movies_in_file = len(movies_data)
    print(f"在文件中共找到 {total_movies_in_file} 部电影。")

    if MOVIES_TO_PROCESS == -1:
        movies_to_process = movies_data
        print("将处理文件中的所有电影。")
    else:
        movies_to_process = movies_data[:MOVIES_TO_PROCESS]
        print(f"根据配置，将处理前 {len(movies_to_process)} 部电影。")

    driver = setup_driver()
    all_reviews_data = []

    for i, movie in enumerate(movies_to_process):
        detail_url = movie.get('detail_url')
        movie_title = movie.get('original_title', '未知标题')

        if not detail_url:
            print(f"跳过电影 #{i + 1} ('{movie_title}')，因为缺少 detail_url。")
            continue

        print(f"\n--- 正在处理电影 {i + 1}/{len(movies_to_process)}: {movie_title} ---")

        for page_num in range(1, PAGES_PER_MOVIE + 1):
            reviews_url = f"{detail_url.strip('/')}/reviews/by/activity/page/{page_num}/"
            print(f"  - 正在抓取第 {page_num} 页: {reviews_url}")

            try:
                driver.get(reviews_url)

                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.listitem"))
                )

                review_elements = driver.find_elements(By.CSS_SELECTOR, "div.listitem")

                if not review_elements:
                    print("  - 此页面上未找到评论。将处理下一部电影。")
                    break

                print(f"  - 在此页面上找到 {len(review_elements)} 条评论。")

                for review_el in review_elements:
                    try:
                        nickname = review_el.find_element(By.CSS_SELECTOR, "strong.displayname").text
                    except NoSuchElementException:
                        nickname = ""

                    try:
                        score_text = review_el.find_element(By.CSS_SELECTOR, "span.rating").text
                        score = convert_score(score_text)
                    except NoSuchElementException:
                        score = ""

                    # 如果没有抓取到用户名或评分，则直接跳过此条评论
                    if not nickname or not score:
                        continue

                    try:
                        content = review_el.find_element(By.CSS_SELECTOR, "div.body-text p").text
                    except NoSuchElementException:
                        content = ""

                    try:
                        content_date = review_el.find_element(By.CSS_SELECTOR, "time.timestamp").get_attribute(
                            'datetime')
                    except NoSuchElementException:
                        content_date = ""

                    try:
                        likes_text = review_el.find_element(By.CSS_SELECTOR, "a[href*='/likes/']").text
                        approvals_num = parse_likes(likes_text)
                    except NoSuchElementException:
                        approvals_num = "0"

                    review_data = {
                        'imdb_id': movie.get('imdb_id', ''),
                        'original_title': movie.get('original_title', ''),
                        'release_year': movie.get('release_year', ''),
                        'nickname': nickname,
                        'score': score,
                        'score_max': '5',
                        'content': content,
                        'content_date': content_date,
                        'approvals_num': approvals_num
                    }
                    all_reviews_data.append(review_data)

            except TimeoutException:
                print(f"  - 页面加载超时或在第 {page_num} 页未找到 '{movie_title}' 的评论。跳过此电影。")
                break
            except Exception as e:
                print(f"  - 处理 '{movie_title}' 时发生意外错误: {e}")
                break

    driver.quit()
    print("\n浏览器驱动已关闭。")

    if not all_reviews_data:
        print("没有抓取到任何符合条件的评论数据，将不会创建输出文件。")
        return

    print(f"准备将 {len(all_reviews_data)} 条符合条件的评论写入 '{OUTPUT_CSV_FILE}'...")
    try:
        with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            headers = [
                'imdb_id', 'original_title', 'release_year', 'nickname', 'score',
                'score_max', 'content', 'content_date', 'approvals_num'
            ]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_reviews_data)
        print("成功将所有抓取到的评论写入CSV文件。")
    except IOError as e:
        print(f"无法写入文件 '{OUTPUT_CSV_FILE}'。原因: {e}")


if __name__ == "__main__":
    scrape_movie_reviews()