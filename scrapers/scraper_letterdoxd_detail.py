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
INPUT_CSV_FILE = 'movies_letterdoxd_url.csv'
OUTPUT_CSV_FILE = 'movies_letterdoxd_details_v10.csv'

# 这里的行号是针对数据行的，不包括CSV文件的表头。第1行指的就是第一条电影数据。
START_ROW = 1809  # 从第几行开始处理
END_ROW = 2008  # 处理到第几行结束（包含此行）


# --- 配置结束 ---

def setup_driver():
    """配置并返回一个经过伪装和优化的WebDriver实例"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.page_load_strategy = 'eager'

    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32",
            webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)

    driver.set_page_load_timeout(60)

    print("  - Performing driver warm-up...")
    driver.get("about:blank")
    time.sleep(1)

    return driver


# --- 数据提取辅助函数 ---
def clean_text(text):
    if text:
        return text.replace('\u00A0', ' ').strip()
    return ""


def safe_find_text(driver, by, selector):
    try:
        text = driver.find_element(by, selector).get_attribute('textContent')
        return clean_text(text)
    except NoSuchElementException:
        return ""


def safe_find_attribute(driver, by, selector, attribute):
    try:
        return driver.find_element(by, selector).get_attribute(attribute)
    except NoSuchElementException:
        return ""


def safe_find_multiple_texts(driver, by, selector, separator='/', exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = []
    try:
        elements = driver.find_elements(by, selector)
        valid_texts = []
        for el in elements:
            text = clean_text(el.get_attribute('textContent'))
            el_id = el.get_attribute('id')
            if text and el_id not in exclude_ids:
                valid_texts.append(text)
        return separator.join(valid_texts)
    except NoSuchElementException:
        return ""


# --- 主程序 ---
def scrape_movie_details():
    print(f"正在从 '{INPUT_CSV_FILE}' 读取电影列表...")
    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as f:
            movies_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_CSV_FILE}' 未找到。")
        return

    total_movies_in_file = len(movies_data)
    print(f"共找到 {total_movies_in_file} 部电影。")

    # --- 新增：根据指定的行数范围来切片数据 ---
    # 将用户提供的基于1的行号转换为基于0的列表索引
    start_index = max(0, START_ROW - 1)
    end_index = min(total_movies_in_file, END_ROW)

    if start_index >= end_index:
        print(f"错误: 开始行 ({START_ROW}) 必须小于结束行 ({END_ROW})。没有要处理的数据。")
        return

    movies_to_process = movies_data[start_index:end_index]
    total_to_process = len(movies_to_process)

    if total_to_process == 0:
        print("根据指定的行号范围，没有需要处理的电影。")
        return

    print(f"根据设置，将处理从第 {START_ROW} 行到第 {END_ROW} 行的 {total_to_process} 部电影。")
    # --- 修改结束 ---

    driver = setup_driver()
    print("浏览器驱动已启动 (Final Version)。")

    for i, movie in enumerate(movies_to_process):
        detail_url = movie.get('detail_url')
        if not detail_url: continue

        # --- 修改：优化日志输出，显示当前在整个文件中的行号 ---
        current_row_number = start_index + i + 1
        print(f"\n--- 正在处理第 {i + 1}/{total_to_process} 部 (文件总行号: {current_row_number}): {detail_url} ---")
        # --- 修改结束 ---

        try:
            # === 阶段 1: 抓取详情页信息 ===
            driver.get(detail_url)
            wait = WebDriverWait(driver, 40)
            poster_image_element = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "#js-poster-col .film-poster img:not([src*='empty-poster'])"))
            )
            print("  - 详情页加载成功, 提取主要数据...")

            original_title = safe_find_text(driver, By.CSS_SELECTOR, "em.quoted-creative-work-title")
            main_title = safe_find_text(driver, By.CSS_SELECTOR, "h1.headline-1 span.name")
            alternative_titles = main_title
            if not original_title:
                original_title = main_title

            release_year = safe_find_text(driver, By.CSS_SELECTOR, ".releasedate a")
            genres = safe_find_multiple_texts(driver, By.CSS_SELECTOR,
                                              "div#tab-genres div.text-sluglist.capitalize:first-of-type a")
            directors = safe_find_multiple_texts(driver, By.CSS_SELECTOR, "p.credits a[href*='/director/']")
            actors = safe_find_multiple_texts(driver, By.CSS_SELECTOR, "#tab-cast .cast-list a.text-slug",
                                              exclude_ids=['has-cast-overflow'])
            summary = safe_find_text(driver, By.CSS_SELECTOR, ".review .truncate p")
            length, imdb_id = "", ""
            footer_text = safe_find_text(driver, By.CSS_SELECTOR, "p.text-link.text-footer")
            if (length_match := re.search(r'(\d+)\s*mins', footer_text)): length = length_match.group(1)
            if (imdb_url := safe_find_attribute(driver, By.CSS_SELECTOR, "a.track-event[data-track-action='IMDb']",
                                                "href")):
                if (imdb_id_match := re.search(r'(tt\d+)', imdb_url)): imdb_id = imdb_id_match.group(1)
            rating, rating_max, rating_num = "", "5", ""
            if (rating_element_title := safe_find_attribute(driver, By.CSS_SELECTOR, "a.tooltip.display-rating",
                                                            "data-original-title")):
                rating = safe_find_text(driver, By.CSS_SELECTOR, "a.tooltip.display-rating")
                if (rating_num_match := re.search(r'based on ([\d,]+) ratings', rating_element_title)):
                    rating_num = rating_num_match.group(1).replace(",", "")
            poster_url = poster_image_element.get_attribute('src')
            backdrop_url = safe_find_attribute(driver, By.ID, "backdrop", "data-backdrop")

            # === 阶段 2: 抓取搜索页的额外别名 ===
            all_alternative_titles = []
            if alternative_titles:
                all_alternative_titles.append(alternative_titles)

            try:
                search_url = detail_url.replace('/film/', '/search/films/', 1)
                print(f"  - 正在导航至搜索页获取额外别名...")
                driver.get(search_url)

                selector_for_alt_titles = "div.film-metadata p"

                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector_for_alt_titles))
                )

                additional_titles_text = safe_find_text(driver, By.CSS_SELECTOR, selector_for_alt_titles)
                if "Alternative titles:" in additional_titles_text:
                    titles_part = additional_titles_text.split("Alternative titles:", 1)[1]
                    cleaned_titles_block = titles_part.replace("…more", "").rstrip(" ×").strip()
                    newly_found_titles = [title.strip() for title in cleaned_titles_block.split(',') if title.strip()]

                    for title in newly_found_titles:
                        if title not in all_alternative_titles:
                            all_alternative_titles.append(title)
                    print(f"  - 成功找到 {len(newly_found_titles)} 个额外别名。")
            except Exception as e:
                print(f"  - 未找到额外别名或搜索页加载失败，跳过。 (原因: {type(e).__name__})")

            # 【【THE FIX】】 Change the separator to "/"
            final_alternative_titles = "/".join(all_alternative_titles)

            # === 阶段 3: 更新并保存 ===
            movie.update({
                'imdb_id': imdb_id, 'original_title': original_title, 'alternative_titles': final_alternative_titles,
                'release_year': release_year, 'genres': genres, 'directors': directors, 'actors': actors,
                'summary': summary, 'length': length, 'rating': rating, 'rating_max': rating_max,
                'rating_num': rating_num, 'detail_url': detail_url, 'poster_url': poster_url,
                'backdrop_url': backdrop_url, 'status': 'Success'
            })
            print(f"  √ 成功提取并合并数据: 《{original_title or main_title}》")

        except Exception as e:
            print(f"  × 处理时发生未知错误: {e}")
            movie['status'] = f'Error: Unknown - {str(e)[:150]}'

    driver.quit()
    print("\n浏览器驱动已关闭。")

    if movies_to_process:
        # --- 修改：写入文件时，应使用原始数据的表头来确保完整性 ---
        headers = list(movies_data[0].keys())
        if 'status' not in headers: headers.append('status')
        print(f"准备将 {len(movies_to_process)} 行数据写入 '{OUTPUT_CSV_FILE}'...")
        with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(movies_to_process)
        print("写入完成！")
    # --- 修改结束 ---


if __name__ == "__main__":
    scrape_movie_details()