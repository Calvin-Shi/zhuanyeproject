from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv


def get_douban_movie_info():

    # 设置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # 指定ChromeDriver路径
    driver_path = "D:\\chromedriver\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe" 
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 打开豆瓣电影页面
        url = 'https://movie.douban.com/explore?support_type=movie&is_all=false&category=%E7%83%AD%E9%97%A8&type=%E5%85%A8%E9%83%A8'
        driver.get(url)

        # 等待页面加载
        time.sleep(5)  

        movie_cards = driver.find_elements(By.CSS_SELECTOR, "ul.subject-list-list li")
        if not movie_cards:
            print("未找到电影卡片")
            return None

        movies = []

        for card in movie_cards:
            try:
                # 提取电影链接 
                link_element = card.find_element(By.TAG_NAME, "a")
                movie_url = link_element.get_attribute("href")

                # 提取电影名称
                name_span = card.find_element(By.CLASS_NAME, "drc-subject-info-title-text")
                movie_name = name_span.text.strip()

                # 提取副标题信息
                subtitle_div = card.find_element(By.CLASS_NAME, "drc-subject-info-subtitle")
                subtitle_text = subtitle_div.text.strip()

                # 解析副标题
                parts = [p.strip() for p in subtitle_text.split('/')]
                year = parts[0] if len(parts) > 0 else "未知"
                country = parts[1] if len(parts) > 1 else "未知"
                genre = ' '.join(parts[2:-2]) if len(parts) > 3 else "未知"
                director = parts[-2] if len(parts) > 2 else "未知"
                actors = parts[-1] if len(parts) > 1 else "未知"

                # 提取评分
                rating_span = card.find_element(By.CLASS_NAME, "drc-rating-num")
                rating = rating_span.text.strip()

                # 添加到列表
                movies.append({
                    "电影名": movie_name,
                    "链接": movie_url,
                    "年份": year,
                    "国家": country,
                    "类型": genre,
                    "导演": director,
                    "主演": actors,
                    "评分": rating
                })
            except Exception as inner_e:
                print(f"提取单个电影信息时发生异常: {inner_e}")
                continue

        return movies

    except Exception as e:
        print(f"发生异常: {e}")
        return None

    finally:
        # 关闭浏览器
        driver.quit()


if __name__ == '__main__':
    movie_list = get_douban_movie_info()
    if movie_list:
        # 保存到CSV文件
        csv_filename = "douban_remen_movies.csv"
        fieldnames = ["电影名", "链接", "年份", "国家", "类型", "导演", "主演", "评分"]
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(movie_list)

        print(f"已成功爬取并保存到 {csv_filename} 文件中。")
    else:
        print("爬取失败。请检查网络连接或网站结构是否已更改。")