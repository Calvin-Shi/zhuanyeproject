from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv

def get_douban_movie_comments():

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
        # 打开指定的电影页面
        url = 'https://movie.douban.com/subject/36767890/'
        driver.get(url)

        # 等待页面加载
        time.sleep(5)

        # 获取电影名和年份
        movie_title_element = driver.find_element(By.CSS_SELECTOR, '#content h1 span:first-child')
        movie_title = movie_title_element.text.strip()
        movie_year_element = driver.find_element(By.CSS_SELECTOR, '#content h1 span.year')
        movie_year = movie_year_element.text.strip('()')

        print(f"电影: {movie_title} ({movie_year})")

        # 找到所有评论项
        comment_items = driver.find_elements(By.CSS_SELECTOR, "#comments-section .comment-item")

        if not comment_items:
            print("未找到评论。")
            return None

        comments = []

        for item in comment_items:
            # 使用 try-except 块处理可能找不到的元素，避免程序崩溃
            try:
                # 提取评论作者
                author_element = item.find_element(By.CSS_SELECTOR, ".comment-info a")
                comment_author = author_element.text.strip()
            except:
                comment_author = "未知作者"

            try:
                # 提取评论评分
                rating_element = item.find_element(By.CSS_SELECTOR, ".comment-info [class^='allstar']")
                comment_rating = rating_element.get_attribute("title")
            except:
                comment_rating = "无评分"

            try:
                # 尝试不同的选择器来获取评论内容
                content_element = item.find_element(By.CSS_SELECTOR, "p.comment-content")
                comment_content = content_element.text.strip()
            except:
                try:
                    content_div = item.find_element(By.CSS_SELECTOR, "div.comment-content")
                    comment_content = content_div.text.strip()
                except:
                    comment_content = "无评论内容"

            # 按照标准化的表头名称将数据添加到列表中
            comments.append({
                "original_title": movie_title,
                "release_year": movie_year,
                "author": comment_author,
                "content": comment_content,
                "score": comment_rating
            })

        return comments

    except Exception as e:
        print(f"发生异常: {e}")
        return None

    finally:
        # 关闭浏览器
        driver.quit()

if __name__ == '__main__':
    comment_list = get_douban_movie_comments()
    if comment_list:
        # 保存到CSV文件，使用标准化的命名和表头
        csv_filename = "reviews_douban.csv"
        fieldnames = ["original_title", "release_year", "author", "content", "score"]
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comment_list)

        print(f"已成功爬取并保存到 {csv_filename} 文件中。")
    else:
        print("爬取失败。请检查网络连接或网站结构是否已更改。")