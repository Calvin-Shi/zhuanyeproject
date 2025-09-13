import csv
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def scrape_with_selenium():
    """
    使用Selenium进行动态内容爬取
    """
    # --- 基础配置 ---
    base_url = "https://letterboxd.com/films/popular/page/{}/"
    csv_file = "movies_letterdoxd.csv"
    headers_csv = [
        "imdb_id", "original_title", "alternative_titles", "release_year",
        "genres", "directors", "actors", "summary", "length", "rating",
        "rating_max", "rating_num", "detail_url", "poster_url", "backdrop_url"
    ]

    print("--- 爬虫程序开始 (Selenium模式) ---")

    # --- Selenium 设置 ---
    # 使用 webdriver-manager 自动配置ChromeDriver
    service = Service(ChromeDriverManager().install())
    # 创建Chrome浏览器实例
    driver = webdriver.Chrome(service=service)
    print("浏览器驱动已启动")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers_csv)
        print(f"CSV文件 '{csv_file}' 已创建并写入表头")

        # 爬取1-220页的电影详情页url数据，数量级为：万
        for page in range(1, 221):
            print(f"\n--- 正在处理第 {page} 页 ---")
            url = base_url.format(page)

            try:
                # 浏览器访问URL
                driver.get(url)
                print(f"浏览器已导航至: {url}")

                # 等待 class='poster-list' 的 ul 元素出现，最多等15秒
                wait = WebDriverWait(driver, 15)
                # 该元素为所有电影海报的容器，等待它加载出来就意味着JS已经执行完了
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "poster-list")))
                print("动态内容加载完成（检测到 'poster-list' 元素）")

                # 查找所有电影链接的<a>标签
                # 这些链接都在 class='frame' 的 <a> 标签里
                film_links = driver.find_elements(By.CSS_SELECTOR, "a.frame")

                print(f"在本页找到 {len(film_links)} 个电影链接")

                if not film_links:
                    print("页面加载完成但未找到电影链接")
                    continue

                for link_element in film_links:
                    # 获取<a>标签的 href 属性
                    detail_url = link_element.get_attribute('href')

                    movie_data = {header: "" for header in headers_csv}
                    movie_data["detail_url"] = detail_url

                    writer.writerow(list(movie_data.values()))

                print(f"已将本页的 {len(film_links)} 个链接全部写入CSV")

                # 暂停一下，避免请求过快
                sleep(1)

            except Exception as e:
                print(f"处理第 {page} 页时发生错误: {e}")
                sleep(3)

    # 关闭浏览器
    driver.quit()
    print("\n--- 爬虫程序执行完毕 ---")
    print(f"所有数据已保存至 {csv_file}")


# 运行主函数
if __name__ == "__main__":
    scrape_with_selenium()