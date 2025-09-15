import pandas as pd

# --- 配置 ---
# 输入的原始CSV文件名
input_filename = 'movies_letterdoxd_details_merged_v1.csv'
# 输出的新CSV文件名
output_filename = 'extracted_movies.csv'
# 需要提取的列的列表
columns_to_extract = ['imdb_id', 'original_title']

# --- 主程序 ---
try:
    # 使用pandas读取CSV文件
    print(f"正在读取文件: {input_filename}...")
    df = pd.read_csv(input_filename)

    # 从数据框（DataFrame）中选择我们需要的列
    print(f"正在提取指定的列: {', '.join(columns_to_extract)}...")
    extracted_df = df[columns_to_extract]

    # 将提取出的数据保存到一个新的CSV文件
    # 设置 index=False 是为了不在新文件中添加一列无用的行号索引
    extracted_df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"操作完成！数据已成功保存到文件: {output_filename}")

except FileNotFoundError:
    print(f"错误：输入文件未找到！")
    print(f"请确认名为 '{input_filename}' 的文件存在于脚本所在的同一目录下。")
except KeyError:
    # 如果CSV文件中没有指定的列名，会触发此错误
    print(f"错误：在文件中找不到指定的列。")
    print(f"请检查 '{input_filename}' 文件中是否确实包含 'imdb_id' 和 'original_title' 这两列。")
except Exception as e:
    # 捕获其他可能的异常
    print(f"在处理过程中发生了未知错误: {e}")