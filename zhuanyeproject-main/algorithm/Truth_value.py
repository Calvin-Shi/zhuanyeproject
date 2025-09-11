# -*- coding: utf-8 -*-
"""
真值层流水线（适配项目结构）：
- 读取：data/movies.csv, data/reviews_douban.csv
- 产出：truth_value_out/ 下的 item_quality.csv, interactions_gt.csv, splits.csv, eval_samples.csv
放置位置：algorithm/Truth_value.py
运行方式（在项目根目录）：
    python -m algorithm.Truth_value
或：
    python algorithm/Truth_value.py
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# 现在文件在 algorithm/ 下，项目根目录是其上一层
ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR / "data"
OUT_DIR    = ROOT_DIR / "truth_value_out"
OUT_DIR.mkdir(exist_ok=True, parents=True)

IN_MOVIES  = DATA_DIR / "movies.csv"
IN_REVIEWS = DATA_DIR / "reviews_douban.csv"

# 工具函数：把分数/人数/时间戳转成数值；统一 user_id/item_id
def _to_float(x):
    try:
        return float(str(x).strip())
    except:
        return np.nan

def _to_int(x):
    s = str(x).strip().replace(",", "")
    s = "".join(ch for ch in s if ch.isdigit())
    return int(s) if s else np.nan

def _parse_dt(s: str):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return pd.NaT

def _read_csv_smart(path: Path) -> pd.DataFrame:
    """兼容带 BOM 的 UTF-8 文件"""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

# 读取与清洗
def step1_load_clean():
    # 电影表
    movies = _read_csv_smart(IN_MOVIES)
    # 均分是 0~10；爬虫里已经把 star = 均分/2（0~5），这里统一兜底
    movies["douban_average_score"] = movies["douban_average_score"].apply(_to_float)
    movies["douban_star_rating"]   = movies["douban_star_rating"].apply(_to_float)
    movies["number_of_ratings"]    = movies["number_of_ratings"].apply(_to_int)

    # 若 star 缺失，用 average_score/2 兜底
    mask = movies["douban_star_rating"].isna()
    movies.loc[mask, "douban_star_rating"] = movies.loc[mask, "douban_average_score"] / 2.0

    # 评论表
    reviews = _read_csv_smart(IN_REVIEWS)
    reviews["score"] = reviews["score"].apply(_to_float)
    reviews["ts"]    = reviews["comment_time"].apply(_parse_dt)
    reviews["user_id"] = reviews["author"].astype(str)
    reviews["item_id"] = reviews["imdb_id"].astype(str)

    return movies, reviews

# 电影“质量真值”——贝叶斯校准
def step2_item_quality(movies: pd.DataFrame, C: int = 80) -> pd.DataFrame:
    """
    s_hat_5 = (m*C + R*N) / (C+N)
    R: 电影的平均星级(0~5)；N:评分人数；m:全局先验均值；C:先验强度
    """
    R = movies["douban_star_rating"]
    N = movies["number_of_ratings"].fillna(0)
    m = R.mean()

    # 避免除零：当 N=0 时直接回退到 m
    s_hat_5 = (m * C + R * N) / (C + N.replace(0, np.nan))
    s_hat_5 = s_hat_5.fillna(m).round(3)

    item_quality = movies[[
        "imdb_id", "original_title", "release_year", "genres",
        "douban_average_score", "douban_star_rating", "number_of_ratings"
    ]].copy()
    item_quality.rename(columns={"imdb_id": "item_id"}, inplace=True)
    item_quality["s_hat_5"] = s_hat_5
    item_quality["prior_m"] = round(m, 3)
    item_quality["C"] = C

    item_quality.to_csv(OUT_DIR / "item_quality.csv", index=False)
    print(f"[OK] {OUT_DIR/'item_quality.csv'} -> {len(item_quality)} rows")
    return item_quality

#  用户→电影“偏好真值”——显式 + 隐式
def _implicit_weight(status: str) -> float:
    """
    隐式行为的置信度权重：
      想看=0.3，看过=0.6，未知=0.1（也可以选择丢弃未知，见下方注释）
    """
    if not isinstance(status, str):
        return 0.0
    if "想看" in status:
        return 0.3
    if "看过" in status:
        return 0.6
    if "未知" in status:
        return 0.1
    return 0.0

def step3_interactions_gt(reviews: pd.DataFrame) -> pd.DataFrame:
    # 显式评分：≥4 为正，≤2 为负；3星丢弃
    exp = reviews.dropna(subset=["score"]).copy()
    exp = exp[(exp["score"] >= 4.0) | (exp["score"] <= 2.0)]
    exp["y"] = (exp["score"] >= 4.0).astype(int)
    exp["weight"] = 1.0
    exp["label_source"] = "explicit_rating"
    exp = exp[["user_id", "item_id", "ts", "y", "weight", "label_source", "score"]]

    # 隐式行为：想看/看过 → 弱正样本；“未知”给很低权重(0.1)
    imp = reviews[reviews["score"].isna()].copy()

    imp["y"] = np.where(imp["user_status"].astype(str).str.contains("想看"), 1,
                 np.where(imp["user_status"].astype(str).str.contains("看过"), 1,
                     np.where(imp["user_status"].astype(str).str.contains("未知"), 1, np.nan)))
    imp = imp.dropna(subset=["y"])
    imp["weight"] = imp["user_status"].apply(_implicit_weight)
    imp["label_source"] = "implicit_status"
    imp["score"] = np.nan
    imp = imp[["user_id", "item_id", "ts", "y", "weight", "label_source", "score"]]

    # 合并 + 时间兜底 + 去重（保留最新）
    df = pd.concat([exp, imp], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    df = df.sort_values(["user_id", "item_id", "ts"]).drop_duplicates(["user_id", "item_id"], keep="last")

    df.to_csv(OUT_DIR / "interactions_gt.csv", index=False)
    print(f"[OK] {OUT_DIR/'interactions_gt.csv'} -> {len(df)} rows (pos={int((df.y==1).sum())}, neg={int((df.y==0).sum())})")
    return df

#  时序切分（train/val/test）
def step4_time_splits(interactions_gt: pd.DataFrame) -> pd.DataFrame:
    """
    每个用户按时间排序：
      - 最后一个正样本 -> test
      - 倒数第二个正样本(若有) -> val
      - 其他 -> train
    """
    df = interactions_gt.copy().sort_values(["user_id", "ts"])

    pos = df[df["y"] == 1]
    last_idx = pos.groupby("user_id").tail(1).index
    # 仅对“有 ≥ 2 个正样本”的用户取倒数第二个
    second_last_idx = (
        pos.groupby("user_id")
          .apply(lambda g: g.iloc[-2].name if len(g) >= 2 else None)
          .dropna()
          .astype(int)
          .values
    )

    splits = pd.Series("train", index=df.index)
    splits.loc[second_last_idx] = "val"
    splits.loc[last_idx] = "test"
    df["split"] = splits.values

    df.to_csv(OUT_DIR / "interactions_gt.csv", index=False)
    df[["user_id", "item_id", "ts", "y", "weight", "label_source", "split"]].to_csv(OUT_DIR / "splits.csv", index=False)

    print(f"[OK] splits -> train={sum(df.split=='train')}  val={sum(df.split=='val')}  test={sum(df.split=='test')}")
    return df

# 离线评测负采样（HR@K/NDCG@K）
def step5_eval_samples(movies: pd.DataFrame, interactions: pd.DataFrame, K: int = 50) -> pd.DataFrame:
    """
    为每个 test 正样本采 K 个负例：
      - 负例按 item 流行度做概率采样（pop^0.5），并过滤用户已看/正样本
    """
    rng = np.random.default_rng(42)
    all_items = movies["imdb_id"].astype(str).unique()

    # 流行度（交互计数 + 1 平滑）
    item_pop = interactions.groupby("item_id").size().rename("cnt")
    pop = pd.Series(0.0, index=pd.Index(all_items, name="item_id"))
    pop.loc[item_pop.index] = item_pop.values
    pop = pop + 1.0
    p = (pop ** 0.5)
    p = p / p.sum()

    # 用户已看集合
    user_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []
    for u, pos_i in interactions[interactions["split"] == "test"][["user_id", "item_id"]].itertuples(index=False):
        seen = user_items.get(u, set())
        cand = rng.choice(all_items, size=K * 5, replace=True, p=p.loc[all_items].values)
        negs = [c for c in cand if c not in seen and c != pos_i][:K]
        if len(negs) < K:
            fill = [x for x in all_items if (x not in seen and x != pos_i)]
            rng.shuffle(fill)
            negs += fill[:(K - len(negs))]
        rows.append({"user_id": u, "pos_item_id": pos_i, **{f"neg_{i+1}": v for i, v in enumerate(negs)}})

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(OUT_DIR / "eval_samples.csv", index=False)
    print(f"[OK] {OUT_DIR/'eval_samples.csv'} -> {len(eval_df)} cases, K={K}")
    return eval_df

# main
def main():
    print(f"[INFO] ROOT_DIR={ROOT_DIR}")
    print(f"[INFO] DATA_DIR={DATA_DIR}")
    print(f"[INFO] OUT_DIR ={OUT_DIR}")

    movies, reviews = step1_load_clean()
    item_quality = step2_item_quality(movies, C=80)
    interactions = step3_interactions_gt(reviews)
    interactions = step4_time_splits(interactions)
    _ = step5_eval_samples(movies, interactions, K=50)

    # 小结
    print("\n[SUMMARY]")
    print(f"  movies      : {len(movies)}")
    print(f"  interactions: {len(interactions)} (pos={int((interactions.y==1).sum())}, neg={int((interactions.y==0).sum())})")
    print(f"  splits      : {interactions['split'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
