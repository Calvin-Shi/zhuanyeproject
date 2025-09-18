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

IN_MOVIES  = DATA_DIR / "movies_letterdoxd_details_merged_v1.csv"
IN_REVIEWS = DATA_DIR / "reviews_letterdoxd_merged_v1.csv"

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
def load_clean():
    # --- 电影详情（列：imdb_id, original_title, alternative_titles, release_year, genres, directors, actors,
    #                summary, length, rating, rating_max, rating_num, detail_url, poster_url, backdrop_url, status）
    movies = _read_csv_smart(IN_MOVIES)
    movies.columns = [c.strip() for c in movies.columns]

    # 基础清洗
    movies["rating"]     = movies["rating"].apply(_to_float)         # 0~5
    movies["rating_num"] = movies["rating_num"].apply(_to_int).fillna(0).astype(int)

    # ——关键：构造 (规范化标题, 年份) 的“唯一”映射——
    # 1) 生成 title 列表：原名 + 别名（按 '/' 分割）
    def _norm(s):  # 统一大小写与空白
        return str(s).strip().casefold()

    movies["original_title_norm"] = movies["original_title"].apply(_norm)
    alt = movies.get("alternative_titles", "")
    movies["alt_list"] = alt.fillna("").astype(str).apply(lambda s: [t.strip() for t in s.split("/") if t.strip()])

    # 2) 展开成多行（每个 title 一个键），带上年份与 imdb_id / rating_num 作决策依据
    rows = []
    for idx, row in movies.iterrows():
        year = str(row.get("release_year", "")).strip()
        imdb = row.get("imdb_id", "")
        rn   = int(row.get("rating_num", 0)) if pd.notna(row.get("rating_num", 0)) else 0
        # 原名
        if row["original_title_norm"]:
            rows.append({"title_norm": row["original_title_norm"], "release_year": year, "imdb_id": imdb, "rating_num": rn})
        # 别名
        for t in row["alt_list"]:
            rows.append({"title_norm": _norm(t), "release_year": year, "imdb_id": imdb, "rating_num": rn})

    if not rows:
        raise RuntimeError("电影详情展开为空，请检查 movies CSV。")

    mv_keys = pd.DataFrame(rows)
    # 3) 对 (title_norm, release_year) 分组，保留 rating_num 最大的那条，得到**唯一**映射
    mv_keys = mv_keys.sort_values(["title_norm", "release_year", "rating_num"], ascending=[True, True, False])
    mv_keys_unique = mv_keys.drop_duplicates(subset=["title_norm", "release_year"], keep="first")[["title_norm", "release_year", "imdb_id"]]

    # --- 评论（列：original_title, release_year, nickname, score, score_max, content, content_date, approvals_num）
    reviews = _read_csv_smart(IN_REVIEWS)
    reviews.columns = [c.strip() for c in reviews.columns]

    req_cols = {"original_title","release_year","nickname","score","content_date","approvals_num"}
    missing = [c for c in req_cols if c not in reviews.columns]
    if missing:
        raise KeyError(f"评论表缺少所需列：{missing}；当前列：{list(reviews.columns)}")

    # 基础清洗
    reviews["score"]   = reviews["score"].apply(_to_float)           # 0~5
    reviews["ts"]      = pd.to_datetime(reviews["content_date"], errors="coerce")
    reviews["user_id"] = reviews["nickname"].astype(str)
    reviews["approvals_num"] = reviews["approvals_num"].apply(_to_int).fillna(0)

    # 规范化标题 + 年份，准备 merge
    reviews["title_norm"]   = reviews["original_title"].apply(_norm)
    reviews["release_year"] = reviews["release_year"].astype(str).str.strip()

    # ——用 merge 回填 imdb_id → item_id（避免 index 唯一性问题）——
    reviews = reviews.merge(
        mv_keys_unique,
        on=["title_norm", "release_year"],
        how="left",
        validate="m:1"  # 每条评论最多匹配到一部电影
    )

    # 统计未匹配
    miss_cnt = reviews["imdb_id"].isna().sum()
    if miss_cnt > 0:
        print(f"[WARN] 有 {miss_cnt} 条评论无法通过 (title, year) 匹配到 imdb_id，将被丢弃并记录到 truth_value_out/unmatched_reviews.csv")
        unmatched = reviews[reviews["imdb_id"].isna()].copy()
        unmatched_out = OUT_DIR / "unmatched_reviews.csv"
        unmatched.to_csv(unmatched_out, index=False)

    # 丢弃未匹配项，并生成 item_id
    reviews = reviews.dropna(subset=["imdb_id"]).copy()
    reviews["item_id"] = reviews["imdb_id"].astype(str)

    # 只保留下游使用的列（其余列保留也无妨，不影响后续）
    # return 原 movies（后续需要 imdb_id/rating/rating_num 等），以及带 item_id 的 reviews
    return movies, reviews





# 电影“质量真值”——贝叶斯校准
def item_quality(movies: pd.DataFrame, C: int = 80) -> pd.DataFrame:
    """
    s_hat_5 = (m*C + R*N) / (C+N)
    R: Letterboxd 星级(0~5)；N:评分人数；m:全局均值；C:先验强度
    """
    R = movies["rating"]
    N = movies["rating_num"].fillna(0)
    m = R.mean()

    s_hat_5 = (m * C + R * N) / (C + N.replace(0, np.nan))
    s_hat_5 = s_hat_5.fillna(m).round(3)

    item_quality = movies[[
        "imdb_id", "original_title", "release_year", "genres",
        "rating", "rating_num"
    ]].copy()
    item_quality.rename(columns={"imdb_id": "item_id"}, inplace=True)
    item_quality["s_hat_5"] = s_hat_5
    item_quality["prior_m"] = round(m, 3)
    item_quality["C"] = C

    # ===== 在这里加：按 item_id 去重，保留 rating_num 最大的一条 =====
    item_quality = (
        item_quality
        .assign(rating_num=item_quality["rating_num"].fillna(0).astype(float))
        .sort_values(["item_id", "rating_num"], ascending=[True, False])
        .drop_duplicates(subset=["item_id"], keep="first")
    )
    # ===============================================================

    item_quality.to_csv(OUT_DIR / "item_quality.csv", index=False)
    print(f"[OK] {OUT_DIR/'item_quality.csv'} -> {len(item_quality)} rows")
    return item_quality



#  用户→电影“偏好真值”——显式
def interactions_gt(reviews: pd.DataFrame) -> pd.DataFrame:
    # 只用显式评分；3星丢弃
    exp = reviews.dropna(subset=["score"]).copy()
    exp = exp[(exp["score"] >= 4.0) | (exp["score"] <= 2.0)]
    exp["y"] = (exp["score"] >= 4.0).astype(int)
    exp["label_source"] = "explicit_rating"

    # ---- 点赞数加权：1 + log1p(approvals_num) → 裁剪 → 归一 ----
    a = exp["approvals_num"].fillna(0).astype(float)
    w_raw = 1.0 + np.log1p(a)                     # 基础：对数缩放
    cap = np.nanpercentile(w_raw, 99)             # 99分位裁剪（稳健）
    w_cap = np.minimum(w_raw, cap)

    mean_target = 1.0                              # 希望整体均值≈1（与原先一致）
    w = w_cap / (np.nanmean(w_cap) + 1e-8) * mean_target
    exp["weight"] = w.clip(0.1, None)             # 下限0.1，防止出现过小权重

    # 收尾：列选择与去重
    df = exp[["user_id", "item_id", "ts", "y", "weight", "label_source", "score"]].copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    df = df.sort_values(["user_id", "item_id", "ts"]).drop_duplicates(["user_id", "item_id"], keep="last")

    df.to_csv(OUT_DIR / "interactions_gt.csv", index=False)
    print(f"[OK] {OUT_DIR/'interactions_gt.csv'} -> {len(df)} rows (pos={int((df.y==1).sum())}, neg={int((df.y==0).sum())})")
    print("[INFO] weight stats:",
          f"min={df['weight'].min():.3f}, p50={df['weight'].median():.3f}, "
          f"mean={df['weight'].mean():.3f}, p95={df['weight'].quantile(0.95):.3f}, max={df['weight'].max():.3f}")
    return df





#  时序切分（train/val/test）
def time_splits(interactions_gt: pd.DataFrame) -> pd.DataFrame:
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
def eval_samples(movies: pd.DataFrame, interactions: pd.DataFrame, K: int = 50,
                 item_quality_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    all_items = movies["imdb_id"].astype(str).unique()

    # 流行度（交互计数 + 1 平滑）
    item_pop = interactions.groupby("item_id").size().rename("cnt")
    pop = pd.Series(0.0, index=pd.Index(all_items, name="item_id"))
    pop.loc[item_pop.index] = item_pop.values
    pop = pop + 1.0

    # ——NEW：融合质量分（s_hat_5）——
    if item_quality_df is not None and "s_hat_5" in item_quality_df.columns:
        q = item_quality_df.set_index("item_id")["s_hat_5"].reindex(all_items).astype(float)
        q = q.fillna(q.mean() if not np.isnan(q.mean()) else 2.5)  # 兜底
        beta = 0.5  # 质量权重强度，可调：0~1 常用
        p = (pop ** 0.5) * (np.maximum(q, 0.1) ** beta)
    else:
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
            fill = rng.permutation(fill).tolist()  # ← 你已修正
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

    movies, reviews = load_clean()
    item_quality_df = item_quality(movies, C=80)
    inter_df = interactions_gt(reviews)
    inter_df = time_splits(inter_df)
    _ = eval_samples(movies, inter_df, K=50,item_quality_df=item_quality_df)

    # 小结
    print("\n[SUMMARY]")
    print(f"  movies      : {len(movies)}")
    print(f"  interactions: {len(inter_df)} (pos={int((inter_df.y==1).sum())}, neg={int((inter_df.y==0).sum())})")
    print(f"  splits      : {inter_df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
