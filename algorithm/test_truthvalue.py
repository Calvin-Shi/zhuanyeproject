# -*- coding: utf-8 -*-
"""
验证 Truth_value.py 全链路产物（Letterboxd 版本）
在项目根目录运行：
    python validate_truth_value_letterboxd.py
会读取：
- data/movies_letterdoxd_details_merged_v1.csv
- data/reviews_letterdoxd_merged_v1.csv
- truth_value_out/item_quality.csv
- truth_value_out/interactions_gt.csv
- truth_value_out/splits.csv
- truth_value_out/eval_samples.csv
并输出：
- truth_value_out/validation_report.md（摘要报告）
- 若干图：item_quality_scatter.png / weight_hist.png / splits_pie.png / neg_pop_hist.png
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "truth_value_out"
OUT.mkdir(exist_ok=True, parents=True)

IN_MOVIES  = DATA / "movies_letterdoxd_details_merged_v1.csv"
IN_REVIEWS = DATA / "reviews_letterdoxd_merged_v1.csv"
OUT_ITEMQ  = OUT / "item_quality.csv"
OUT_INTER  = OUT / "interactions_gt.csv"
OUT_SPLITS = OUT / "splits.csv"
OUT_EVAL   = OUT / "eval_samples.csv"
REPORT     = OUT / "validation_report.md"

# ---------- utils ----------
def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

# ---------- 1) 基础一致性：键/去重/缺失 ----------
def check_consistency(movies: pd.DataFrame, reviews: pd.DataFrame, itemq: pd.DataFrame) -> dict:
    res = {}
    # item_quality 的主键唯一性
    dup_ids = itemq["item_id"].value_counts()
    res["item_quality_dup_item_ids"] = int((dup_ids > 1).sum())

    # movies 内部 imdb_id 是否唯一（用于溯源问题）
    if "imdb_id" in movies.columns:
        mv_dup = movies["imdb_id"].value_counts()
        res["movies_dup_imdb_id"] = int((mv_dup > 1).sum())
    else:
        res["movies_dup_imdb_id"] = None

    # reviews 必备列 & 缺失
    req_cols = ["user_id","item_id","score","ts"]
    res["reviews_missing_counts"] = {c:int(reviews[c].isna().sum()) if c in reviews.columns else None for c in req_cols}

    # unmatched 导出文件是否存在（来自主流程）
    res["unmatched_file_exists"] = (OUT/"unmatched_reviews.csv").exists()
    if res["unmatched_file_exists"]:
        try:
            res["unmatched_rows"] = len(_read(OUT/"unmatched_reviews.csv"))
        except Exception:
            res["unmatched_rows"] = None
    return res

# ---------- 2) item_quality：平滑前后对比 ----------
def validate_item_quality(itemq: pd.DataFrame) -> dict:
    res = {}
    # 直接用 item_quality.csv 自带列
    cols = {c.strip().casefold(): c for c in itemq.columns}
    raw_col = cols.get("rating"); cal_col = cols.get("s_hat_5"); num_col = cols.get("rating_num")
    if raw_col is None or cal_col is None:
        raise KeyError(f"item_quality.csv 缺少 rating 或 s_hat_5，现有列={list(itemq.columns)}")

    df = itemq[[raw_col, cal_col] + ([num_col] if num_col else [])].copy()
    df.rename(columns={raw_col: "raw", cal_col: "cal"}, inplace=True)
    if num_col:
        df.rename(columns={num_col: "rating_num"}, inplace=True)
        df["rating_num"] = pd.to_numeric(df["rating_num"], errors="coerce").fillna(0)
    else:
        df["rating_num"] = 1.0

    # 长尾（<=50）/ 头部（>=1000）可按你数据分布调整
    tail = df[df["rating_num"] <= 50]
    head = df[df["rating_num"] >= 1000]

    # 指标：样本量>=2才计算方差，否则返回 NaN，避免DoF警告
    res["pearson_raw_cal"] = float(df[["raw", "cal"]].corr().iloc[0, 1])
    res["tail_var_raw"] = float(np.nanvar(tail["raw"])) if len(tail) >= 2 else np.nan
    res["tail_var_cal"] = float(np.nanvar(tail["cal"])) if len(tail) >= 2 else np.nan
    res["mean_raw"], res["mean_cal"] = float(np.nanmean(df["raw"])), float(np.nanmean(df["cal"]))

    # 散点图（点大小与 rating_num 相关）
    plt.figure(figsize=(6, 5))
    size = 10 + 40 * np.clip(np.log1p(df["rating_num"]) / np.log(1000 + 1), 0, 1)
    plt.scatter(df["raw"], df["cal"], s=size)
    plt.plot([0, 5], [0, 5])
    plt.xlabel("raw rating (0-5)"); plt.ylabel("calibrated s_hat_5 (0-5)")
    plt.title("Item quality: raw vs calibrated")
    p = OUT / "item_quality_scatter.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    res["fig_item_quality_scatter"] = str(p)
    return res



# ---------- 3) interactions：标签/权重/唯一性 ----------
def validate_interactions(inter: pd.DataFrame) -> dict:
    res = {}
    # 正负分布
    res["pos_neg_counts"] = inter["y"].value_counts().to_dict()
    # 权重
    w = pd.to_numeric(inter["weight"], errors="coerce")
    res["weight_stats"] = {
        "min": float(np.nanmin(w)),
        "p50": float(np.nanpercentile(w,50)),
        "mean": float(np.nanmean(w)),
        "p95": float(np.nanpercentile(w,95)),
        "max": float(np.nanmax(w)),
    }
    # 唯一性
    res["dup_user_item_pairs"] = int(inter.duplicated(["user_id","item_id"], keep=False).sum())

    # 直方图
    plt.figure(figsize=(6,4))
    plt.hist(w.dropna(), bins=30)
    plt.xlabel("weight"); plt.ylabel("count"); plt.title("Interaction weight distribution")
    p = OUT/"weight_hist.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    res["fig_weight_hist"] = str(p)
    return res

# ---------- 4) splits：用户级留后一核验 ----------
def validate_splits(inter: pd.DataFrame) -> dict:
    res = {}
    res["split_counts"] = inter["split"].value_counts().to_dict()

    df = inter.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    pos = df[df["y"]==1].sort_values(["user_id","ts"])

    # 每用户最后一条正样本应在 test
    last_pos = pos.groupby("user_id").tail(1)
    res["last_pos_in_test_pct"] = float((last_pos["split"]=="test").mean()) if len(last_pos)>0 else np.nan

    # 倒数第二条正样本应在 val（若有≥2条）
    second_last = (
        pos.groupby("user_id").tail(2)
           .groupby("user_id").head(1)
    )
    res["second_last_in_val_pct"] = float((second_last["split"]=="val").mean()) if len(second_last)>0 else np.nan

    # 饼图
    plt.figure(figsize=(5,5))
    df["split"].value_counts().plot(kind="pie", autopct="%.1f%%")
    plt.title("Split distribution"); plt.ylabel("")
    p = OUT/"splits_pie.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    res["fig_splits_pie"] = str(p)
    return res

# ---------- 5) eval_samples：覆盖率/违规/负样本分布 ----------
def validate_eval(inter: pd.DataFrame, evaldf: pd.DataFrame) -> dict:
    res = {}
    # test 正样本集合
    test_pos = inter[(inter["split"] == "test") & (inter["y"] == 1)][["user_id", "item_id"]] \
                  .rename(columns={"item_id": "pos_item_id"})

    # 覆盖率：左连接 + indicator，看有多少行不是left_only（即匹配成功）
    cov = evaldf.merge(test_pos, on=["user_id", "pos_item_id"], how="left", indicator=True)
    res["coverage_pct"] = float((cov["_merge"] != "left_only").mean()) if len(cov) > 0 else np.nan

    # 每个样本的负例个数
    neg_cols = [c for c in evaldf.columns if c.startswith("neg_")]
    if neg_cols:
        k = evaldf[neg_cols].notna().sum(axis=1)
        res["neg_count_min"], res["neg_count_max"] = int(k.min()), int(k.max())
    else:
        res["neg_count_min"], res["neg_count_max"] = 0, 0

    # 违规：负例=正例 / 负例落在已看集合
    seen_map = inter.groupby("user_id")["item_id"].apply(set).to_dict()
    v_pos_eq, v_in_seen = 0, 0
    for _, row in evaldf.iterrows():
        u = row["user_id"]; pos = row["pos_item_id"]
        negs = [row[c] for c in neg_cols if pd.notna(row[c])]
        v_pos_eq += sum(1 for x in negs if x == pos)
        seen = seen_map.get(u, set())
        v_in_seen += sum(1 for x in negs if x in seen)
    res["violations_pos_eq"] = int(v_pos_eq)
    res["violations_neg_in_seen"] = int(v_in_seen)

    # 负样本流行度分布（与全局对比）
    pop = inter["item_id"].value_counts()
    neg_items = pd.unique(evaldf[neg_cols].values.ravel()) if neg_cols else np.array([])
    neg_items = neg_items[pd.notna(neg_items)]
    neg_pop = pop.reindex(neg_items).dropna().astype(int)

    plt.figure(figsize=(6, 4))
    plt.hist(np.log1p(pop.values), bins=30, alpha=0.5, label="all items")
    plt.hist(np.log1p(neg_pop.values), bins=30, alpha=0.5, label="negatives")
    plt.xlabel("log1p(popularity)"); plt.ylabel("count"); plt.title("Popularity: negatives vs all")
    plt.legend()
    p = OUT / "neg_pop_hist.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    res["fig_neg_pop_hist"] = str(p)
    return res


# ---------- report ----------

def write_report(sections: dict):
    lines = []
    lines.append("# Validation Report (Truth Value Pipeline)\n\n")

    c = sections.get("cons", {})
    lines += ["## 1. 基础一致性\n",
              f"- item_quality 主键重复个数：{c.get('item_quality_dup_item_ids')}\n",
              f"- movies 中 imdb_id 重复个数：{c.get('movies_dup_imdb_id')}\n",
              f"- reviews 缺失计数：{c.get('reviews_missing_counts')}\n",
              f"- unmatched_reviews.csv 存在：{c.get('unmatched_file_exists')}，行数：{c.get('unmatched_rows')}\n\n"]

    q = sections.get("itemq", {})
    lines += ["## 2. 电影质量校准\n",
              f"- 皮尔逊相关（raw vs s_hat_5）：{q.get('pearson_raw_cal')}\n",
              f"- 长尾方差：raw={q.get('tail_var_raw')} / cal={q.get('tail_var_cal')}（越低越稳健）\n",
              f"- 均值：raw={q.get('mean_raw')} / cal={q.get('mean_cal')}\n",
              f"- 图：{q.get('fig_item_quality_scatter')}\n\n"]

    it = sections.get("inter", {})
    lines += ["## 3. 交互真值\n",
              f"- 正负样本分布：{it.get('pos_neg_counts')}\n",
              f"- 权重统计：{it.get('weight_stats')}\n",
              f"- (user,item) 重复对数：{it.get('dup_user_item_pairs')}\n",
              f"- 图：{it.get('fig_weight_hist')}\n\n"]

    sp = sections.get("splits", {})
    lines += ["## 4. 时序切分\n",
              f"- split 分布：{sp.get('split_counts')}\n",
              f"- 每用户最后一条正样本在 test 的占比：{sp.get('last_pos_in_test_pct')}\n",
              f"- 每用户倒数第二条正样本在 val 的占比：{sp.get('second_last_in_val_pct')}\n",
              f"- 图：{sp.get('fig_splits_pie')}\n\n"]

    ev = sections.get("eval", {})
    lines += ["## 5. 评测样本\n",
              f"- 覆盖率（与 test 正样本一一对应）：{ev.get('coverage_pct')}\n",
              f"- 每个样本负例数：min={ev.get('neg_count_min')}, max={ev.get('neg_count_max')}\n",
              f"- 违规计数（负例=正例 / 负例在已看中）：{(ev.get('violations_pos_eq'), ev.get('violations_neg_in_seen'))}\n",
              f"- 图：{ev.get('fig_neg_pop_hist')}\n\n"]

    REPORT.write_text("".join(lines), encoding="utf-8")

# ---------- main ----------
if __name__ == "__main__":
    movies  = _read(IN_MOVIES)
    reviews = _read(IN_REVIEWS)
    itemq   = _read(OUT_ITEMQ)
    inter   = _read(OUT_INTER)
    splits  = _read(OUT_SPLITS) if OUT_SPLITS.exists() else inter
    evaldf  = _read(OUT_EVAL)

    secs = {}
    secs["cons"]  = check_consistency(movies, reviews, itemq)
    secs["itemq"] = validate_item_quality(itemq)
    secs["inter"] = validate_interactions(inter)
    secs["splits"] = validate_splits(inter)
    secs["eval"]   = validate_eval(inter, evaldf)

    write_report(secs)
    print("[OK] validation done ->", REPORT)
