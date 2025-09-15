'''# -*- coding: utf-8 -*-
"""
Django 管理命令，用于训练混合推荐模型并生成推荐结果。

通过将模型训练和推荐生成过程封装在此命令中，可以方便地通过
`python manage.py generate_recommendations` 来调用，并易于被定时任务（如 cron）调度。
'''

import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix
import implicit

from django.core.management.base import BaseCommand
from django.conf import settings

# --- 核心推荐逻辑封装 --- #

class HybridRecModel:
    """
    一个封装了数据加载、模型训练和推荐生成的混合推荐模型。
    经过重构，现在可以从文件或直接从内存中的DataFrame进行训练。
    """
    # 重构：构造函数不再接收文件路径，使类更通用
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.user_map, self.item_map, self.all_users, self.all_items = [None] * 4
        self.movies_df, self.als_model, self.als_user_factors, self.als_item_factors = [None] * 4
        self.content_sim_matrix, self.movie_content_map, self.user_pos_history = [None] * 3
        self.item_user_matrix = None
        self.stdout = None
        self.stderr = None

    # 重构：这是一个辅助方法，专门用于从CSV文件加载数据
    def _read_data_from_files(self, data_path, truth_path):
        """从CSV文件加载数据"""
        self.stdout.write("[INFO] Reading data from CSV files...")
        try:
            reviews_df = pd.read_csv(truth_path / "splits.csv")
            movies_df = pd.read_csv(data_path / "movies.csv")
            return reviews_df, movies_df
        except FileNotFoundError as e:
            self.stderr.write(f"[ERROR] Data file not found: {e}. Exiting.")
            raise

    # 重构：这是核心的数据预处理逻辑，现在接收DataFrame作为输入
    def _preprocess_data(self, reviews_df, movies_df):
        """使用DataFrame预处理数据"""
        self.stdout.write("[STEP 1/4] Preprocessing data...")
        
        # 适配来自数据库的数据：将'rating'列映射为'weight'列，如果它们存在的话
        if 'rating' in reviews_df.columns and 'weight' not in reviews_df.columns:
            reviews_df['weight'] = reviews_df['rating']
        if 'y' not in reviews_df.columns:
            reviews_df['y'] = 1

        splits_df = reviews_df
        self.movies_df = movies_df

        splits_df['user_id'] = splits_df['user_id'].astype(str)
        splits_df['item_id'] = splits_df['item_id'].astype(str)
        self.movies_df['imdb_id'] = self.movies_df['imdb_id'].astype(str)

        self.all_users = sorted(splits_df['user_id'].unique())
        self.all_items = sorted(splits_df['item_id'].unique())
        self.user_map = {uid: i for i, uid in enumerate(self.all_users)}
        self.item_map = {iid: i for i, iid in enumerate(self.all_items)}
        
        self.movies_df = self.movies_df[self.movies_df['imdb_id'].isin(self.all_items)].reset_index(drop=True)
        self.movie_content_map = {mid: i for i, mid in enumerate(self.movies_df['imdb_id'])}
        
        self.train_df_als = splits_df[splits_df['y'] == 1].copy()
        self.user_pos_history = self.train_df_als.groupby('user_id')['item_id'].apply(list).to_dict()

    def _train_content_model(self):
        self.stdout.write("[STEP 2/4] Building Content-Based model...")
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'])
        self.content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def _train_als_model(self):
        self.stdout.write("[STEP 3/4] Training ALS model...")
        self.train_df_als['user_idx'] = self.train_df_als['user_id'].map(self.user_map)
        self.train_df_als['item_idx'] = self.train_df_als['item_id'].map(self.item_map)
        
        self.item_user_matrix = coo_matrix(
            (self.train_df_als['weight'] + 1, (self.train_df_als['item_idx'], self.train_df_als['user_idx'])),
            shape=(len(self.all_items), len(self.all_users))
        ).tocsr()
        
        self.als_model = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.01, iterations=20)
        self.als_model.fit(self.item_user_matrix)
        
        self.als_user_factors = self.als_model.item_factors
        self.als_item_factors = self.als_model.user_factors

    # 重构：新的训练入口，供离线命令使用，从文件读取数据
    def train_from_files(self, data_path, truth_path, stdout, stderr):
        """从文件训练模型的完整流程"""
        self.stdout = stdout
        self.stderr = stderr
        reviews_df, movies_df = self._read_data_from_files(data_path, truth_path)
        self._preprocess_data(reviews_df, movies_df)
        self._train_content_model()
        self._train_als_model()
        self.stdout.write(self.style.SUCCESS("[SUCCESS] All models trained successfully from files."))

    # 重构：新的训练入口，供在线API使用，从DataFrame读取数据
    def train_from_dataframes(self, reviews_df, movies_df, stdout, stderr):
        """从DataFrame训练模型的完整流程"""
        self.stdout = stdout
        self.stderr = stderr
        self._preprocess_data(reviews_df, movies_df)
        self._train_content_model()
        self._train_als_model()
        self.stdout.write(self.style.SUCCESS("[SUCCESS] All models trained successfully from DataFrames."))

    def _predict_scores(self, user_id, candidate_item_ids):
        if user_id not in self.user_map:
            return {}

        user_idx = self.user_map[user_id]
        user_vec = self.als_user_factors[user_idx]
        als_scores = {item_id: user_vec.dot(self.als_item_factors[self.item_map[item_id]]) if item_id in self.item_map else 0 for item_id in candidate_item_ids}

        history = self.user_pos_history.get(user_id, [])
        content_scores = {}
        if not history:
            content_scores = {item_id: 0 for item_id in candidate_item_ids}
        else:
            history_indices = [self.movie_content_map[item_id] for item_id in history if item_id in self.movie_content_map]
            for item_id in candidate_item_ids:
                if item_id in self.movie_content_map:
                    cand_idx = self.movie_content_map[item_id]
                    sims = self.content_sim_matrix[cand_idx, history_indices]
                    content_scores[item_id] = np.mean(sims) if sims.size > 0 else 0
                else:
                    content_scores[item_id] = 0

        als_scores_arr = np.array([als_scores[item_id] for item_id in candidate_item_ids]).reshape(-1, 1)
        content_scores_arr = np.array([content_scores[item_id] for item_id in candidate_item_ids]).reshape(-1, 1)

        norm_als = self.scaler.fit_transform(als_scores_arr).flatten()
        norm_content = self.scaler.fit_transform(content_scores_arr).flatten()

        final_scores = {item_id: (0.7 * norm_als[i]) + (0.3 * norm_content[i]) for i, item_id in enumerate(candidate_item_ids)}
        return final_scores

    def get_recommendations_for_user(self, user_id, top_n=50):
        """Generates recommendations for a single user."""
        if self.als_model is None:
            raise RuntimeError("Model is not trained yet. Please call .train() first.")
        
        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]
        user_items_for_filtering = self.item_user_matrix.T.tocsr()

        user_vec = self.als_user_factors[user_idx]
        scores = user_vec.dot(self.als_item_factors.T)
        
        liked_item_indices = user_items_for_filtering[user_idx].indices
        scores[liked_item_indices] = -np.inf
        
        candidate_indices = np.argsort(scores)[::-1][:200]
        candidate_ids = [self.all_items[i] for i in candidate_indices]
        
        final_scores = self._predict_scores(user_id, candidate_ids)
        
        sorted_items = sorted(final_scores.keys(), key=lambda item: final_scores[item], reverse=True)
        return sorted_items[:top_n]

    def generate_all_user_recommendations(self, top_n=50):
        self.stdout.write(f"[STEP 4/4] Generating Top-{top_n} recommendations for all users...")
        if self.als_model is None:
            self.stderr.write("[ERROR] Model is not trained yet. Please call .train() first.")
            return {}

        all_recommendations = {}
        user_items_for_filtering = self.item_user_matrix.T.tocsr()
        
        for user_id in tqdm(self.all_users, desc="Generating Recommendations"):
            user_idx = self.user_map[user_id]
            
            user_vec = self.als_user_factors[user_idx]
            scores = user_vec.dot(self.als_item_factors.T)
            
            liked_item_indices = user_items_for_filtering[user_idx].indices
            scores[liked_item_indices] = -np.inf
            
            candidate_indices = np.argsort(scores)[::-1][:200]
            candidate_ids = [self.all_items[i] for i in candidate_indices]
            
            final_scores = self._predict_scores(user_id, candidate_ids)
            
            sorted_items = sorted(final_scores.keys(), key=lambda item: final_scores[item], reverse=True)
            all_recommendations[user_id] = sorted_items[:top_n]

        return all_recommendations

# --- Django 管理命令 --- #

class Command(BaseCommand):
    help = 'Trains the hybrid recommendation model and generates recommendations for all users.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting recommendation generation process..."))
        
        project_root = settings.BASE_DIR.parent
        raw_data_path = project_root / 'data'
        processed_data_path = project_root / 'truth_value_out'
        output_path = project_root / 'all_user_recommendations.json'

        model = HybridRecModel()
        model.train_from_files(
            data_path=raw_data_path, 
            truth_path=processed_data_path, 
            stdout=self.stdout, 
            stderr=self.stderr
        )

        all_user_recs = model.generate_all_user_recommendations(top_n=50)

        self.stdout.write(f"[DEMO] Saving recommendations to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_user_recs, f, ensure_ascii=False, indent=4)

        self.stdout.write(self.style.SUCCESS("Successfully generated and saved all recommendations."))
''
