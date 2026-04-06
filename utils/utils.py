import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def load_stopwords(filepath="Data/files/vietnamese-stopwords.txt"):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f.readlines()])
        return stopwords
    return set()
STOPWORDS = load_stopwords()


def clean_search_query(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    
    return " ".join(filtered_words)


def load_data_for_recommender():
    df = pd.read_csv("Data/cleaned_data_recommend.csv")
    path_sim = "models/nha_cosine_sim.pkl" 
    if os.path.exists(path_sim):
        cosine_sim = joblib.load(path_sim)
    else:
        cosine_sim = np.eye(len(df))
    return df, cosine_sim


def processing_cluster_logic(model, df):

    df_temp = df.copy()
    
    # feature
    df_temp['price_m2'] = (df_temp['gia_ban'] / df_temp['dien_tich_dat']) / 1e6
    df_temp['log_price_m2'] = np.log1p(df_temp['price_m2'])
    df_temp['log_dien_tich_dat'] = np.log1p(df_temp['dien_tich_dat'])
    
    # predict
    features = model.feature_names_in_ 
    df_temp['cluster'] = model.predict(df_temp[features])
    
    # mapping
    sorted_clusters = df_temp.groupby('cluster')['price_m2'].mean().sort_values().index.tolist()
    labels = ["Diện tích to",  "Diện tích nhỏ"]
    mapping = {cluster_id: labels[i] if i < len(labels) else f"Nhóm {cluster_id}" 
            for i, cluster_id in enumerate(sorted_clusters)}
    
    df_temp['Phân khúc'] = df_temp['cluster'].map(mapping)
    cluster_stats = df_temp.groupby('cluster')['price_m2'].mean().to_dict()
    
    return mapping, cluster_stats, df_temp

# ======= Caching Data =======
@st.cache_data(max_entries=1, show_spinner=False)
def get_cached_data_recommender():
    df, cosine_sim = load_data_for_recommender()
    return df, cosine_sim

@st.cache_data(max_entries=1, show_spinner=False)
def get_cached_data_clustering():
    """Load dữ liệu và Model cho Clustering"""
    df_cluster = pd.read_csv("Data/well_formed_data.csv")
    model = joblib.load("models/kmeans.pkl")
    
    # Thực hiện logic mapping ngay tại bước load để trả về kết quả cuối cùng
    cluster_mapping, cluster_stats, df_viz = processing_cluster_logic(model, df_cluster)
    
    return df_cluster, model, cluster_mapping, cluster_stats, df_viz

@st.cache_data(max_entries=1, show_spinner=False)
def get_house_options(df_houses_input):
    """Tạo danh sách label cho selectbox một lần duy nhất và lưu vào cache"""
    return (df_houses_input['id'].astype(str) + " - Quận " + df_houses_input['quan'] + 
            " - Phường " + df_houses_input['phuong_val'] + " - " + 
            df_houses_input['gia_ban_ty'].astype(str) + " Tỷ - " + 
            df_houses_input['mo_ta_clean'].str[:80] + "...").tolist()
# ===========================


def update_weights(changed_key, suffix):        # một thanh thay đổi, hai thanh còn lại điều chỉnh tỉ lệ nghịch
    keys = [f'w_c{suffix}', f'w_p{suffix}', f'w_l{suffix}']
    other_keys = [k for k in keys if k != changed_key]
    
    new_val = st.session_state[changed_key]
    remaining = 1.0 - new_val
    
    current_others_sum = sum(st.session_state[k] for k in other_keys)
    
    if current_others_sum > 0:
        for k in other_keys:
            st.session_state[k] = (st.session_state[k] / current_others_sum) * remaining
    else:
        # Nếu 2 thanh kia đang là 0, chia đều phần còn lại
        for k in other_keys:
            st.session_state[k] = remaining / len(other_keys)\
                

def get_location_onehot_matrix(df):
    location_dummies = pd.get_dummies(df['quan'])
    return location_dummies.values, location_dummies.columns.tolist()


def calculate_price_similarity(target_price, house_price):
    price_diff = abs(target_price - house_price)
    return np.exp(-0.3 * price_diff)


def get_hybrid_recommendations(target_idx, df, cosine_sim, w_content=0.5, w_price=0.25, w_loc=0.25, top_n=5):
    loc_matrix, _ = get_location_onehot_matrix(df)
    
    target_house = df.iloc[target_idx]
    target_loc_vector = loc_matrix[target_idx].reshape(1, -1)
    
    loc_similarities = cosine_similarity(target_loc_vector, loc_matrix)[0]

    scores = []
    for idx, row in df.iterrows():
        if idx == target_idx: continue
            
        sim_content = cosine_sim[target_idx][idx]
        sim_price = calculate_price_similarity(target_house['gia_ban_ty'], row['gia_ban_ty'])
        sim_loc = loc_similarities[idx]
        
        hybrid_score = (w_content * sim_content + w_price * sim_price + w_loc * sim_loc)
        scores.append((idx, hybrid_score))
        
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in scores[:top_n]]
    
    result_df = df.iloc[top_indices].copy()
    result_df['hybrid_score'] = [x[1] for x in scores[:top_n]]
    
    return result_df


def get_hybrid_recommendations_from_search(target_quan, target_price, search_text, df, w_content=0.5, w_price=0.25, w_loc=0.25, top_n=6):
    # location
    location_dummies = pd.get_dummies(df['quan'])
    all_quans = location_dummies.columns.tolist()
    query_loc_vector = np.zeros(len(all_quans))
    if target_quan in all_quans:
        query_loc_vector[all_quans.index(target_quan)] = 1
    # Tính cosine similarity cho vị trí
    loc_similarities = cosine_similarity(query_loc_vector.reshape(1, -1), location_dummies.values)[0]
    
    # TF-IDF
    cleaned_query = clean_search_query(search_text)
    if cleaned_query:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['mo_ta_clean'].fillna(''))
        query_vector = tfidf.transform([cleaned_query])
        content_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    else:
        content_similarities = np.ones(len(df))
    
    # hybrid Score
    scores = []
    for idx, row in df.iterrows():
        sim_content = content_similarities[idx]
        sim_loc = loc_similarities[idx]
        sim_price = calculate_price_similarity(target_price, row['gia_ban_ty'])      
        hybrid_score = (w_content * sim_content + w_price * sim_price + w_loc * sim_loc)
        scores.append((idx, hybrid_score))
        
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in scores[:top_n]]
    
    result_df = df.iloc[top_indices].copy()
    result_df['hybrid_score'] = [x[1] for x in scores[:top_n]]
    
    return result_df


def get_model_metrics(model, X):
    inertia = model.inertia_
    labels = model.predict(X)
    sil_score = silhouette_score(X, labels)
    return inertia, sil_score