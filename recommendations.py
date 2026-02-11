import logging
import random
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

logger = logging.getLogger(__name__)

# Module-level data (initialized by load_product_data)
df = None
products_df = None
cosine_sim = None
product_indices = None


@st.cache_resource
def load_product_data(path: str):

    global df, products_df, cosine_sim, product_indices
    df = pd.read_csv(path)
    df['pure_names'] = df['pure_names'].astype(str).fillna('')
    df['cleaned_body'] = df['cleaned_body'].astype(str).fillna('')
    df['tags'] = df['tags'].astype(str).fillna('')
    df['product_id'] = df['product_id'].astype(str).str.strip().str.replace('.0$', '', regex=True)
    df['customer_id'] = df['customer_id'].astype(str).str.strip().str.replace('.0$', '', regex=True)
    if 'price' not in df.columns:
        raise ValueError("Missing 'price' column in your_cleaned_product_data.csv")

    products_df = df[['product_id', 'pure_names', 'true_vendor', 'price', 'cleaned_body', 'tags']].drop_duplicates(subset='product_id').reset_index(drop=True)

    tfidf_name = TfidfVectorizer(stop_words='english', max_features=500).fit_transform(products_df['pure_names'])
    tfidf_body = TfidfVectorizer(stop_words='english', max_features=1000).fit_transform(products_df['cleaned_body'])
    tfidf_tags = TfidfVectorizer(stop_words='english').fit_transform(products_df['tags'])

    combined_tfidf = hstack([tfidf_name * 0.5, tfidf_tags * 0.2, tfidf_body * 0.3])
    cosine_sim = cosine_similarity(combined_tfidf, combined_tfidf)
    product_indices = pd.Series(products_df.index, index=products_df['product_id'].astype(str)).drop_duplicates()
    logger.info("Recommendation system setup completed successfully")
    return df, products_df, cosine_sim, product_indices


def content_based_recommendations(product_id, top_n=5):
    product_id = str(product_id).strip().replace('.0', '')
    if product_id not in product_indices:
        logger.warning("Product ID %s not found in product_indices", product_id)
        st.warning(f"Product ID {product_id} not found in product_indices")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'similarity'])

    idx = product_indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices_similar = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]

    recs = products_df.iloc[product_indices_similar][['product_id', 'pure_names', 'true_vendor']].copy()
    recs['similarity'] = similarity_values
    price_info = df[['product_id', 'price']].copy()
    price_info['product_id'] = price_info['product_id'].astype(str).str.strip()
    recs['product_id'] = recs['product_id'].astype(str).str.strip()
    price_info['price'] = pd.to_numeric(price_info['price'], errors='coerce')
    price_max = price_info.groupby('product_id', as_index=False)['price'].max()

    recs = recs.merge(price_max, on='product_id', how='left')
    recs = recs.drop_duplicates(subset='product_id').reset_index(drop=True)
    logger.info("Generated %d content-based recommendations for product %s", len(recs), product_id)
    return recs[['product_id', 'pure_names', 'true_vendor', 'price', 'similarity']]


@st.cache_resource
def load_ml_models():
    """Load XGBoost model and encoders with caching."""
    logger = logging.getLogger(__name__)
    try:
        model = joblib.load('xgb_model.pkl')
        user_encoder = joblib.load('user_encoder.pkl')
        product_encoder = joblib.load('product_encoder.pkl')
        logger.info("Loaded ML models and encoders successfully")
        return model, user_encoder, product_encoder
    except FileNotFoundError:
        logger.error("Missing model or encoder files")
        return None, None, None


def collaborative_recommendations(user_id, top_n=5):
    user_id = str(user_id).strip().replace('.0', '')
    logger.debug("collaborative_recommendations called for user_id: %s", user_id)
    try:
        int(float(user_id))
    except ValueError:
        logger.warning("Invalid user ID format: %s", user_id)
        st.warning(f"Invalid user ID format: {user_id}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    try:
        model, user_encoder, product_encoder = load_ml_models()
        if model is None:
            st.error("❌ Missing model or encoder files (xgb_model.pkl, user_encoder.pkl, product_encoder.pkl).")
            return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])
    except Exception as e:
        logger.error("Error loading models: %s", e)
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    if user_id not in user_encoder.classes_:
        logger.warning("User ID %s not found in user_encoder.classes_", user_id)
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    user_enc = user_encoder.transform([user_id])[0]
    product_ids = products_df['product_id'].astype(str).str.strip().tolist()
    valid_product_ids = [pid for pid in product_ids if pid in product_encoder.classes_]
    if not valid_product_ids:
        logger.warning("No valid product IDs found for user %s", user_id)
        st.warning(f"No valid product IDs found for user {user_id}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    logger.info("Found %d valid product IDs for user %s", len(valid_product_ids), user_id)
    st.info(f"Found {len(valid_product_ids)} valid product IDs for user {user_id}")
    product_encs = product_encoder.transform(valid_product_ids)
    price_lookup = products_df.set_index('product_id')['price'].to_dict()

    candidate_data = pd.DataFrame({
        'user_encoded': [user_enc] * len(valid_product_ids),
        'product_encoded': product_encs,
        'price': [price_lookup.get(pid, 0) for pid in valid_product_ids]
    })

    try:
        preds = model.predict_proba(candidate_data)[:, 1]
        logger.info("Predictions generated for user %s", user_id)
    except Exception as e:
        logger.error("Error predicting with XGBoost model for user %s: %s", user_id, e)
        st.warning(f"Error predicting with XGBoost model for user {user_id}: {e}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    candidate_data['product_id'] = valid_product_ids
    candidate_data['score'] = preds
    top_recs = candidate_data.sort_values('score', ascending=False).head(top_n)

    if top_recs.empty:
        logger.warning("No top recommendations generated for user %s", user_id)
        st.warning(f"No top recommendations generated for user {user_id}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    top_recs['product_id'] = top_recs['product_id'].astype(str).str.strip()
    products_df['product_id'] = products_df['product_id'].astype(str).str.strip()
    unmatched_ids = [pid for pid in top_recs['product_id'] if pid not in products_df['product_id'].tolist()]
    if unmatched_ids:
        logger.warning("Unmatched product IDs for user %s: %s", user_id, unmatched_ids)
        st.warning(f"Unmatched product IDs for user {user_id}: {unmatched_ids}")

    merged = top_recs.merge(products_df[['product_id', 'pure_names', 'true_vendor', 'price']], on='product_id', how='left')
    if merged.empty or 'price' not in merged.columns:
        logger.warning("Merge failed for user %s. Top_recs product_ids: %s", user_id, top_recs['product_id'].tolist())
        st.warning(f"Merge failed for user {user_id}. Top_recs product_ids: {top_recs['product_id'].tolist()}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'score'])

    merged['price'] = pd.to_numeric(merged['price'], errors='coerce').fillna(0)
    merged = merged[['product_id', 'pure_names', 'true_vendor', 'price', 'score']]
    logger.info("Generated %d collaborative recommendations for user %s", len(merged), user_id)
    st.info(f"Generated {len(merged)} collaborative recommendations for user {user_id}")
    return merged


def diversify_recommendations(product_id, top_n=5, pool_n=20, similarity_threshold=0.60):
    logger.debug("diversify_recommendations called for product_id: %s", product_id)
    product_id = str(product_id).strip().replace('.0', '')
    if product_id not in product_indices:
        logger.warning("Product ID %s not found in product_indices", product_id)
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'similarity'])

    idx = product_indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:pool_n+1]
    product_indices_similar = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]
    similar_vectors = cosine_sim[product_indices_similar, :][:, product_indices_similar]

    groups = []
    assigned = set()
    for i in range(len(product_indices_similar)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i+1, len(product_indices_similar)):
            if j not in assigned and similar_vectors[i][j] > similarity_threshold:
                group.append(j)
                assigned.add(j)
        groups.append(group)

    selected_indices = []
    for group in groups:
        chosen_idx = random.choice(group)
        selected_indices.append(product_indices_similar[chosen_idx])
        if len(selected_indices) >= top_n:
            break

    recs = products_df.iloc[selected_indices][['product_id', 'pure_names', 'true_vendor']].copy()
    recs['similarity'] = [similarity_values[product_indices_similar.index(idx)] for idx in selected_indices]
    recs = recs.merge(df[['product_id', 'price']].sort_values("price", ascending=False).drop_duplicates("product_id"), on='product_id', how='left')
    recs['price'] = pd.to_numeric(recs['price'], errors='coerce').fillna(0)
    logger.info("Generated %d diversified recommendations for product %s", len(recs), product_id)
    return recs.reset_index(drop=True)[['product_id', 'pure_names', 'true_vendor', 'price', 'similarity']]


def diversify_top_recommendations(recs_df, top_n=5, similarity_threshold=0.60):
    logger.debug("diversify_top_recommendations called with %d recommendations", len(recs_df))
    if recs_df.empty:
        logger.warning("Empty recommendations DataFrame in diversify_top_recommendations")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'similarity'])

    product_ids = recs_df['product_id'].astype(str).str.strip().tolist()
    indices = [product_indices[pid] for pid in product_ids if pid in product_indices]
    selected = []
    selected_similarities = []
    for i in range(len(indices)):
        if len(selected) >= top_n:
            break
        current_idx = indices[i]
        too_similar = False
        for sel_idx in selected:
            sim = cosine_sim[current_idx, sel_idx]
            if sim > similarity_threshold:
                too_similar = True
                break
        if not too_similar:
            selected.append(current_idx)
            selected_similarities.append(recs_df.iloc[i]['similarity'])

    final_recs = products_df.iloc[selected][['product_id', 'pure_names', 'true_vendor']].copy()
    final_recs['product_id'] = final_recs['product_id'].astype(str).str.strip()
    final_recs['similarity'] = selected_similarities
    price_df = df[['product_id', 'price']].drop_duplicates(subset='product_id').copy()
    price_df['product_id'] = price_df['product_id'].astype(str).str.strip()

    final_recs = final_recs.merge(price_df, on='product_id', how='left')
    final_recs['price'] = pd.to_numeric(final_recs['price'], errors='coerce').fillna(0)
    logger.info("Diversified to %d recommendations", len(final_recs))
    return final_recs.reset_index(drop=True)[['product_id', 'pure_names', 'true_vendor', 'price', 'similarity']]


def multi_seed_content_recommendations(user_id, per_seed_recs=10, similarity_threshold=0.85, final_top_n=5):
    logger.debug("multi_seed_content_recommendations called for user_id: %s", user_id)
    user_id = str(user_id).strip().replace('.0', '')
    try:
        int(float(user_id))
    except ValueError:
        logger.warning("Invalid user ID format in content-based: %s", user_id)
        st.warning(f"Invalid user ID format in content-based: {user_id}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'similarity'])

    user_history = df[df['customer_id'] == user_id]
    past_product_ids = user_history['product_id'].astype(str).str.strip().unique().tolist()
    past_indices = [product_indices[pid] for pid in past_product_ids if pid in product_indices]

    if not past_indices:
        logger.warning("No purchase history found for user %s", user_id)
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'similarity'])

    similarity_matrix = cosine_sim[np.ix_(past_indices, past_indices)]
    groups = []
    assigned = set()
    for i in range(len(past_indices)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i+1, len(past_indices)):
            if j not in assigned and similarity_matrix[i][j] > similarity_threshold:
                group.append(j)
                assigned.add(j)
        groups.append(group)

    chosen_indices = [random.choice(group) for group in groups]
    seed_product_ids = [past_product_ids[i] for i in chosen_indices]
    all_recs = pd.concat([content_based_recommendations(pid, per_seed_recs) for pid in seed_product_ids], ignore_index=True)
    all_recs = all_recs.drop_duplicates(subset='product_id')

    if all_recs.empty or 'similarity' not in all_recs.columns:
        logger.warning("No content-based recommendations generated for user %s", user_id)
        st.warning(f"No content-based recommendations generated for user {user_id}")
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'similarity'])

    final_recs = diversify_recommendations(seed_product_ids[0] if seed_product_ids else None, top_n=final_top_n) if all_recs.empty else diversify_top_recommendations(all_recs, top_n=final_top_n)
    logger.info("Generated %d content-based recommendations for user %s", len(final_recs), user_id)
    return final_recs[['product_id', 'pure_names', 'true_vendor', 'price', 'similarity']]


def hybrid_recommendations(user_id, top_n_each=5):
    logger.debug("hybrid_recommendations called for user_id: %s", user_id)
    user_recs = collaborative_recommendations(user_id, top_n_each)
    user_recs = user_recs.copy()
    user_recs["source"] = "collaborative"

    content_recs = multi_seed_content_recommendations(user_id, per_seed_recs=top_n_each, final_top_n=top_n_each)
    content_recs = content_recs.copy()
    content_recs["source"] = "content-based"

    if user_recs.empty and content_recs.empty:
        logger.warning("No recommendations (collaborative or content-based) for user %s", user_id)
        return pd.DataFrame(columns=['product_id', 'pure_names', 'true_vendor', 'price', 'source'])

    hybrid = pd.concat([user_recs, content_recs], ignore_index=True)
    hybrid = hybrid.drop_duplicates(subset="product_id").reset_index(drop=True)
    logger.info("Generated %d hybrid recommendations for user %s", len(hybrid), user_id)
    return hybrid[['product_id', 'pure_names', 'true_vendor', 'price', 'source']]
