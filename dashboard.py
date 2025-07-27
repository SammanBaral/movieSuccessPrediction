import streamlit as st
from movie_success_pipeline import load_model, predict_movie_success
import pandas as pd
import json
import numpy as np
import re

st.title("🎬 Movie Success Predictor (Social Media AI)")
st.markdown("""
Enter a movie name to scan social media posts (from Sentiment140 sample) and predict its overall box office success based on public sentiment, hashtags, and engagement.
""")

model_bundle = load_model()

# Load the preprocessed dataset
with open('sentiment140_for_pipeline.json', 'r', encoding='utf-8') as f:
    all_posts = json.load(f)

def filter_posts_by_movie(movie_name, posts):
    pattern = re.compile(re.escape(movie_name), re.IGNORECASE)
    return [p for p in posts if pattern.search(p['text'])]

def aggregate_and_predict(posts, model_bundle):
    if not posts:
        return None, None, None, None
    preds, confs, explanations = [], [], []
    for p in posts:
        pred, conf, explanation = predict_movie_success(
            p['text'], p['hashtags'], p['likes'], p['shares'], p['comments'], model_bundle
        )
        preds.append(pred)
        confs.append(conf)
        explanations.append(explanation)
    # Majority vote for prediction
    pred_series = pd.Series(preds)
    final_pred = pred_series.value_counts().idxmax()
    avg_conf = np.mean(confs)
    # Aggregate feature importances
    feat_imp = {}
    for explanation in explanations:
        for feat, imp in explanation:
            feat_imp[feat] = feat_imp.get(feat, 0) + imp
    for feat in feat_imp:
        feat_imp[feat] /= len(explanations)
    top_features = sorted(feat_imp.items(), key=lambda x: -x[1])[:5]
    return final_pred, avg_conf, top_features, pred_series.value_counts().to_dict()

movie_name = st.text_input("Enter movie name to scan posts:")

if st.button("Predict Movie Success for Movie") and movie_name.strip():
    posts = filter_posts_by_movie(movie_name, all_posts)
    st.markdown(f"Found **{len(posts)}** posts mentioning '{movie_name}'.")
    if not posts:
        st.warning("No posts found for this movie in the sample.")
    else:
        pred, conf, top_features, pred_dist = aggregate_and_predict(posts, model_bundle)
        st.markdown(f"### Prediction: **{pred}**")
        st.markdown(f"**Confidence (avg):** {conf:.2f}")
        st.markdown("#### Prediction Distribution:")
        st.write(pred_dist)
        st.markdown("#### Top Contributing Features (avg):")
        feat_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
        st.bar_chart(feat_df.set_index("Feature"))
        st.markdown("---")
        st.markdown(
            f"*Model: RandomForest, Features: text, hashtags, engagement, sentiment, emotion, TF-IDF.\nData: Sentiment140 (10k tweets, mapped to Hit/Average/Flop). Posts filtered by movie name.*"
        )

with st.expander("How does this work?"):
    st.write("""
    - The app scans all social media posts in the dataset for mentions of the movie name you enter.
    - It predicts the success of each post, then aggregates the results for an overall movie prediction.
    - Feature importances and prediction distribution are shown for transparency.
    """)
