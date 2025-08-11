import streamlit as st
from movie_success_pipeline import load_model, predict_movie_success

st.title("🎬 Movie Success Predictor (Social Media AI)")

model_bundle = load_model()

text = st.text_area("Enter social media post/caption:")
hashtags = st.text_input("Hashtags (comma-separated):")
likes = st.number_input("Likes", min_value=0, value=0)
shares = st.number_input("Shares", min_value=0, value=0)
comments = st.number_input("Comments", min_value=0, value=0)

if st.button("Predict Movie Success"):
    hashtags_list = [h.strip() for h in hashtags.split(",") if h.strip()]
    pred, conf, explanation = predict_movie_success(
        text, hashtags_list, likes, shares, comments, model_bundle
    )
    st.markdown(f"### Prediction: **{pred}**")
    st.markdown(f"**Confidence:** {conf:.2f}")
    st.markdown("#### Top Contributing Features:")
    for feat, imp in explanation:
        st.write(f"- {feat}: {imp:.3f}")
