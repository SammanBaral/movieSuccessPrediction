import streamlit as st
from movie_success_pipeline import load_model, predict_movie_success
import pandas as pd
import json
import numpy as np
import re
# Import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available, using basic charts")
from fetch_twitter_data import fetch_tweets, generate_synthetic_movie_data
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
}
.hit-box { background-color: #D4EDDA; border: 2px solid #28A745; }
.average-box { background-color: #FFF3CD; border: 2px solid #FFC107; }
.flop-box { background-color: #F8D7DA; border: 2px solid #DC3545; }
.metric-card {
    background-color: #F8F9FA;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007BFF;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎬 Pre-Release Movie Success Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">Predict box office success using pre-release social media sentiment and engagement</p>
    <p style="color: #888;">🎯 For Producers • 📊 For Marketers • 📈 For Analysts</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model_bundle = load_model()
    st.success("✅ ML Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.header("🎛️ Configuration")
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Generate Synthetic Data", "Use Existing Dataset"],
    help="Choose how to collect movie-related social media data"
)

num_samples = st.sidebar.slider(
    "Number of Social Media Posts",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="More samples = more accurate prediction"
)

# Try to load existing dataset as fallback
try:
    with open('sentiment140_for_pipeline.json', 'r', encoding='utf-8') as f:
        existing_posts = json.load(f)
except:
    existing_posts = []

def collect_movie_data(movie_name, data_source, num_samples):
    """Collect movie-related social media data"""
    if data_source == "Generate Synthetic Data":
        # Generate pre-release focused synthetic data
        posts = generate_synthetic_movie_data(movie_name, num_samples)
        return posts, "synthetic"
    else:
        # Try to find existing posts about the movie
        filtered_posts = filter_posts_by_movie(movie_name, existing_posts)
        if len(filtered_posts) < 10:
            # Fallback to synthetic if not enough existing data
            posts = generate_synthetic_movie_data(movie_name, num_samples)
            return posts, "synthetic_fallback"
        return filtered_posts[:num_samples], "existing"

def filter_posts_by_movie(movie_name, posts):
    pattern = re.compile(re.escape(movie_name), re.IGNORECASE)
    return [p for p in posts if pattern.search(p['text'])]

def analyze_emotions_detailed(posts):
    """Detailed emotion analysis for dashboard charts"""
    emotion_data = []
    sentiment_data = []
    
    for post in posts:
        # Get emotion analysis
        from movie_success_pipeline import emotion_features, sentiment_features
        
        dominant_emotion, confidence, emotions = emotion_features(post['text'])
        vader_compound, tb_polarity, tb_subjectivity = sentiment_features(post['text'])
        
        emotion_data.append({
            'text': post['text'][:50] + '...',
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'joy': emotions['joy'],
            'excitement': emotions['excitement'],
            'sadness': emotions['sadness'],
            'fear': emotions['fear'],
            'anger': emotions['anger'],
            'likes': post['likes'],
            'shares': post['shares'],
            'comments': post['comments']
        })
        
        sentiment_data.append({
            'text': post['text'][:50] + '...',
            'vader_compound': vader_compound,
            'tb_polarity': tb_polarity,
            'tb_subjectivity': tb_subjectivity,
            'engagement': post['likes'] + post['shares'] + post['comments']
        })
    
    return pd.DataFrame(emotion_data), pd.DataFrame(sentiment_data)

def aggregate_and_predict(posts, model_bundle):
    if not posts:
        return None, None, None, None, None, None
    
    preds, confs, explanations = [], [], []
    detailed_predictions = []
    
    for i, p in enumerate(posts):
        pred, conf, explanation = predict_movie_success(
            p['text'], p['hashtags'], p['likes'], p['shares'], p['comments'], model_bundle
        )
        preds.append(pred)
        confs.append(conf)
        explanations.append(explanation)
        
        detailed_predictions.append({
            'post_id': i,
            'text': p['text'][:100] + '...',
            'prediction': pred,
            'confidence': conf,
            'likes': p['likes'],
            'shares': p['shares'],
            'comments': p['comments'],
            'hashtags': len(p['hashtags'])
        })
    
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
    top_features = sorted(feat_imp.items(), key=lambda x: -x[1])[:8]
    
    pred_distribution = pred_series.value_counts().to_dict()
    
    return final_pred, avg_conf, top_features, pred_distribution, detailed_predictions, pd.DataFrame(detailed_predictions)

def create_prediction_charts(emotion_df, sentiment_df, pred_df, top_features, pred_distribution):
    """Create comprehensive charts for the dashboard"""
    
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Emotion Distribution', 'Sentiment Analysis', 'Prediction Distribution',
                'Feature Importance', 'Engagement vs Sentiment', 'Post Predictions'
            ],
            specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Emotion Distribution (Pie Chart)
        emotion_counts = emotion_df['dominant_emotion'].value_counts()
        fig.add_trace(
            go.Pie(labels=emotion_counts.index, values=emotion_counts.values, name="Emotions"),
            row=1, col=1
        )
        
        # 2. Sentiment Analysis (Scatter)
        fig.add_trace(
            go.Scatter(
                x=sentiment_df['tb_polarity'],
                y=sentiment_df['tb_subjectivity'],
                mode='markers',
                marker=dict(
                    size=sentiment_df['engagement']/10,
                    color=sentiment_df['vader_compound'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                text=sentiment_df['text'],
                name="Sentiment"
            ),
            row=1, col=2
        )
        
        # 3. Prediction Distribution (Bar)
        pred_labels = list(pred_distribution.keys())
        pred_values = list(pred_distribution.values())
        colors = ['#28A745' if x=='Hit' else '#FFC107' if x=='Average' else '#DC3545' for x in pred_labels]
        
        fig.add_trace(
            go.Bar(x=pred_labels, y=pred_values, marker_color=colors, name="Predictions"),
            row=1, col=3
        )
        
        # 4. Feature Importance (Bar)
        feat_names = [f[0] for f in top_features]
        feat_values = [f[1] for f in top_features]
        
        fig.add_trace(
            go.Bar(x=feat_values, y=feat_names, orientation='h', name="Features"),
            row=2, col=1
        )
        
        # 5. Engagement vs Sentiment (Scatter)
        fig.add_trace(
            go.Scatter(
                x=sentiment_df['vader_compound'],
                y=sentiment_df['engagement'],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                text=sentiment_df['text'],
                name="Engagement"
            ),
            row=2, col=2
        )
        
        # 6. Individual Post Predictions (Bar)
        pred_colors = ['#28A745' if x=='Hit' else '#FFC107' if x=='Average' else '#DC3545' for x in pred_df['prediction']]
        
        fig.add_trace(
            go.Bar(
                x=pred_df['post_id'][:20],  # Show first 20 posts
                y=pred_df['confidence'][:20],
                marker_color=pred_colors[:20],
                text=pred_df['prediction'][:20],
                name="Post Predictions"
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="📊 Comprehensive Movie Success Analysis",
            title_x=0.5
        )
        
        return fig
    except Exception as e:
        print(f"Error creating plotly charts: {e}")
        return None

def create_basic_charts(emotion_df, sentiment_df, pred_df, top_features, pred_distribution):
    """Create basic charts using streamlit when plotly is not available"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎭 Emotion Distribution")
        emotion_counts = emotion_df['dominant_emotion'].value_counts()
        st.bar_chart(emotion_counts)
    
    with col2:
        st.subheader("📊 Prediction Distribution")
        pred_chart_df = pd.DataFrame(list(pred_distribution.items()), columns=['Prediction', 'Count'])
        st.bar_chart(pred_chart_df.set_index('Prediction'))
    
    with col3:
        st.subheader("🔧 Top Features")
        feat_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
        st.bar_chart(feat_df.set_index("Feature"))
    
    # Additional charts in second row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💭 Sentiment vs Engagement")
        sentiment_chart = pd.DataFrame({
            'Sentiment': sentiment_df['vader_compound'],
            'Engagement': sentiment_df['engagement']
        })
        st.scatter_chart(sentiment_chart.set_index('Sentiment'))
    
    with col2:
        st.subheader("🎯 Individual Predictions")
        pred_confidence = pred_df[['post_id', 'confidence']].head(20)
        st.line_chart(pred_confidence.set_index('post_id'))

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    movie_name = st.text_input(
        "🎬 Enter Movie Name (Pre-Release)",
        placeholder="e.g., Oppenheimer, Barbie, Avatar 3...",
        help="Enter the name of an upcoming or recently released movie"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    predict_button = st.button(
        "🔮 Predict Box Office Success",
        type="primary",
        use_container_width=True
    )

if predict_button and movie_name.strip():
    with st.spinner(f"🔍 Analyzing pre-release social media buzz for '{movie_name}'..."):
        # Collect movie data
        posts, data_type = collect_movie_data(movie_name, data_source, num_samples)
        
        if not posts:
            st.error("❌ No data could be collected for this movie.")
            st.stop()
        
        # Show data collection info
        data_info_map = {
            "synthetic": "🤖 Generated synthetic pre-release social media data",
            "synthetic_fallback": "🤖 Generated synthetic data (insufficient existing data)",
            "existing": "📊 Using existing social media data"
        }
        
        st.info(f"{data_info_map[data_type]} - {len(posts)} posts analyzed")
        
        # Perform prediction and analysis
        final_pred, avg_conf, top_features, pred_dist, detailed_preds, pred_df = aggregate_and_predict(posts, model_bundle)
        emotion_df, sentiment_df = analyze_emotions_detailed(posts)
        
        # Display main prediction result
        pred_class_map = {
            "Hit": ("hit-box", "🎯 HIT", "High box office success expected!"),
            "Average": ("average-box", "📊 AVERAGE", "Moderate box office performance expected"),
            "Flop": ("flop-box", "📉 FLOP", "Low box office performance expected")
        }
        
        box_class, pred_display, pred_desc = pred_class_map[final_pred]
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h2>{pred_display}</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">{pred_desc}</p>
            <p style="font-size: 1rem; opacity: 0.8;">Confidence: {avg_conf:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Prediction Confidence",
                f"{avg_conf:.1%}",
                delta=f"{avg_conf-0.5:.1%}" if avg_conf > 0.5 else f"{avg_conf-0.5:.1%}"
            )
        
        with col2:
            dominant_emotion = emotion_df['dominant_emotion'].mode()[0]
            emotion_pct = (emotion_df['dominant_emotion'] == dominant_emotion).mean()
            st.metric(
                "😊 Dominant Emotion",
                dominant_emotion.title(),
                f"{emotion_pct:.1%} of posts"
            )
        
        with col3:
            avg_sentiment = sentiment_df['vader_compound'].mean()
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric(
                "💭 Overall Sentiment",
                sentiment_label,
                f"{avg_sentiment:.2f}"
            )
        
        with col4:
            total_engagement = emotion_df[['likes', 'shares', 'comments']].sum().sum()
            avg_engagement = total_engagement / len(posts)
            st.metric(
                "🔥 Avg Engagement",
                f"{avg_engagement:.0f}",
                "per post"
            )
        
        # Comprehensive charts
        st.markdown("---")
        
        if PLOTLY_AVAILABLE:
            try:
                charts_fig = create_prediction_charts(emotion_df, sentiment_df, pred_df, top_features, pred_dist)
                if charts_fig:
                    st.plotly_chart(charts_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Advanced charts unavailable, showing basic charts")
                    create_basic_charts(emotion_df, sentiment_df, pred_df, top_features, pred_dist)
            except Exception as e:
                st.error(f"Error creating advanced charts: {e}")
                create_basic_charts(emotion_df, sentiment_df, pred_df, top_features, pred_dist)
        else:
            st.info("📊 Using basic charts (install plotly for advanced visualizations)")
            create_basic_charts(emotion_df, sentiment_df, pred_df, top_features, pred_dist)
        
        # Detailed analysis tabs
        tab1, tab2, tab3 = st.tabs(["📝 Sample Posts", "📊 Detailed Data", "🔍 Model Insights"])
        
        with tab1:
            st.subheader("Sample Social Media Posts")
            sample_posts = posts[:10]
            for i, post in enumerate(sample_posts):
                with st.expander(f"Post {i+1}: {post['text'][:60]}..."):
                    st.write(f"**Full Text:** {post['text']}")
                    st.write(f"**Hashtags:** {', '.join(post['hashtags'])}")
                    st.write(f"**Engagement:** {post['likes']} likes, {post['shares']} shares, {post['comments']} comments")
                    if 'label' in post:
                        st.write(f"**Label:** {post['label']}")
        
        with tab2:
            st.subheader("Emotion Analysis Data")
            st.dataframe(emotion_df.head(20))
            
            st.subheader("Sentiment Analysis Data")
            st.dataframe(sentiment_df.head(20))
        
        with tab3:
            st.subheader("Model Feature Importance")
            feat_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
            st.dataframe(feat_df)
            
            st.subheader("Individual Post Predictions")
            st.dataframe(pred_df.head(20))
            
            st.markdown("""
            **Model Information:**
            - Algorithm: RandomForest Classifier
            - Features: Text (TF-IDF), Sentiment (VADER, TextBlob), Emotions, Hashtags, Engagement
            - Training Data: Social media posts with Hit/Average/Flop labels
            - Prediction Method: Majority vote across all posts
            """)

elif movie_name.strip() and not predict_button:
    st.info("👆 Click 'Predict Box Office Success' to analyze the movie")

# Information section
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    ### 🎯 Pre-Release Movie Success Prediction
    
    This system analyzes social media buzz **before** a movie's release to predict its box office performance:
    
    **📊 Data Collection:**
    - Collects social media posts mentioning the movie
    - Focuses on pre-release sentiment and engagement
    - Generates synthetic data when real data isn't available
    
    **🧠 AI Analysis:**
    - **Sentiment Analysis**: VADER and TextBlob for emotional tone
    - **Emotion Detection**: Joy, excitement, sadness, fear, anger
    - **Engagement Metrics**: Likes, shares, comments analysis
    - **Text Features**: TF-IDF vectorization of post content
    
    **🎬 Prediction:**
    - Classifies each post as indicating Hit/Average/Flop potential
    - Aggregates predictions using majority voting
    - Provides confidence scores and feature explanations
    
    **📈 Use Cases:**
    - **Producers**: Early performance indicators for investment decisions
    - **Marketers**: Adjust marketing strategies based on public sentiment
    - **Analysts**: Data-driven box office forecasting
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎬 Movie Success Predictor | Built with Streamlit & Machine Learning</p>
    <p>💡 Helping entertainment industry professionals make data-driven decisions</p>
</div>
""", unsafe_allow_html=True)
