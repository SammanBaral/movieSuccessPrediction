"""
Enhanced Streamlit Dashboard with Advanced Visualizations
for Movie Success Prediction Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots impor            fig_timeline = px.line(
                timeline_data, 
                x='days_before_release', 
                y='post_count',
                color='label',
                title='Pre-Release Buzz Timeline',
                labels={'days_before_release': 'Days Before Release', 'post_count': 'Number of Posts'}
            )
            fig_timeline.update_xaxes(autorange="reversed")  # Recent dates on right
            st.plotly_chart(fig_timeline, use_container_width=True)plots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta
import json
import re
from collections import Counter

# Import your existing pipeline with error handling
try:
    from movie_success_pipeline import load_model, predict_movie_success, load_data, build_features
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    st.error(f"Original pipeline not available: {e}")
    # Create dummy functions for demo
    def load_model():
        return None
    def predict_movie_success(*args, **kwargs):
        return "Demo", 0.75, [("feature1", 0.3), ("feature2", 0.2)]

# Configure page
st.set_page_config(
    page_title="🎬 Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.stMetric {
    background-color: #f0f2f6;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 10px;
}
.prediction-box {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_historical_data():
    """Load and cache historical data for analysis"""
    try:
        # Try to load the pre-release dataset first
        df = pd.read_json('pre_release_movie_dataset.json')
        st.success("✅ Loaded pre-release movie dataset!")
    except:
        try:
            # Fallback to your existing data
            df = pd.read_json('your_data.json')
            st.info("📁 Loaded existing dataset (your_data.json)")
        except:
            # Create sample data if nothing exists
            df = pd.DataFrame({
                'text': ['Sample movie review'] * 10,
                'hashtags': [['#Movie']] * 10,
                'likes': np.random.randint(0, 1000, 10),
                'shares': np.random.randint(0, 100, 10),
                'comments': np.random.randint(0, 50, 10),
                'label': np.random.choice(['Hit', 'Average', 'Flop'], 10)
            })
            st.warning("⚠️ Using sample data - no dataset files found")
    return df

def create_sentiment_analysis_viz(df):
    """Create sentiment analysis visualizations"""
    
    # Add sentiment scores if not present
    if 'sentiment_score' not in df.columns:
        from textblob import TextBlob
        df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment_category'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        fig_sentiment = px.histogram(
            df, x='sentiment_category', color='sentiment_category',
            title='Sentiment Distribution',
            color_discrete_map={
                'Positive': '#2E8B57',
                'Neutral': '#FFD700', 
                'Negative': '#DC143C'
            }
        )
        fig_sentiment.update_layout(showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Sentiment vs Success correlation
        if 'label' in df.columns:
            fig_correlation = px.box(
                df, x='label', y='sentiment_score', color='label',
                title='Sentiment Score by Movie Success Category'
            )
            st.plotly_chart(fig_correlation, use_container_width=True)

def create_engagement_analysis(df):
    """Create engagement metrics analysis"""
    
    # Calculate engagement metrics
    df['total_engagement'] = df['likes'] + df['shares'] + df['comments']
    df['engagement_rate'] = df['total_engagement'] / (df['likes'] + 1)  # Avoid division by zero
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Engagement metrics by category
        if 'label' in df.columns:
            engagement_by_category = df.groupby('label')[['likes', 'shares', 'comments']].mean()
            
            fig_engagement = go.Figure()
            for metric in ['likes', 'shares', 'comments']:
                fig_engagement.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=engagement_by_category.index,
                    y=engagement_by_category[metric]
                ))
            
            fig_engagement.update_layout(
                title='Average Engagement by Success Category',
                barmode='group'
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
    
    with col2:
        # Engagement scatter plot
        fig_scatter = px.scatter(
            df, x='likes', y='comments', size='shares',
            color='label' if 'label' in df.columns else None,
            title='Engagement Patterns',
            hover_data=['total_engagement']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col3:
        # Engagement distribution
        fig_dist = px.histogram(
            df, x='total_engagement', nbins=20,
            title='Total Engagement Distribution'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

def create_text_analysis_viz(df):
    """Create text analysis visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word cloud
        st.subheader("🔤 Word Cloud")
        
        # Combine all text
        all_text = ' '.join(df['text'].astype(str))
        
        # Clean text for word cloud
        cleaned_text = re.sub(r'http\S+', '', all_text)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
        
        if cleaned_text.strip():
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis'
            ).generate(cleaned_text)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No valid text data for word cloud")
    
    with col2:
        # Text length analysis
        df['text_length'] = df['text'].astype(str).str.len()
        
        fig_length = px.histogram(
            df, x='text_length', nbins=20,
            title='Text Length Distribution'
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Hashtag analysis
        if 'hashtags' in df.columns:
            all_hashtags = []
            for hashtag_list in df['hashtags']:
                if isinstance(hashtag_list, list):
                    all_hashtags.extend(hashtag_list)
            
            if all_hashtags:
                hashtag_counts = Counter(all_hashtags)
                top_hashtags = dict(hashtag_counts.most_common(10))
                
                fig_hashtags = px.bar(
                    x=list(top_hashtags.keys()),
                    y=list(top_hashtags.values()),
                    title='Top 10 Hashtags'
                )
                fig_hashtags.update_layout(xaxis_title='Hashtags', yaxis_title='Frequency')
                st.plotly_chart(fig_hashtags, use_container_width=True)

def create_prediction_explanation_viz(explanation):
    """Create visualization for prediction explanations"""
    
    if not explanation:
        return
    
    features, importances = zip(*explanation)
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker_color='skyblue'
    ))
    
    fig.update_layout(
        title='Feature Importance for Prediction',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=400
    )
    
    return fig

def create_pre_release_timeline_viz(df):
    """Create pre-release timeline visualizations"""
    
    if 'days_before_release' in df.columns and 'buzz_type' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Buzz activity over time
            timeline_data = df.groupby(['days_before_release', 'label']).size().reset_index(name='post_count')
            
            fig_timeline = px.scatter(
                timeline_data, 
                x='days_before_release', 
                y='post_count',
                color='label',
                title='Pre-Release Buzz Timeline',
                labels={'days_before_release': 'Days Before Release', 'post_count': 'Number of Posts'}
            )
            fig_timeline.update_xaxis(autorange="reversed")  # Recent dates on right
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Buzz type distribution by success
            buzz_success = df.groupby(['buzz_type', 'label']).size().reset_index(name='count')
            
            fig_buzz = px.bar(
                buzz_success,
                x='buzz_type',
                y='count',
                color='label',
                title='Buzz Types by Movie Success',
                labels={'buzz_type': 'Type of Buzz', 'count': 'Number of Posts'}
            )
            fig_buzz.update_xaxis(tickangle=45)
            st.plotly_chart(fig_buzz, use_container_width=True)
        
        # Engagement patterns by timeline
        st.subheader("📈 Engagement Patterns Over Time")
        
        if len(df) > 0:
            # Create engagement timeline
            engagement_timeline = df.groupby(['days_before_release', 'label']).agg({
                'likes': 'mean',
                'shares': 'mean', 
                'comments': 'mean'
            }).reset_index()
            
            fig_engagement = px.line(
                engagement_timeline,
                x='days_before_release',
                y='likes',
                color='label',
                title='Average Likes Over Pre-Release Timeline'
            )
            fig_engagement.update_xaxis(autorange="reversed")
            st.plotly_chart(fig_engagement, use_container_width=True)

def create_movie_comparison_viz(df):
    """Create movie-by-movie comparison visualizations"""
    
    if 'movie_name' in df.columns:
        st.subheader("🎬 Movie-by-Movie Analysis")
        
        # Movie performance overview
        movie_stats = df.groupby(['movie_name', 'label']).agg({
            'likes': ['mean', 'sum'],
            'shares': ['mean', 'sum'],
            'comments': ['mean', 'sum'],
            'text': 'count'
        }).round(0)
        
        movie_stats.columns = ['avg_likes', 'total_likes', 'avg_shares', 'total_shares', 
                              'avg_comments', 'total_comments', 'post_count']
        movie_stats = movie_stats.reset_index()
        
        # Top movies by engagement
        col1, col2 = st.columns(2)
        
        with col1:
            top_movies = movie_stats.nlargest(10, 'total_likes')
            fig_top = px.bar(
                top_movies,
                x='total_likes',
                y='movie_name',
                color='label',
                title='Top Movies by Total Likes',
                orientation='h'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Movie success vs engagement
            fig_scatter = px.scatter(
                movie_stats,
                x='avg_likes',
                y='post_count',
                color='label',
                size='total_likes',
                hover_name='movie_name',
                title='Movie Success vs Pre-Release Buzz Volume'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def main():
    # Header
    st.title("🎬 Enhanced Movie Success Predictor")
    st.markdown("### Advanced Analytics & Real-time Prediction Dashboard")
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Analytics Dashboard", "🔮 Make Prediction", "📈 Model Performance", "💾 Data Management"]
    )
    
    # Load data
    df = load_historical_data()
    
    if page == "📊 Analytics Dashboard":
        st.header("📊 Data Analytics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Posts", len(df))
        with col2:
            avg_likes = df['likes'].mean() if 'likes' in df.columns else 0
            st.metric("Avg Likes", f"{avg_likes:.0f}")
        with col3:
            if 'label' in df.columns:
                hit_rate = (df['label'] == 'Hit').mean() * 100
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            else:
                st.metric("Hit Rate", "N/A")
        with col4:
            total_engagement = df[['likes', 'shares', 'comments']].sum().sum() if all(col in df.columns for col in ['likes', 'shares', 'comments']) else 0
            st.metric("Total Engagement", f"{total_engagement:,}")
        
        # Visualizations
        st.subheader("📈 Sentiment Analysis")
        create_sentiment_analysis_viz(df)
        
        st.subheader("👥 Engagement Analysis")
        create_engagement_analysis(df)
        
        st.subheader("📝 Text Analysis")
        create_text_analysis_viz(df)
        
        st.subheader("📅 Pre-Release Timeline Analysis")
        create_pre_release_timeline_viz(df)
        
        st.subheader("🎬 Movie Comparison Analysis")
        create_movie_comparison_viz(df)
    
    elif page == "🔮 Make Prediction":
        st.header("🔮 Movie Success Prediction")
        
        # Load model
        try:
            model_bundle = load_model()
            
            # Input form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    text = st.text_area(
                        "📝 Social Media Post/Review:",
                        placeholder="Enter a movie review or social media post...",
                        height=100
                    )
                    hashtags = st.text_input(
                        "🏷️ Hashtags (comma-separated):",
                        placeholder="#Movie, #MustWatch, #Epic"
                    )
                
                with col2:
                    likes = st.number_input("👍 Likes", min_value=0, value=100)
                    shares = st.number_input("🔄 Shares", min_value=0, value=20)
                    comments = st.number_input("💬 Comments", min_value=0, value=15)
                
                submitted = st.form_submit_button("🎯 Predict Movie Success")
            
            if submitted and text:
                # Process hashtags
                hashtags_list = [h.strip() for h in hashtags.split(",") if h.strip()]
                
                # Make prediction
                pred, conf, explanation = predict_movie_success(
                    text, hashtags_list, likes, shares, comments, model_bundle
                )
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎬 Prediction: {pred}</h2>
                    <h3>🎯 Confidence: {conf:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature importance visualization
                if explanation:
                    st.subheader("🔍 Prediction Explanation")
                    fig_explanation = create_prediction_explanation_viz(explanation)
                    st.plotly_chart(fig_explanation, use_container_width=True)
                    
                    # Detailed explanation
                    st.subheader("📋 Feature Details")
                    for i, (feature, importance) in enumerate(explanation):
                        st.write(f"**{i+1}. {feature}**: {importance:.4f}")
        
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.info("Please train the model first by running: `python movie_success_pipeline.py`")
    
    elif page == "📈 Model Performance":
        st.header("📈 Model Performance Analysis")
        
        try:
            # Load and analyze model performance
            if 'label' in df.columns:
                # Build features for analysis
                df_features, _ = build_features(df)
                
                # Label distribution
                label_counts = df['label'].value_counts()
                fig_labels = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Dataset Label Distribution"
                )
                st.plotly_chart(fig_labels, use_container_width=True)
                
                # Feature correlation heatmap
                st.subheader("🔥 Feature Correlation Heatmap")
                numeric_cols = df_features.select_dtypes(include=[np.number]).columns[:20]  # Limit for readability
                corr_matrix = df_features[numeric_cols].corr()
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No labeled data available for performance analysis")
                
        except Exception as e:
            st.error(f"Performance analysis failed: {e}")
    
    elif page == "💾 Data Management":
        st.header("💾 Data Management")
        
        # Data overview
        st.subheader("📋 Current Dataset Overview")
        st.dataframe(df.head())
        
        # Data collection interface
        st.subheader("🔄 Collect New Data")
        
        movie_name = st.text_input("🎬 Movie Name:", placeholder="Enter movie name to collect data")
        imdb_id = st.text_input("🎭 IMDb ID (optional):", placeholder="e.g., tt15398776")
        
        if st.button("🚀 Start Data Collection"):
            if movie_name:
                with st.spinner("Collecting data from multiple sources..."):
                    try:
                        from enhanced_data_collector import EnhancedMovieDataCollector
                        
                        collector = EnhancedMovieDataCollector()
                        new_data = collector.collect_all_data(movie_name, imdb_id if imdb_id else None)
                        
                        if new_data:
                            filename = f"real_data_{movie_name.lower().replace(' ', '_')}.json"
                            collector.save_data(new_data, filename)
                            
                            st.success(f"✅ Collected {len(new_data)} items and saved to {filename}")
                            
                            # Show sample of collected data
                            st.subheader("📊 Sample of Collected Data")
                            sample_df = pd.DataFrame([{
                                'text': item.text[:100] + '...',
                                'source': item.source,
                                'likes': item.likes,
                                'shares': item.shares,
                                'comments': item.comments
                            } for item in new_data[:5]])
                            st.dataframe(sample_df)
                        else:
                            st.warning("No data collected. Please check your API keys and network connection.")
                    except Exception as e:
                        st.error(f"Data collection failed: {e}")
            else:
                st.warning("Please enter a movie name")
        
        # API setup instructions
        with st.expander("🔧 API Setup Instructions"):
            st.markdown("""
            To collect real data, you need to set up API keys:
            
            **Reddit API (PRAW):**
            1. Go to https://www.reddit.com/prefs/apps/
            2. Create a new app
            3. Set environment variables:
               ```
               REDDIT_CLIENT_ID=your_client_id
               REDDIT_CLIENT_SECRET=your_client_secret
               ```
            
            **YouTube Data API:**
            1. Go to Google Cloud Console
            2. Enable YouTube Data API v3
            3. Create API key
            4. Set environment variable:
               ```
               YOUTUBE_API_KEY=your_api_key
               ```
            
            **News API:**
            1. Go to https://newsapi.org/
            2. Register for free API key
            3. Set environment variable:
               ```
               NEWS_API_KEY=your_api_key
               ```
            """)

if __name__ == "__main__":
    main()
