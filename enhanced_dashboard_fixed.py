"""
Enhanced Streamlit Dashboard with Advanced Visualizations
for Movie Success Prediction Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Set page config
st.set_page_config(
    page_title="Movie Success Prediction Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_pre_release_data():
    """Load the pre-release movie dataset"""
    try:
        with open('pre_release_movie_dataset.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error("Pre-release dataset not found. Please run the data generator first.")
        return pd.DataFrame()

def create_overview_metrics(df):
    """Create overview metrics cards"""
    if df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", len(df['movie_name'].unique()))
    
    with col2:
        st.metric("Total Pre-release Data Points", len(df))
    
    with col3:
        hit_rate = (df['label'] == 'Hit').mean() * 100
        st.metric("Hit Rate", f"{hit_rate:.1f}%")
    
    with col4:
        # Calculate engagement rate from likes, shares, comments
        df_calc = df.copy()
        df_calc['engagement_rate'] = (df_calc['likes'] + df_calc['shares'] + df_calc['comments']) / 1000
        avg_engagement = df_calc['engagement_rate'].mean()
        st.metric("Avg Engagement Rate", f"{avg_engagement:.3f}")

def create_success_distribution_viz(df):
    """Create success category distribution visualization"""
    if df.empty:
        return
    
    st.subheader("📊 Movie Success Distribution")
    
    # Success distribution
    success_counts = df['label'].value_counts()
    
    fig = px.pie(
        values=success_counts.values,
        names=success_counts.index,
        title="Distribution of Movie Success Categories",
        color_discrete_map={'Hit': '#2E8B57', 'Average': '#FF8C00', 'Flop': '#DC143C'}
    )
    
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def create_pre_release_timeline_viz(df):
    """Create pre-release timeline analysis"""
    if df.empty:
        return
    
    st.subheader("📈 Pre-Release Timeline Analysis")
    
    # Calculate engagement rate and post count
    df_calc = df.copy()
    df_calc['engagement_rate'] = (df_calc['likes'] + df_calc['shares'] + df_calc['comments']) / 1000
    df_calc['post_count'] = 1  # Each row represents one post
    
    # Create timeline data
    timeline_data = df_calc.groupby(['days_before_release', 'label']).agg({
        'post_count': 'sum',
        'engagement_rate': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Post count timeline
        fig_timeline = px.line(
            timeline_data, 
            x='days_before_release', 
            y='post_count',
            color='label',
            title='Pre-Release Buzz Timeline',
            labels={'days_before_release': 'Days Before Release', 'post_count': 'Number of Posts'}
        )
        fig_timeline.update_xaxes(autorange="reversed")  # Recent dates on right
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        # Engagement rate timeline
        fig_engagement = px.line(
            timeline_data, 
            x='days_before_release', 
            y='engagement_rate',
            color='label',
            title='Pre-Release Engagement Timeline',
            labels={'days_before_release': 'Days Before Release', 'engagement_rate': 'Engagement Rate'}
        )
        fig_engagement.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_engagement, use_container_width=True)

def create_movie_comparison_viz(df):
    """Create movie comparison visualization"""
    if df.empty:
        return
    
    st.subheader("🎬 Movie Comparison Analysis")
    
    # Movie selection
    movies = df['movie_name'].unique()
    selected_movies = st.multiselect(
        "Select movies to compare:",
        movies,
        default=movies[:3] if len(movies) >= 3 else movies
    )
    
    if len(selected_movies) < 2:
        st.warning("Please select at least 2 movies for comparison.")
        return
    
    # Filter data for selected movies
    filtered_df = df[df['movie_name'].isin(selected_movies)]
    
    # Calculate engagement rate and create comparison metrics
    filtered_df = filtered_df.copy()
    filtered_df['engagement_rate'] = (filtered_df['likes'] + filtered_df['shares'] + filtered_df['comments']) / 1000
    filtered_df['post_count'] = 1  # Each row represents one post
    
    movie_metrics = filtered_df.groupby('movie_name').agg({
        'post_count': 'sum',
        'engagement_rate': 'mean',
        'label': 'first'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Total buzz comparison
        fig_buzz = px.bar(
            movie_metrics,
            x='movie_name',
            y='post_count',
            color='label',
            title='Total Pre-Release Buzz by Movie',
            color_discrete_map={'Hit': '#2E8B57', 'Average': '#FF8C00', 'Flop': '#DC143C'}
        )
        fig_buzz.update_xaxes(tickangle=45)
        st.plotly_chart(fig_buzz, use_container_width=True)
    
    with col2:
        # Engagement comparison
        fig_engagement = px.bar(
            movie_metrics,
            x='movie_name',
            y='engagement_rate',
            color='label',
            title='Average Engagement Rate by Movie',
            color_discrete_map={'Hit': '#2E8B57', 'Average': '#FF8C00', 'Flop': '#DC143C'}
        )
        fig_engagement.update_xaxes(tickangle=45)
        st.plotly_chart(fig_engagement, use_container_width=True)

def create_buzz_type_analysis(df):
    """Create buzz type analysis"""
    if df.empty:
        return
    
    st.subheader("💬 Buzz Type Analysis")
    
    # Buzz type distribution
    buzz_dist = df.groupby(['buzz_type', 'label']).size().reset_index(name='count')
    
    fig = px.bar(
        buzz_dist,
        x='buzz_type',
        y='count',
        color='label',
        title='Buzz Type Distribution by Success Category',
        color_discrete_map={'Hit': '#2E8B57', 'Average': '#FF8C00', 'Flop': '#DC143C'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_prediction_interface(df):
    """Create prediction interface"""
    st.subheader("🔮 Movie Success Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            movie_name = st.text_input("Movie Title", "New Movie")
            genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"])
            budget = st.number_input("Budget (in millions)", min_value=1, max_value=500, value=50)
        
        with col2:
            social_buzz = st.slider("Social Media Buzz Score", 0.0, 1.0, 0.5)
            engagement_rate = st.slider("Engagement Rate", 0.0, 1.0, 0.3)
            days_before = st.number_input("Days Before Release", min_value=1, max_value=365, value=30)
        
        submitted = st.form_submit_button("Predict Success")
        
        if submitted:
            # Create prediction (demo)
            if PIPELINE_AVAILABLE:
                prediction, confidence, features = predict_movie_success(
                    movie_name, genre, budget, social_buzz, engagement_rate
                )
            else:
                prediction = "Hit" if social_buzz > 0.6 and engagement_rate > 0.4 else "Average"
                confidence = 0.75
                features = [("social_buzz", social_buzz), ("engagement", engagement_rate)]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Predicted Success: **{prediction}**")
                st.info(f"Confidence: {confidence:.1%}")
            
            with col2:
                if features:
                    st.write("**Key Factors:**")
                    for feature, importance in features[:5]:
                        st.write(f"- {feature}: {importance:.3f}")

def create_data_insights(df):
    """Create data insights section"""
    if df.empty:
        return
    
    st.subheader("💡 Data Insights")
    
    insights = []
    
    # Calculate engagement rate
    df_calc = df.copy()
    df_calc['engagement_rate'] = (df_calc['likes'] + df_calc['shares'] + df_calc['comments']) / 1000
    df_calc['post_count'] = 1  # Each row represents one post
    
    # Calculate insights
    hit_movies = df_calc[df_calc['label'] == 'Hit']
    flop_movies = df_calc[df_calc['label'] == 'Flop']
    
    if not hit_movies.empty and not flop_movies.empty:
        hit_avg_engagement = hit_movies['engagement_rate'].mean()
        flop_avg_engagement = flop_movies['engagement_rate'].mean()
        
        if flop_avg_engagement > 0:
            insights.append(f"📈 Hit movies have {hit_avg_engagement/flop_avg_engagement:.1f}x higher engagement rates than flops")
        
        hit_avg_posts = hit_movies.groupby('movie_name')['post_count'].sum().mean()
        flop_avg_posts = flop_movies.groupby('movie_name')['post_count'].sum().mean()
        
        if flop_avg_posts > 0:
            insights.append(f"📱 Hit movies generate {hit_avg_posts/flop_avg_posts:.1f}x more social media posts")
        
        # Peak buzz timing
        peak_engagement = df_calc.groupby('days_before_release')['engagement_rate'].mean()
        if not peak_engagement.empty:
            peak_day = peak_engagement.idxmax()
            insights.append(f"⏰ Peak engagement typically occurs {peak_day} days before release")
        
        # Genre analysis
        genre_success = df_calc.groupby('genre')['label'].apply(lambda x: (x == 'Hit').mean()).sort_values(ascending=False)
        if not genre_success.empty:
            best_genre = genre_success.index[0]
            success_rate = genre_success.iloc[0]
            insights.append(f"🎭 {best_genre} movies have the highest success rate at {success_rate:.1%}")
    
    # Display insights
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.write("Not enough data for insights generation.")

def main():
    """Main dashboard function"""
    st.title("🎬 Movie Success Prediction Dashboard")
    st.markdown("### Advanced Analytics for Pre-Release Movie Success Prediction")
    
    # Load data
    df = load_pre_release_data()
    
    if df.empty:
        st.error("No data available. Please ensure the pre-release dataset is generated.")
        return
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Success category filter
    success_categories = st.sidebar.multiselect(
        "Success Categories",
        df['label'].unique(),
        default=df['label'].unique()
    )
    
    # Time range filter
    max_days = df['days_before_release'].max()
    min_days = df['days_before_release'].min()
    
    time_range = st.sidebar.slider(
        "Days Before Release Range",
        min_value=int(min_days),
        max_value=int(max_days),
        value=(int(min_days), int(max_days))
    )
    
    # Genre filter
    if 'genre' in df.columns:
        genres = st.sidebar.multiselect(
            "Genres",
            df['genre'].unique(),
            default=df['genre'].unique()
        )
        df = df[df['genre'].isin(genres)]
    
    # Apply filters
    df_filtered = df[
        (df['label'].isin(success_categories)) &
        (df['days_before_release'] >= time_range[0]) &
        (df['days_before_release'] <= time_range[1])
    ]
    
    # Overview metrics
    create_overview_metrics(df_filtered)
    
    # Main visualizations
    create_success_distribution_viz(df_filtered)
    create_pre_release_timeline_viz(df_filtered)
    create_movie_comparison_viz(df_filtered)
    create_buzz_type_analysis(df_filtered)
    
    # Prediction interface
    create_prediction_interface(df_filtered)
    
    # Data insights
    create_data_insights(df_filtered)
    
    # Footer
    st.markdown("---")
    st.markdown("**Enhanced Movie Success Prediction Dashboard** | Built with Streamlit & Plotly")

if __name__ == "__main__":
    main()
