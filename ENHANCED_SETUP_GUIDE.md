# Enhanced Movie Success Prediction - Setup Guide

## 🚀 Quick Start

### 1. Install Enhanced Dependencies
```bash
pip install -r enhanced_requirements.txt
```

### 2. Set Up API Keys (for real data collection)

Create a `.env` file in your project root:
```env
# Reddit API (for Reddit data collection)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# YouTube Data API (for YouTube comments)
YOUTUBE_API_KEY=your_youtube_api_key

# News API (for news articles)
NEWS_API_KEY=your_news_api_key
```

### 3. API Setup Instructions

#### Reddit API Setup:
1. Go to https://www.reddit.com/prefs/apps/
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Note down your `client_id` and `client_secret`

#### YouTube Data API:
1. Go to Google Cloud Console (console.cloud.google.com)
2. Create a new project or select existing
3. Enable "YouTube Data API v3"
4. Create credentials (API Key)
5. Restrict the API key to YouTube Data API v3

#### News API:
1. Go to https://newsapi.org/
2. Register for a free account
3. Get your API key from the dashboard

### 4. Optional: Advanced NLP Models
```bash
# For emotion analysis
pip install transformers torch

# For advanced text processing
pip install spacy
python -m spacy download en_core_web_sm

# For better visualizations
pip install wordcloud plotly seaborn
```

## 🎯 Usage

### Enhanced Data Collection:
```bash
python enhanced_data_collector.py
```

### Train Enhanced Model:
```bash
python enhanced_ml_pipeline.py your_data.json
```

### Run Enhanced Dashboard:
```bash
streamlit run enhanced_dashboard.py
```

## 📊 New Features

### Real Data Sources:
- ✅ Reddit discussions and comments
- ✅ IMDb reviews and ratings  
- ✅ YouTube trailer comments
- ✅ News articles about movies

### Enhanced ML Features:
- 📝 Advanced text analysis (length, readability, punctuation)
- 😊 Deep learning emotion classification
- 🏷️ Sophisticated hashtag analysis
- 👥 Engagement pattern recognition
- ⏰ Temporal features (time-based patterns)
- 🔤 N-gram TF-IDF features

### Better Visualizations:
- 📈 Interactive sentiment analysis charts
- 👥 Engagement pattern visualizations
- 🔤 Word clouds and text analysis
- 📊 Model performance comparisons
- 🎯 Feature importance explanations
- 📅 Time series analysis

### Multiple ML Models:
- 🌳 Random Forest (ensemble)
- 🚀 Gradient Boosting
- ⚡ XGBoost
- 📊 Logistic Regression
- 🎯 Support Vector Machine

## 📁 Project Structure

```
thesis/
├── enhanced_data_collector.py     # Real data collection
├── enhanced_ml_pipeline.py        # Advanced ML pipeline
├── enhanced_dashboard.py          # Interactive dashboard
├── enhanced_requirements.txt      # Dependencies
├── .env                          # API keys (create this)
├── your_data.json               # Original sample data
├── real_data_*.json             # Collected real data
└── enhanced_movie_success_model.pkl  # Trained model
```

## 🔧 Troubleshooting

### Common Issues:

1. **SSL Certificate Errors:**
   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r enhanced_requirements.txt
   ```

2. **Reddit API Errors:**
   - Check your client_id and client_secret
   - Ensure you're using a "script" type app
   - Add rate limiting delays

3. **Transformer Model Errors:**
   ```bash
   pip install transformers[torch]
   # Or for CPU only:
   pip install transformers
   ```

4. **Spacy Model Missing:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🎓 Academic Benefits

### For Your Thesis:
- **Real-world data** from multiple sources
- **State-of-the-art NLP** techniques  
- **Comprehensive evaluation** with multiple models
- **Feature engineering** best practices
- **Explainable AI** with feature importance
- **Interactive visualizations** for presentations
- **Scalable architecture** for future extensions

### Research Opportunities:
- Compare synthetic vs. real data performance
- Analyze temporal patterns in movie buzz
- Study cross-platform sentiment differences
- Investigate feature importance across models
- Explore ensemble methods and model stacking

## 📈 Performance Improvements

### Expected Enhancements:
- **Better accuracy** with real data and advanced features
- **More robust predictions** with ensemble methods
- **Better interpretability** with feature explanations
- **Professional visualizations** for thesis presentation
- **Scalable data collection** for ongoing research

## 🔄 Next Steps

1. Set up API keys for real data collection
2. Run enhanced data collection for your target movies
3. Train the enhanced model with new features
4. Compare performance with your original model
5. Use the enhanced dashboard for thesis demonstrations
6. Consider publishing findings or extending for other domains

## 💡 Future Enhancements

- **Real-time prediction API** with Flask/FastAPI
- **Docker containerization** for deployment
- **Automated data pipeline** with scheduling
- **A/B testing framework** for model comparison
- **Integration with movie databases** (TMDb, OMDb)
- **Cross-validation with box office data**
- **Multi-language support** for international movies
