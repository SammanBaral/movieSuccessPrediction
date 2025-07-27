# Movie Success Prediction from Social Media

This project predicts movie box office success (Hit, Average, Flop) using social media posts, hashtags, and engagement stats.

## Features
- Text preprocessing and feature engineering
- Sentiment and emotion analysis
- Hashtag and engagement feature extraction
- RandomForest-based classifier
- Streamlit dashboard for interactive predictions

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data:**
   - Edit or expand `your_data.json` with your own samples.

3. **Train the model:**
   ```bash
   python movie_success_pipeline.py
   ```
   This will train the model and save it as `movie_success_model.pkl`.

4. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
   Enter a post, hashtags, and engagement stats to get predictions and feature explanations.

## Files
- `movie_success_pipeline.py`: ML pipeline and feature engineering
- `dashboard.py`: Streamlit dashboard
- `your_data.json`: Sample data (edit or replace with your own)
- `requirements.txt`: Python dependencies

## Customization
- To use real Twitter data, add a data collection script and preprocess as needed.
- You can swap out the classifier or add more features for research.

---
For questions or thesis support, contact your project advisor or the AI assistant. 