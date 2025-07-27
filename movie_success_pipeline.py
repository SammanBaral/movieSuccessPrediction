import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import xgboost as xgb
import json
import sys
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(data_file):
    if data_file.endswith('.json'):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        raise ValueError('Unsupported file format')
    return df

# --- 1. Text Preprocessing ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9# ]+", "", text)
    text = text.lower()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --- 2. Hashtag Features ---
def hashtag_features(hashtags):
    hashtags = [h.lower() for h in hashtags]
    freq = len(hashtags)
    diversity = len(set(hashtags)) / (len(hashtags) + 1e-5)
    return freq, diversity

# --- 3. Sentiment & Emotion Analysis ---
def sentiment_features(text):
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(text)
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity
    return vader['compound'], tb_polarity, tb_subjectivity

def emotion_features(text):
    emotions = {'joy': 0, 'excitement': 0, 'sadness': 0, 'fear': 0, 'anger': 0}
    joy_words = ['love', 'amazing', 'great', 'happy', 'joy', 'fun']
    excitement_words = ['excited', 'mustwatch', 'epic', 'thrilling', 'wow']
    sadness_words = ['sad', 'cry', 'tears', 'heartbreaking']
    fear_words = ['scary', 'fear', 'terrifying', 'horror']
    anger_words = ['angry', 'hate', 'worst', 'bad']
    text = text.lower()
    for w in joy_words:
        if w in text: emotions['joy'] += 1
    for w in excitement_words:
        if w in text: emotions['excitement'] += 1
    for w in sadness_words:
        if w in text: emotions['sadness'] += 1
    for w in fear_words:
        if w in text: emotions['fear'] += 1
    for w in anger_words:
        if w in text: emotions['anger'] += 1
    total = sum(emotions.values()) + 1e-5
    for k in emotions:
        emotions[k] /= total
    dominant = max(emotions, key=emotions.get)
    confidence = emotions[dominant]
    return dominant, confidence, emotions

# --- 4. Feature Engineering Pipeline ---
def build_features(df):
    # Preprocess text
    df['clean_text'] = df['text'].apply(preprocess_text)
    # Hashtag features
    df['hashtag_freq'], df['hashtag_diversity'] = zip(*df['hashtags'].apply(hashtag_features))
    # Sentiment
    sent_feats = df['text'].apply(sentiment_features)
    df['vader_compound'] = [x[0] for x in sent_feats]
    df['tb_polarity'] = [x[1] for x in sent_feats]
    df['tb_subjectivity'] = [x[2] for x in sent_feats]
    # Emotion
    emo_feats = df['text'].apply(emotion_features)
    df['dominant_emotion'] = [x[0] for x in emo_feats]
    df['emotion_confidence'] = [x[1] for x in emo_feats]
    # Engagement
    for col in ['likes', 'shares', 'comments']:
        if col not in df:
            df[col] = 0
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    # Encode dominant emotion
    df = pd.get_dummies(df, columns=['dominant_emotion'])
    return df, tfidf

# --- 5. Model Training ---
def train_model(df, label_col='label'):
    feature_cols = [c for c in df.columns if c not in ['text', 'hashtags', 'label', 'clean_text']]
    X = df[feature_cols]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))
    return model, X_test, y_test, feature_cols

# --- 6. Save Model ---
def save_model(model, tfidf, feature_cols, path='movie_success_model.pkl'):
    joblib.dump({'model': model, 'tfidf': tfidf, 'feature_cols': feature_cols}, path)

# --- 7. Load Model ---
def load_model(path='movie_success_model.pkl'):
    return joblib.load(path)

# --- 8. Explain Prediction ---
def explain_prediction(model, X, feature_cols, top_n=5):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_cols[i], importances[i]) for i in indices]

# --- 9. Predict Function ---
def predict_movie_success(text, hashtags, likes=0, shares=0, comments=0, model_bundle=None):
    tfidf = model_bundle['tfidf']
    model = model_bundle['model']
    feature_cols = model_bundle['feature_cols']
    # Build features for single sample
    clean = preprocess_text(text)
    freq, diversity = hashtag_features(hashtags)
    vader, tb_pol, tb_subj = sentiment_features(text)
    dom_emo, emo_conf, emotions = emotion_features(text)
    tfidf_vec = tfidf.transform([clean]).toarray()[0]
    # Build feature vector
    data = {
        'hashtag_freq': freq,
        'hashtag_diversity': diversity,
        'vader_compound': vader,
        'tb_polarity': tb_pol,
        'tb_subjectivity': tb_subj,
        'emotion_confidence': emo_conf,
        'likes': likes,
        'shares': shares,
        'comments': comments,
        **{f'tfidf_{i}': tfidf_vec[i] for i in range(len(tfidf_vec))},
        **{f'dominant_emotion_{e}': 1 if dom_emo == e else 0 for e in ['joy', 'excitement', 'sadness', 'fear', 'anger']}
    }
    # Fill missing features
    for f in feature_cols:
        if f not in data:
            data[f] = 0
    X = pd.DataFrame([data])[feature_cols]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf = np.max(proba)
    explanation = explain_prediction(model, X, feature_cols)
    return pred, conf, explanation

# --- 10. Example Usage ---
if __name__ == "__main__":
    # Allow user to specify data file
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'sentiment140_for_pipeline.json'
    print(f"Loading data from {data_file}")
    df = load_data(data_file)
    df, tfidf = build_features(df)
    model, X_test, y_test, feature_cols = train_model(df)
    save_model(model, tfidf, feature_cols)
    # Example prediction
    model_bundle = load_model()
    pred, conf, explanation = predict_movie_success(
        "This movie is a must-watch! Absolutely loved it. #Epic #MustWatch",
        ["#Epic", "#MustWatch"],
        likes=1200, shares=300, comments=150,
        model_bundle=model_bundle
    )
    print("Prediction:", pred, "Confidence:", conf)
    print("Explanation (top features):", explanation)
