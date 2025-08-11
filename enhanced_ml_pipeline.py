"""
Enhanced Movie Success Pipeline with Advanced Feature Engineering
and Model Evaluation
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import xgboost as xgb
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class EnhancedFeatureEngineer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize advanced analyzers if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.emotion_classifier = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
            except:
                self.emotion_classifier = None
        else:
            self.emotion_classifier = None
            
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
        else:
            self.nlp = None
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags for cleaning (keep for separate analysis)
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep emoticons
        text = re.sub(r'[^A-Za-z0-9\s!?.:;,\'\"()]+', '', text)
        # Multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        text = self.clean_text(text)
        
        # Tokenization
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return " ".join(tokens)
    
    def extract_text_features(self, text):
        """Extract various text-based features"""
        if pd.isna(text):
            text = ""
        
        text = str(text)
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        
        # Readability (simple measure)
        words = text.split()
        if words:
            features['avg_sentence_length'] = len(words) / features['sentence_count'] if features['sentence_count'] > 0 else 0
        else:
            features['avg_sentence_length'] = 0
        
        return features
    
    def extract_hashtag_features(self, hashtags):
        """Enhanced hashtag analysis"""
        if pd.isna(hashtags) or not hashtags:
            hashtags = []
        
        if isinstance(hashtags, str):
            # Extract hashtags from text
            hashtags = re.findall(r'#\w+', hashtags)
        
        features = {}
        features['hashtag_count'] = len(hashtags)
        features['hashtag_diversity'] = len(set(hashtags)) / (len(hashtags) + 1)
        
        # Sentiment-based hashtag classification
        positive_indicators = ['love', 'amazing', 'best', 'awesome', 'great', 'fantastic', 'epic', 'brilliant']
        negative_indicators = ['hate', 'worst', 'terrible', 'awful', 'bad', 'boring', 'disappointing']
        
        hashtag_text = ' '.join(hashtags).lower()
        features['positive_hashtag_ratio'] = sum(indicator in hashtag_text for indicator in positive_indicators) / (len(hashtags) + 1)
        features['negative_hashtag_ratio'] = sum(indicator in hashtag_text for indicator in negative_indicators) / (len(hashtags) + 1)
        
        return features
    
    def extract_sentiment_features(self, text):
        """Enhanced sentiment analysis"""
        if pd.isna(text):
            text = ""
        
        text = str(text)
        features = {}
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        features.update({f'vader_{k}': v for k, v in vader_scores.items()})
        
        # TextBlob sentiment
        blob = TextBlob(text)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # Advanced emotion classification if available
        if self.emotion_classifier:
            try:
                emotions = self.emotion_classifier(text[:512])  # Limit text length
                for emotion_result in emotions[0]:  # First result
                    features[f'emotion_{emotion_result["label"].lower()}'] = emotion_result['score']
            except:
                pass
        
        return features
    
    def extract_engagement_features(self, likes, shares, comments):
        """Enhanced engagement feature engineering"""
        features = {}
        
        # Basic engagement metrics
        features['likes'] = likes
        features['shares'] = shares  
        features['comments'] = comments
        
        # Derived engagement metrics
        features['total_engagement'] = likes + shares + comments
        features['engagement_ratio'] = (shares + comments) / (likes + 1)  # Avoid division by zero
        features['comment_like_ratio'] = comments / (likes + 1)
        features['share_like_ratio'] = shares / (likes + 1)
        
        # Engagement intensity (log transformed to handle outliers)
        features['log_likes'] = np.log1p(likes)
        features['log_shares'] = np.log1p(shares)
        features['log_comments'] = np.log1p(comments)
        features['log_total_engagement'] = np.log1p(features['total_engagement'])
        
        # Engagement categories
        features['high_engagement'] = 1 if features['total_engagement'] > 100 else 0
        features['viral_potential'] = 1 if shares > likes * 0.1 else 0
        
        return features
    
    def extract_temporal_features(self, timestamp):
        """Extract temporal features if timestamp is available"""
        features = {}
        
        if pd.isna(timestamp):
            return features
        
        try:
            if isinstance(timestamp, str):
                dt = pd.to_datetime(timestamp)
            else:
                dt = timestamp
            
            features['hour'] = dt.hour
            features['day_of_week'] = dt.dayofweek
            features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
            features['month'] = dt.month
            
            # Time-based patterns
            features['is_prime_time'] = 1 if 18 <= dt.hour <= 22 else 0  # 6-10 PM
            features['is_morning'] = 1 if 6 <= dt.hour <= 12 else 0
            
        except:
            pass
        
        return features

class EnhancedMovieSuccessPredictor:
    def __init__(self):
        self.feature_engineer = EnhancedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        self.feature_columns = []
        
    def build_features(self, df):
        """Enhanced feature building pipeline"""
        print("🔧 Building enhanced features...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Text preprocessing
        df_features['clean_text'] = df_features['text'].apply(self.feature_engineer.preprocess_text)
        
        # Extract all feature types
        print("  📝 Extracting text features...")
        text_features = df_features['text'].apply(self.feature_engineer.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        print("  🏷️ Extracting hashtag features...")
        hashtag_features = df_features['hashtags'].apply(self.feature_engineer.extract_hashtag_features)
        hashtag_features_df = pd.DataFrame(hashtag_features.tolist())
        
        print("  😊 Extracting sentiment features...")
        sentiment_features = df_features['text'].apply(self.feature_engineer.extract_sentiment_features)
        sentiment_features_df = pd.DataFrame(sentiment_features.tolist())
        
        print("  👥 Extracting engagement features...")
        engagement_features = df_features.apply(
            lambda row: self.feature_engineer.extract_engagement_features(
                row.get('likes', 0), row.get('shares', 0), row.get('comments', 0)
            ), axis=1
        )
        engagement_features_df = pd.DataFrame(engagement_features.tolist())
        
        # Temporal features if available
        if 'timestamp' in df_features.columns:
            print("  ⏰ Extracting temporal features...")
            temporal_features = df_features['timestamp'].apply(self.feature_engineer.extract_temporal_features)
            temporal_features_df = pd.DataFrame(temporal_features.tolist())
        else:
            temporal_features_df = pd.DataFrame()
        
        # TF-IDF features
        print("  🔤 Extracting TF-IDF features...")
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=2)
        tfidf_matrix = tfidf.fit_transform(df_features['clean_text'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # Combine all features
        feature_dfs = [
            text_features_df,
            hashtag_features_df,
            sentiment_features_df,
            engagement_features_df,
            tfidf_df
        ]
        
        if not temporal_features_df.empty:
            feature_dfs.append(temporal_features_df)
        
        # Concatenate all features
        all_features = pd.concat(feature_dfs, axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(0)
        
        # Store feature columns and vectorizer
        self.feature_columns = all_features.columns.tolist()
        self.vectorizers['tfidf'] = tfidf
        
        print(f"  ✅ Total features created: {len(self.feature_columns)}")
        return all_features
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        print("🏋️ Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models_config.items():
            print(f"  🔄 Training {model_name}...")
            
            try:
                # Use scaled data for linear models
                if model_name in ['logistic_regression', 'svm']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"    ✅ {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"    ❌ {model_name}: Failed - {e}")
                continue
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
            self.best_model_name = best_model_name
            self.models = results
            
            print(f"🏆 Best model: {best_model_name}")
            
            return results
        else:
            raise Exception("No models trained successfully")
    
    def hyperparameter_tuning(self, X, y, model_name='random_forest'):
        """Perform hyperparameter tuning for the specified model"""
        print(f"🎯 Hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        # Get base model
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        model = base_models[model_name]
        param_grid = param_grids[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"  🎯 Best parameters: {grid_search.best_params_}")
        print(f"  📊 Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def predict(self, text, hashtags, likes=0, shares=0, comments=0, timestamp=None):
        """Make prediction for a single instance"""
        if not self.models or self.best_model_name not in self.models:
            raise Exception("No trained model available")
        
        # Create temporary dataframe
        temp_df = pd.DataFrame([{
            'text': text,
            'hashtags': hashtags,
            'likes': likes,
            'shares': shares,
            'comments': comments,
            'timestamp': timestamp
        }])
        
        # Build features
        features = self.build_single_sample_features(temp_df)
        
        # Get best model
        best_model_info = self.models[self.best_model_name]
        model = best_model_info['model']
        
        # Scale if needed
        if self.best_model_name in ['logistic_regression', 'svm']:
            features_scaled = self.scalers['standard'].transform(features)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance[:10]
        else:
            top_features = []
        
        confidence = np.max(probabilities)
        
        return prediction, confidence, top_features
    
    def build_single_sample_features(self, df):
        """Build features for a single sample using stored vectorizers"""
        # Reuse the feature engineering from training
        df_features = df.copy()
        df_features['clean_text'] = df_features['text'].apply(self.feature_engineer.preprocess_text)
        
        # Extract features (same as training)
        text_features = df_features['text'].apply(self.feature_engineer.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        hashtag_features = df_features['hashtags'].apply(self.feature_engineer.extract_hashtag_features)
        hashtag_features_df = pd.DataFrame(hashtag_features.tolist())
        
        sentiment_features = df_features['text'].apply(self.feature_engineer.extract_sentiment_features)
        sentiment_features_df = pd.DataFrame(sentiment_features.tolist())
        
        engagement_features = df_features.apply(
            lambda row: self.feature_engineer.extract_engagement_features(
                row.get('likes', 0), row.get('shares', 0), row.get('comments', 0)
            ), axis=1
        )
        engagement_features_df = pd.DataFrame(engagement_features.tolist())
        
        # Temporal features if available
        if 'timestamp' in df_features.columns:
            temporal_features = df_features['timestamp'].apply(self.feature_engineer.extract_temporal_features)
            temporal_features_df = pd.DataFrame(temporal_features.tolist())
        else:
            temporal_features_df = pd.DataFrame()
        
        # TF-IDF using stored vectorizer
        tfidf_matrix = self.vectorizers['tfidf'].transform(df_features['clean_text'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # Combine features
        feature_dfs = [
            text_features_df,
            hashtag_features_df,
            sentiment_features_df,
            engagement_features_df,
            tfidf_df
        ]
        
        if not temporal_features_df.empty:
            feature_dfs.append(temporal_features_df)
        
        all_features = pd.concat(feature_dfs, axis=1)
        all_features = all_features.fillna(0)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in all_features.columns:
                all_features[col] = 0
        
        # Reorder columns to match training
        all_features = all_features[self.feature_columns]
        
        return all_features
    
    def save_model(self, filepath='enhanced_movie_success_model.pkl'):
        """Save the complete model bundle"""
        model_bundle = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'scalers': self.scalers,
            'vectorizers': self.vectorizers,
            'feature_engineer': self.feature_engineer
        }
        
        joblib.dump(model_bundle, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='enhanced_movie_success_model.pkl'):
        """Load the complete model bundle"""
        model_bundle = joblib.load(filepath)
        
        self.models = model_bundle['models']
        self.best_model_name = model_bundle['best_model_name']
        self.feature_columns = model_bundle['feature_columns']
        self.scalers = model_bundle['scalers']
        self.vectorizers = model_bundle['vectorizers']
        self.feature_engineer = model_bundle['feature_engineer']
        
        print(f"📂 Model loaded from {filepath}")
        print(f"🏆 Best model: {self.best_model_name}")

def load_data(data_file):
    """Load data from various formats"""
    if data_file.endswith('.json'):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        raise ValueError('Unsupported file format')
    return df

def main():
    """Main training pipeline"""
    print("🎬 Enhanced Movie Success Prediction Pipeline")
    print("=" * 50)
    
    # Load data
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'your_data.json'
    print(f"📂 Loading data from {data_file}")
    
    try:
        df = load_data(data_file)
        print(f"✅ Loaded {len(df)} samples")
        
        if 'label' not in df.columns:
            print("❌ No 'label' column found in data")
            return
        
        # Initialize predictor
        predictor = EnhancedMovieSuccessPredictor()
        
        # Build features
        X = predictor.build_features(df)
        y = df['label']
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Label distribution: {y.value_counts().to_dict()}")
        
        # Train models
        results = predictor.train_models(X, y)
        
        # Optional: Hyperparameter tuning for best model
        tune_hyperparams = input("\n🎯 Perform hyperparameter tuning? (y/N): ").lower().strip() == 'y'
        
        if tune_hyperparams:
            best_tuned_model = predictor.hyperparameter_tuning(X, y, predictor.best_model_name)
            if best_tuned_model:
                # Update the best model
                predictor.models[predictor.best_model_name]['model'] = best_tuned_model
        
        # Save model
        predictor.save_model()
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_prediction, test_confidence, test_features = predictor.predict(
            "This movie is absolutely amazing! Best film of the year! #MustWatch #Epic",
            ["#MustWatch", "#Epic"],
            likes=1500, shares=300, comments=200
        )
        
        print(f"🎬 Test Prediction: {test_prediction}")
        print(f"🎯 Confidence: {test_confidence:.2%}")
        print(f"🔍 Top features: {test_features[:5]}")
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
