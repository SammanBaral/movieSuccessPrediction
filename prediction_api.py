"""
Flask API for Live Social Media Movie Prediction
Provides REST API endpoints for the dashboard
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from live_social_predictor import SocialMediaMoviePredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the predictor
predictor = SocialMediaMoviePredictor()

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movie Success Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #2196F3; font-weight: bold; }
            .url { font-family: monospace; background: #e8e8e8; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>🎬 Movie Success Prediction API</h1>
        <p>Real-time social media analysis for movie success prediction</p>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/predict</div>
            <p><strong>Body:</strong> {"movie_name": "Movie Title"}</p>
            <p><strong>Description:</strong> Predicts movie success based on live social media data</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/health</div>
            <p><strong>Description:</strong> Check API health status</p>
        </div>
        
        <h2>Example Usage:</h2>
        <pre>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"movie_name": "Oppenheimer"}'
        </pre>
        
        <h2>Dashboard Integration:</h2>
        <p>This API is designed to work with the Movie Success Prediction Dashboard.</p>
        <p>Access the dashboard at: <a href="http://localhost:8080">http://localhost:8080</a></p>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict_movie():
    """Predict movie success based on social media data"""
    try:
        data = request.get_json()
        
        if not data or 'movie_name' not in data:
            return jsonify({
                'error': 'Missing movie_name in request body',
                'example': {'movie_name': 'Oppenheimer'}
            }), 400
        
        movie_name = data['movie_name'].strip()
        
        if not movie_name:
            return jsonify({
                'error': 'Movie name cannot be empty'
            }), 400
        
        # Get prediction
        results = predictor.predict_movie_success(movie_name)
        
        return jsonify({
            'success': True,
            'data': results,
            'message': f'Prediction completed for {movie_name}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process prediction request'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Movie Success Prediction API',
        'version': '1.0.0',
        'features': [
            'Real-time social media analysis',
            'Sentiment analysis',
            'Engagement metrics',
            'Success prediction'
        ]
    })

@app.route('/recent-predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent prediction results"""
    try:
        # Look for recent prediction files
        prediction_files = [f for f in os.listdir('.') if f.startswith('prediction_') and f.endswith('.json')]
        
        recent_predictions = []
        for file in sorted(prediction_files, reverse=True)[:10]:  # Last 10 predictions
            try:
                with open(file, 'r') as f:
                    prediction_data = json.load(f)
                    recent_predictions.append({
                        'movie_name': prediction_data['movie_name'],
                        'prediction': prediction_data['prediction'],
                        'confidence': prediction_data['confidence'],
                        'timestamp': prediction_data['timestamp']
                    })
            except:
                continue
        
        return jsonify({
            'success': True,
            'data': recent_predictions,
            'count': len(recent_predictions)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Movie Success Prediction API...")
    print("📡 API will be available at: http://localhost:5000")
    print("📊 Dashboard available at: http://localhost:8080")
    print("💡 Use /predict endpoint to get movie predictions")
    print("🔧 Use /health to check API status")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
