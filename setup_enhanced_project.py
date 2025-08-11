"""
Setup script for Enhanced Movie Success Prediction Project
Run this to set up your environment and check dependencies
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "enhanced_requirements.txt"
        ])
        print("  ✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to install requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    import nltk
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"  ✅ Downloaded {dataset}")
        except Exception as e:
            print(f"  ⚠️ Failed to download {dataset}: {e}")

def setup_spacy():
    """Set up spaCy language model"""
    print("🔤 Setting up spaCy...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("  ✅ spaCy English model installed")
        return True
    except subprocess.CalledProcessError:
        print("  ⚠️ spaCy model installation failed (optional)")
        return False

def create_env_template():
    """Create .env template file"""
    print("🔧 Creating environment template...")
    
    env_template = """# Enhanced Movie Success Prediction - API Keys
# Copy this file to .env and fill in your actual API keys

# Reddit API (https://www.reddit.com/prefs/apps/)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# YouTube Data API (https://console.cloud.google.com/)
YOUTUBE_API_KEY=your_youtube_api_key_here

# News API (https://newsapi.org/)
NEWS_API_KEY=your_news_api_key_here

# Optional: OpenAI API for advanced text analysis
OPENAI_API_KEY=your_openai_api_key_here
"""
    
    env_file = Path(".env.template")
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print(f"  ✅ Created {env_file}")
    print("  📝 Copy this to .env and add your API keys")

def check_dependencies():
    """Check if all dependencies are properly installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'nltk', 'textblob', 
        'vaderSentiment', 'requests', 'beautifulsoup4', 'streamlit',
        'plotly', 'matplotlib', 'seaborn', 'joblib', 'xgboost'
    ]
    
    optional_packages = [
        'transformers', 'torch', 'spacy', 'praw', 'wordcloud'
    ]
    
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            installed.append(package)
            print(f"  ✅ {package}")
        except pkg_resources.DistributionNotFound:
            missing.append(package)
            print(f"  ❌ {package} - REQUIRED")
    
    print("\n📋 Optional packages:")
    for package in optional_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"  ✅ {package}")
        except pkg_resources.DistributionNotFound:
            print(f"  ⚠️ {package} - optional")
    
    return len(missing) == 0

def test_imports():
    """Test importing key modules"""
    print("🧪 Testing imports...")
    
    test_modules = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('nltk', None),
        ('textblob', None),
        ('vaderSentiment', None),
        ('requests', None),
        ('streamlit', 'st'),
        ('plotly.express', 'px')
    ]
    
    for module, alias in test_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")

def create_sample_data():
    """Create sample data if none exists"""
    print("📊 Checking sample data...")
    
    if not os.path.exists('your_data.json'):
        print("  🔧 Creating sample data...")
        
        sample_data = [
            {
                "text": "This movie is absolutely amazing! Best film of the year! #MustWatch #Epic",
                "hashtags": ["#MustWatch", "#Epic"],
                "likes": 1500,
                "shares": 300,
                "comments": 200,
                "label": "Hit"
            },
            {
                "text": "Disappointing movie. Expected much more. #Overrated #Skip",
                "hashtags": ["#Overrated", "#Skip"],
                "likes": 50,
                "shares": 5,
                "comments": 20,
                "label": "Flop"
            },
            {
                "text": "Decent movie, nothing special but watchable. #Okay #Average",
                "hashtags": ["#Okay", "#Average"],
                "likes": 200,
                "shares": 30,
                "comments": 50,
                "label": "Average"
            }
        ]
        
        import json
        with open('your_data.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print("  ✅ Sample data created")
    else:
        print("  ✅ Sample data exists")

def main():
    """Main setup function"""
    print("🎬 Enhanced Movie Success Prediction - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        return False
    
    # Install requirements
    print("\n" + "=" * 50)
    if not install_requirements():
        print("❌ Setup failed: Could not install requirements")
        return False
    
    # Download NLTK data
    print("\n" + "=" * 50)
    download_nltk_data()
    
    # Setup spaCy (optional)
    print("\n" + "=" * 50)
    setup_spacy()
    
    # Create environment template
    print("\n" + "=" * 50)
    create_env_template()
    
    # Check dependencies
    print("\n" + "=" * 50)
    if not check_dependencies():
        print("⚠️ Some required dependencies are missing")
    
    # Test imports
    print("\n" + "=" * 50)
    test_imports()
    
    # Create sample data
    print("\n" + "=" * 50)
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Run: python enhanced_data_collector.py (with API keys)")
    print("3. Run: python enhanced_ml_pipeline.py")
    print("4. Run: streamlit run enhanced_dashboard.py")
    print("\n📖 See ENHANCED_SETUP_GUIDE.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    main()
