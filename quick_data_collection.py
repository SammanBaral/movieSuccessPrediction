"""
Quick Movie Data Collection Script
Run this immediately to get started with real data
"""

import json
import pandas as pd
from free_data_collector import FreeMovieDataCollector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_data_collection():
    """
    Quick data collection for immediate use
    """
    print("🚀 Quick Movie Data Collection")
    print("=" * 50)
    
    collector = FreeMovieDataCollector()
    
    # Start with popular recent movies that should have good data
    movies_to_collect = [
        ("Oppenheimer", "2023"),
        ("Barbie", "2023"),
        ("Spider-Man No Way Home", "2021"),
        ("Top Gun Maverick", "2022"),
        ("Avatar The Way of Water", "2022")
    ]
    
    all_movie_data = []
    
    for i, (movie_title, year) in enumerate(movies_to_collect, 1):
        print(f"\n🎬 [{i}/{len(movies_to_collect)}] Processing: {movie_title} ({year})")
        
        try:
            # Collect data for this movie
            movie_data, movie_info = collector.collect_comprehensive_movie_data(
                movie_title, year, total_samples=200  # 200 samples per movie
            )
            
            if movie_data:
                all_movie_data.extend(movie_data)
                
                # Save individual movie data
                filename = f"movie_data_{movie_title.lower().replace(' ', '_').replace(':', '')}.json"
                collector.save_data(movie_data, filename)
                
                print(f"  ✅ Collected {len(movie_data)} samples")
                print(f"  💾 Saved to {filename}")
                
                # Show label distribution
                labels = {}
                for item in movie_data:
                    labels[item['label']] = labels.get(item['label'], 0) + 1
                print(f"  📊 Labels: {labels}")
                
            else:
                print(f"  ❌ No data collected for {movie_title}")
                
        except Exception as e:
            print(f"  ❌ Error collecting {movie_title}: {e}")
            continue
    
    # Save combined dataset
    if all_movie_data:
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 30)
        print(f"Total samples collected: {len(all_movie_data)}")
        
        # Overall label distribution
        all_labels = {}
        all_sources = {}
        for item in all_movie_data:
            all_labels[item['label']] = all_labels.get(item['label'], 0) + 1
            all_sources[item['source']] = all_sources.get(item['source'], 0) + 1
        
        print(f"Label distribution: {all_labels}")
        print(f"Source distribution: {all_sources}")
        
        # Save combined dataset
        combined_filename = "combined_movie_dataset.json"
        collector.save_data(all_movie_data, combined_filename)
        print(f"💾 Combined dataset saved to: {combined_filename}")
        
        # Create a sample for immediate testing
        sample_data = all_movie_data[:50]  # First 50 samples
        sample_filename = "quick_test_dataset.json"
        collector.save_data(sample_data, sample_filename)
        print(f"🧪 Test dataset (50 samples) saved to: {sample_filename}")
        
        return True
    else:
        print("❌ No data collected at all!")
        return False

def verify_data_quality(filename):
    """
    Quick verification of collected data quality
    """
    print(f"\n🔍 Verifying data quality: {filename}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        print(f"  📊 Dataset shape: {df.shape}")
        print(f"  📝 Required columns present: {set(['text', 'hashtags', 'likes', 'shares', 'comments', 'label']).issubset(df.columns)}")
        print(f"  🏷️ Label distribution:")
        for label, count in df['label'].value_counts().items():
            print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"  📈 Engagement stats:")
        print(f"    Avg likes: {df['likes'].mean():.0f}")
        print(f"    Avg shares: {df['shares'].mean():.0f}")
        print(f"    Avg comments: {df['comments'].mean():.0f}")
        
        print(f"  📝 Sample texts:")
        for i, text in enumerate(df['text'].head(3)):
            print(f"    {i+1}. {text[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Movie Success Prediction - Quick Data Collection")
    print("This script will collect real movie data immediately!")
    print()
    
    # Run data collection
    success = quick_data_collection()
    
    if success:
        print(f"\n🎉 SUCCESS! Data collection completed!")
        print(f"📁 Files created:")
        print(f"  - combined_movie_dataset.json (full dataset)")
        print(f"  - quick_test_dataset.json (test subset)")
        print(f"  - movie_data_*.json (individual movies)")
        
        # Verify the combined dataset
        verify_data_quality("combined_movie_dataset.json")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Run: python enhanced_ml_pipeline.py combined_movie_dataset.json")
        print(f"2. Run: streamlit run enhanced_dashboard.py")
        print(f"3. Compare with your original model performance!")
        
    else:
        print(f"\n❌ Data collection failed. Check the errors above.")
        print(f"💡 You can still use your existing data to test the enhanced pipeline:")
        print(f"   python enhanced_ml_pipeline.py your_data.json")
