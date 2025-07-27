import json
import pandas as pd

# Load the generated data
df = pd.read_json('twitter_data_Oppenheimer.json')

print("🎬 Generated Data Analysis")
print("=" * 40)
print(f"Total samples: {len(df)}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")
print(f"Average likes: {df['likes'].mean():.1f}")
print(f"Average shares: {df['shares'].mean():.1f}")
print(f"Average comments: {df['comments'].mean():.1f}")

print("\n📊 Sample Posts:")
print("-" * 40)
for i, row in df.head(5).iterrows():
    print(f"{i+1}. [{row['label']}] {row['text']}")
    print(f"   Hashtags: {row['hashtags']}")
    print(f"   Engagement: {row['likes']} likes, {row['shares']} shares, {row['comments']} comments")
    print()

print("✅ Data is ready for ML training!")
