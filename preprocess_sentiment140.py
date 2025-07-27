import pandas as pd
import re
import json
import random

# Sentiment140 columns: 0=sentiment, 1=id, 2=date, 3=flag, 4=user, 5=text
# Map: 4 (positive) -> Hit, 2 (neutral) -> Average, 0 (negative) -> Flop
sentiment_map = {4: 'Hit', 2: 'Average', 0: 'Flop'}

# Read a sample (change nrows for more/less data)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, usecols=[0,5], names=['sentiment','text'], nrows=10000)

def extract_hashtags(text):
    return re.findall(r'#\w+', text)

def make_sample(row):
    return {
        'text': row['text'],
        'hashtags': extract_hashtags(row['text']),
        'likes': random.randint(0, 1000),
        'shares': random.randint(0, 300),
        'comments': random.randint(0, 100),
        'label': sentiment_map.get(row['sentiment'], 'Average')
    }

# Convert
samples = [make_sample(row) for _, row in df.iterrows()]

# Save as JSON
with open('sentiment140_for_pipeline.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Saved {len(samples)} samples to sentiment140_for_pipeline.json") 