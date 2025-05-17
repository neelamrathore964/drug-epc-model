import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def convert_rating_to_sentiment(rating):
    if rating >= 7:
        return 'positive'
    elif rating <= 4:
        return 'negative'
    else:
        return 'neutral'

def preprocess(df):
    df = df.dropna(subset=['condition']).copy() # Drop rows with missing condition
    df['clean_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['rating'].apply(convert_rating_to_sentiment) 
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
    return df[['clean_review', 'sentiment', 'sentiment_encoded']]

def data_summary(train_clean, test_clean):
    print("📦 Data Summary")
    print(f"Total training records: {len(train_clean)}")
    print(f"Total test records: {len(test_clean)}")

    print("\n Sentiment distribution in training data:")
    print(train_clean['sentiment'].value_counts(normalize=True) * 100)

def eda(train_clean, folder_path):
    sns.countplot(x='sentiment', data=train_clean)
    plt.title('Sentiment Distribution')
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "sentiment_analysis_train.png"))
    plt.close()
