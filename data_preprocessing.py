import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

print("Starting preprocessing...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
    return " ".join(cleaned_tokens)

try:
    df = pd.read_csv("simulated_feedback.csv")
except FileNotFoundError:
    print("Error: simulated_feedback.csv not found.")
    print("Please run simulate_data.py first.")
    exit()

original_rows = len(df)
df.dropna(subset=['feedback'], inplace=True)
print(f"Removed {original_rows - len(df)} rows with missing feedback.")

original_rows = len(df)
df.drop_duplicates(subset=['feedback_id', 'feedback'], inplace=True)
print(f"Removed {original_rows - len(df)} duplicate rows.")

print("Applying NLTK preprocessing...")
df['preprocessed_feedback'] = df['feedback'].apply(preprocess_text)

df.to_csv("cleaned_feedback.csv", index=False)

print(f"Preprocessing complete. Saved cleaned_feedback.csv")
print("\n--- Cleaned Data Head ---")
print(df.head())