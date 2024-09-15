import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text (tokenize, remove stopwords, lemmatize)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    # Remove stopwords and non-alphabetic words, then lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

df = pd.read_csv('sample.csv')

# Step 1: Handle NaN or non-string values in the 'document' column
df['document'] = df['document'].apply(lambda x: str(x) if isinstance(x, str) else '')

# Step 2: Get the number of unique speakers
num_speakers = df['speaker'].nunique()

# Step 3: Group by speaker and calculate total speaking time
speaker_duration = df.groupby('speaker')['duration'].sum()

# Step 4: Group by speaker and aggregate all their sentences into one document
speaker_docs = df.groupby('speaker')['document'].apply(lambda x: ' '.join(x)).reset_index()

# Step 5: NLP Preprocessing and Keyword Frequency Analysis
def analyze_keywords(text):
    tokens = preprocess_text(text)
    return Counter(tokens)

# Add a new column with preprocessed keyword frequency for each speaker
speaker_docs['keyword_frequency'] = speaker_docs['document'].apply(analyze_keywords)

# Step 6: Combine results into speaker_summary DataFrame
speaker_summary = pd.merge(speaker_duration.reset_index(), speaker_docs, on='speaker')

# Save the final speaker_summary as HTML
speaker_summary.to_html('speaker_summary.html')

# Output the number of speakers and the summary DataFrame
print(f"Number of unique speakers: {num_speakers}")
print(speaker_summary[['speaker', 'duration', 'keyword_frequency']])
