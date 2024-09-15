import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
import torch

# Convert to DataFrame
df = pd.read_csv('sample.csv')

# Check the column names to ensure 'document' exists
print(df.columns)

# Handle NaN by filling missing text with an empty string and convert all values to strings
df['text'] = df['document'].fillna('').astype(str)

# Load pre-trained sentiment analysis model and tokenizer
classifier = pipeline('sentiment-analysis')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Function to get sentiment
def get_sentiment(text):
    if text.strip():  # Ensure we have non-empty text
        # Tokenize and truncate input text
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors='pt', padding='max_length')
        outputs = classifier.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = classifier.model.config.id2label[prediction]
        return sentiment
    return 'neutral'  # Assign 'neutral' for empty text or NaN

# Apply sentiment analysis on the cleaned 'text' column
df['sentiment'] = df['text'].apply(get_sentiment)

# Group by speaker and get sentiment counts
speaker_sentiments = df.groupby(['speaker', 'sentiment']).size().unstack(fill_value=0)

# Plotting the sentiment distribution for each speaker
speaker_sentiments.plot(kind='bar', stacked=True)
plt.title('Sentiment Distribution by Speaker')
plt.xlabel('Speakers')
plt.ylabel('Count of Sentiments')
plt.show()
