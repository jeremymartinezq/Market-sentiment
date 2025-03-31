from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

def sentiment_analysis(text):
    # Using TextBlob for basic sentiment analysis
    blob = TextBlob(text)
    textblob_score = blob.sentiment.polarity

    # Using VADER for enhanced sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(text)['compound']

    return textblob_score, vader_score

# Example texts
texts = [
    "The stock market is looking very bullish today!",
    "The recent market downturn has been concerning.",
    "Tech stocks are experiencing unprecedented growth."
]

results = {'Text': [], 'TextBlob Score': [], 'VADER Score': []}

for text in texts:
    tb_score, vd_score = sentiment_analysis(text)
    results['Text'].append(text)
    results['TextBlob Score'].append(tb_score)
    results['VADER Score'].append(vd_score)

# Convert results to DataFrame
df = pd.DataFrame(results)

# Plotting sentiment scores
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(kind='bar', x='Text', y=['TextBlob Score', 'VADER Score'], ax=ax, color=['skyblue', 'salmon'])
plt.title('Sentiment Scores Analysis')
plt.xlabel('Text')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
