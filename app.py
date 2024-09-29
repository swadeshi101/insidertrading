import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np

# Streamlit UI components
st.title("Insider Trading Detection using Sentiment Analysis and Financial Data")

# User input for stock ticker
stock_ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL):", "AAPL")

# Fetch stock data using Alpha Vantage API
API_KEY = 'K6IMQGA8ZU095MNO'  # Alpha Vantage API Key
ts = TimeSeries(key=API_KEY, output_format='pandas')

if stock_ticker:
    stock_data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='full')
    st.write(f"Stock data for {stock_ticker}")
    st.dataframe(stock_data.head())

# Sentiment Analysis section
st.subheader("Sentiment Analysis of News Headlines")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample news data (this would typically come from a news scraping process or NewsAPI)
news_data = [
    {"date": "2023-01-10", "headline": "Apple announces record-breaking sales quarter."},
    {"date": "2023-02-05", "headline": "Reports suggest Apple facing supply chain issues."}
]

# Function to analyze sentiment of news headlines
def analyze_sentiment(news_data):
    sentiments = []
    for news in news_data:
        sentiment_score = analyzer.polarity_scores(news['headline'])['compound']
        sentiments.append({
            "date": news['date'],
            "headline": news['headline'],
            "sentiment_score": sentiment_score
        })
    return pd.DataFrame(sentiments)

# Get sentiment scores
sentiment_data = analyze_sentiment(news_data)
sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
st.write("Sentiment Data")
st.dataframe(sentiment_data)

# Visualizing the sentiment scores
st.subheader("Sentiment Score Visualization")
st.line_chart(sentiment_data[['date', 'sentiment_score']].set_index('date'))

# Logistic Regression model (assuming X and y are pre-defined)
st.subheader("Logistic Regression Model Performance")

# Example data for X and y (normally, these should come from the sentiment and financial data)
X = np.random.rand(100, 3)  # Placeholder for feature data
y = np.random.randint(0, 2, 100)  # Placeholder for target data

# Initialize and train the model
model = LogisticRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Display Cross-Validation results
st.write("Cross-Validation Accuracy Scores:", cv_scores)
st.write("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

# Plotting ROC-AUC and Precision-Recall Curves
st.subheader("Model Evaluation")

model.fit(X, y)
y_scores = model.predict_proba(X)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y, y_scores)
fpr, tpr, _ = roc_curve(y, y_scores)

# Precision-Recall
precision, recall, _ = precision_recall_curve(y, y_scores)

# Plot ROC-AUC and Precision-Recall curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic')
ax1.legend(loc='lower right')

# Precision-Recall Curve
ax2.plot(recall, precision, label='Precision-Recall curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend()

st.pyplot(fig)

# Ending the Streamlit app
st.write("End of Analysis")
