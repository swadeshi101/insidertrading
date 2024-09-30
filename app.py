import streamlit as st
import datetime
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np

# Set Streamlit layout
st.set_page_config(
    page_title="Insider Trading Detection App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("💼 Insider Trading Detection App")
st.subheader("Detect insider trading patterns using sentiment analysis and financial data")

# Sidebar for user inputs
st.sidebar.header("User Configuration")
stock_ticker = st.sidebar.text_input("Enter the stock ticker symbol (e.g., AAPL):", "AAPL")

# Date input fix - date range selection
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date(2023, 12, 31))

# Ensure start date is before end date
if start_date > end_date:
    st.sidebar.error("Error: End date must be after start date.")
else:
    date_range = (start_date, end_date)

# Fetch stock data from Alpha Vantage
API_KEY = 'X9MGIVT9R81BY4PF'  # Replace this with your valid Alpha Vantage API Key
ts = TimeSeries(key=API_KEY, output_format='pandas')

try:
    if stock_ticker:
        stock_data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='full')
        stock_data.index = pd.to_datetime(stock_data.index)  # Ensure index is datetime
        
        # Filter stock data by selected date range
        stock_data_filtered = stock_data.loc[date_range[0]:date_range[1]]
        
        if stock_data_filtered.empty:
            st.write(f"No stock data available for {stock_ticker} in the selected date range.")
        else:
            st.write(f"Stock data for {stock_ticker} from {start_date} to {end_date}")
            st.dataframe(stock_data_filtered.head())

            # Plot stock price data
            st.subheader("📈 Stock Price Visualization")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=stock_data_filtered['4. close'], ax=ax, color="orange")
            ax.set_title(f"Closing Prices of {stock_ticker} (Filtered)", fontsize=16)
            ax.set_ylabel("Price")
            ax.set_xlabel("Date")
            st.pyplot(fig)
except Exception as e:
    st.error(f"Error fetching stock data: {str(e)}")

# Sentiment Analysis section
st.subheader("📰 Sentiment Analysis of News Headlines")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample news data (this would typically come from a news scraping process or NewsAPI)
news_data = [
    {"date": "2023-01-10", "headline": "Apple announces record-breaking sales quarter."},
    {"date": "2023-02-05", "headline": "Reports suggest Apple facing supply chain issues."},
    # Add more headlines as necessary
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

# Plot sentiment score data
st.subheader("📊 Sentiment Score Visualization")
if sentiment_data.empty:
    st.write("No sentiment data available.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='date', y='sentiment_score', data=sentiment_data, marker="o", color="green", ax=ax)
    ax.set_title("Sentiment Analysis Over Time", fontsize=16)
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Date")
    st.pyplot(fig)

# Display raw sentiment data
st.write("Sentiment Data")
st.dataframe(sentiment_data)

# Logistic Regression model (assuming X and y are pre-defined)
st.subheader("📉 Logistic Regression Model Performance")

# Example data for X and y (normally, these should come from the sentiment and financial data)
X = np.random.rand(100, 3)  # Placeholder for feature data
y = np.random.randint(0, 2, 100)  # Placeholder for target data

# Initialize and train the model
model = LogisticRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Display Cross-Validation results
st.write("Cross-Validation Accuracy Scores:", cv_scores)
st.write("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

# Model Evaluation with ROC-AUC and Precision-Recall curves
st.subheader("📊 Model Evaluation")

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
ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color="blue")
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic')
ax1.legend(loc='lower right')

# Precision-Recall Curve
ax2.plot(recall, precision, label='Precision-Recall curve', color="purple")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend()

st.pyplot(fig)

# End of the app
st.write("🎉 End of Analysis - Thank you for using the app!")
