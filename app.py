import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Insider Trading Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (same as before)
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    color: #ffffff;
    background-color: #4682b4;
    border-radius: 5px;
}
.stTextInput>div>div>input {
    border-radius: 5px;
}
.stTab {
    background-color: #e6f2ff;
    color: #4682b4;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🕵️‍♂️ Insider Trading Detection Dashboard")
st.markdown("Analyze stock sentiment and detect potential anomalies in trading patterns.")

# Sidebar for user input
st.sidebar.header("📊 Analysis Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL")
start_date = st.sidebar.date_input("Start Date:", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date:", datetime.now())

# Advanced options
st.sidebar.header("🔧 Advanced Options")
anomaly_threshold = st.sidebar.slider("Anomaly Detection Threshold", 0.01, 0.2, 0.1, 0.01)
sentiment_window = st.sidebar.slider("Sentiment Rolling Window", 1, 7, 3, 1)

# Function to fetch stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Function to clean and process date strings
def clean_and_parse_date(date_str):
    # Split the string if it contains 'T' and take only the date part
    cleaned_date = date_str.split('T')[0] if 'T' in date_str else date_str
    try:
        return pd.to_datetime(cleaned_date)
    except ValueError:
        return pd.NaT  # Return NaT if parsing fails

# Function to fetch and analyze news sentiment using Alpha Vantage
@st.cache_data
def get_news_sentiment(ticker):
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching news data: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    if 'feed' not in data:
        st.error("No news data available for the given ticker.")
        return pd.DataFrame()
    
    articles = data['feed']
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for article in articles:
        sentiment = analyzer.polarity_scores(article['title'])
        sentiment_scores.append({
            'date': article['time_published'][:10],
            'sentiment_score': sentiment['compound'],
            'title': article['title'],
            'url': article['url']
        })
    
    sentiment_data = pd.DataFrame(sentiment_scores)
    
    # Clean and parse the dates safely
    sentiment_data['date'] = sentiment_data['date'].apply(clean_and_parse_date)
    sentiment_data.dropna(subset=['date'], inplace=True)  # Drop rows with invalid dates
    
    return sentiment_data

# Function to plot interactive stock chart
def plot_stock_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Stock Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), name='20-day MA'))
    fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
    return fig

# Function to plot sentiment and anomalies
# Function to plot sentiment and anomalies
def plot_sentiment_and_anomalies(combined_data):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price', color='tab:blue')
    ax1.plot(combined_data.index, combined_data['Close'], color='tab:blue', label='Stock Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sentiment Score', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(combined_data.index, combined_data['sentiment_score'], color='tab:red', label='Sentiment Score')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Highlight anomalies
    anomalies = combined_data[combined_data['anomaly'] == -1]
    ax1.scatter(anomalies.index, anomalies['Close'], color='orange', label='Anomaly', zorder=5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'{ticker} Stock Price and Sentiment with Anomalies')
    plt.show()

# Function for anomaly detection
def detect_anomalies(data, threshold=0.1):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume']])
    
    isolation_forest = IsolationForest(contamination=threshold)
    data['anomaly'] = isolation_forest.fit_predict(scaled_data)
    
    return data

# Main function
if ticker:
    stock_data = get_stock_data(ticker, start_date, end_date)
    sentiment_data = get_news_sentiment(ticker)

    if not sentiment_data.empty:
        # Convert stock and sentiment data to the same timezone format (tz-naive)
        stock_data.index = stock_data.index.tz_localize(None)
        sentiment_data['date'] = sentiment_data['date'].dt.tz_localize(None)

        # Combine stock data and sentiment data on the date
        combined_data = stock_data.join(sentiment_data.set_index('date')['sentiment_score'], how='left')
        combined_data['sentiment_score'].fillna(method='ffill', inplace=True)  # Fill missing sentiment values
        combined_data = detect_anomalies(combined_data, anomaly_threshold)

        # Plot stock chart
        st.plotly_chart(plot_stock_chart(stock_data), use_container_width=True)

        # Plot sentiment and anomalies
        st.pyplot(plot_sentiment_and_anomalies(combined_data))
    else:
        st.error("No sentiment data available to plot.")
else:
    st.warning("Please enter a valid stock ticker.")
