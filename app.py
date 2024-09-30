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

# Function to fetch and analyze news sentiment using Alpha Vantage
@st.cache_data
def get_news_sentiment(ticker):
    api_key = 'X9MGIVT9R81BY4PF'  # Replace with your Alpha Vantage API key
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
    
    return pd.DataFrame(sentiment_scores)

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
def plot_sentiment_anomalies(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['sentiment_score'], name='Sentiment Score'))
    anomalies = data[data['anomaly'] == -1]
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['sentiment_score'], mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
    fig.update_layout(title="Sentiment Score with Anomalies", xaxis_title="Date", yaxis_title="Sentiment Score")
    return fig

# Main app logic
if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        # Get stock data
        stock_data = get_stock_data(ticker, start_date, end_date)
        
        # Get news sentiment
        sentiment_data = get_news_sentiment(ticker)
        if sentiment_data.empty:
            st.error("Failed to fetch news data. Please check your API key and try again.")
        else:
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            sentiment_data.set_index('date', inplace=True)
            
            # Combine stock and sentiment data
            combined_data = stock_data.join(sentiment_data['sentiment_score'], how='left')
            combined_data['sentiment_score'].fillna(method='ffill', inplace=True)
            
            # Feature engineering
            combined_data['rolling_mean'] = combined_data['sentiment_score'].rolling(window=sentiment_window).mean()
            combined_data['rolling_std'] = combined_data['sentiment_score'].rolling(window=sentiment_window).std()
            
            # Anomaly detection
            iso_forest = IsolationForest(contamination=anomaly_threshold, random_state=42)
            combined_data['anomaly'] = iso_forest.fit_predict(combined_data[['Close', 'sentiment_score']])
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Analysis", "😊 Sentiment Analysis", "🔍 Anomaly Detection", "📰 News Headlines"])
            
            with tab1:
                st.header("📈 Stock Price Analysis")
                st.plotly_chart(plot_stock_chart(stock_data), use_container_width=True)
                
                # Display key statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}", f"{stock_data['Close'].pct_change().iloc[-1]:.2%}")
                col2.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
                col3.metric("52-Week High", f"${stock_data['Close'].rolling(window=252).max().iloc[-1]:.2f}")
            
            with tab2:
                st.header("😊 Sentiment Analysis")
                st.plotly_chart(plot_sentiment_anomalies(combined_data), use_container_width=True)
                
                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                sns.histplot(combined_data['sentiment_score'], kde=True, ax=ax)
                st.pyplot(fig)
            
            with tab3:
                st.header("🔍 Anomaly Detection")
                
                # Display anomalies
                anomalies = combined_data[combined_data['anomaly'] == -1]
                st.dataframe(anomalies[['Close', 'sentiment_score']])
                
                # Correlation heatmap
                st.subheader("Feature Correlation")
                corr_matrix = combined_data[['Close', 'sentiment_score', 'rolling_mean', 'rolling_std']].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            
            with tab4:
                st.header("📰 Recent News Headlines")
                for _, news in sentiment_data.sort_index(ascending=False).head(10).iterrows():
                    st.markdown(f"**[{news['title']}]({news['url']})**")
                    st.markdown(f"Sentiment Score: {news['sentiment_score']:.2f}")
                    st.markdown("---")

# Instructions (same as before)
st.sidebar.markdown("---")
st.sidebar.header("📝 Instructions")
st.sidebar.markdown("""
1. Enter a stock ticker symbol (e.g., AAPL for Apple Inc.)
2. Select the date range for analysis
3. Adjust advanced options if needed
4. Click 'Run Analysis' to see the results
5. Explore the different tabs for detailed insights
""")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Your Name | Data source: Yahoo Finance & Alpha Vantage")

# Add a fun fact about insider trading
st.sidebar.markdown("---")
st.sidebar.header("💡 Did You Know?")
fun_facts = [
    "The term 'insider trading' was coined in the 1960s.",
    "The first insider trading case in the US was in 1909.",
    "Some countries allow certain forms of insider trading.",
    "Insider trading can sometimes be detected through unusual options activity.",
    "The SEC uses AI and machine learning to detect potential insider trading."
]
st.sidebar.info(np.random.choice(fun_facts))
