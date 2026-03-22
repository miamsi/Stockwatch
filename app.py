import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Institutional Equity Dashboard", layout="wide", page_icon="📈")

st.title("📈 Institutional Equity Intelligence Dashboard")
st.markdown("""
*Built for Quantitative Portfolio Analysis & Strategy Consulting* This dashboard integrates fundamental valuation, macroeconomic correlations (IHSG/IDR), and real-time sentiment analysis.
""")

# ==========================================
# 2. SIDEBAR CONTEXT & INPUTS
# ==========================================
st.sidebar.header("Data Parameters")
ticker_input = st.sidebar.text_input("Enter Ticker (e.g., BBCA.JK, ASII.JK, AAPL)", "BBCA.JK")
lookback_years = st.sidebar.slider("Historical Lookback (Years)", 1, 5, 2)

@st.cache_data(ttl=3600)
def fetch_data(ticker, years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    
    stock_df = yf.download(ticker, start=start_date, end=end_date)
    ihsg_df = yf.download("^JKSE", start=start_date, end=end_date)
    idr_df = yf.download("IDR=X", start=start_date, end=end_date)
    
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    news = ticker_obj.news
    
    return stock_df, ihsg_df, idr_df, info, news

try:
    with st.spinner(f"Fetching institutional data for {ticker_input}..."):
        stock_data, ihsg_data, idr_data, stock_info, stock_news = fetch_data(ticker_input, lookback_years)

    # 👉 NEW: Tabs for product-level UI
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Fundamentals",
        "🌍 Macro",
        "📰 Sentiment",
        "🔮 Scenario"
    ])

    # ==========================================
    # 3. MODULE A: FUNDAMENTAL ENGINE
    # ==========================================
    with tab1:
        st.header(f"Module A: Fundamental Health - {stock_info.get('shortName', ticker_input)}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = stock_info.get('currentPrice', 'N/A')
            st.metric("Current Price", f"{stock_info.get('currency', '')} {current_price}")
        with col2:
            pe_ratio = stock_info.get('trailingPE', 'N/A')
            st.metric("Trailing P/E", round(pe_ratio, 2) if isinstance(pe_ratio, float) else pe_ratio)
        with col3:
            profit_margin = stock_info.get('profitMargins', 0)
            st.metric("Profit Margin", f"{round(profit_margin * 100, 2)}%")
        with col4:
            debt_to_equity = stock_info.get('debtToEquity', 'N/A')
            st.metric("Debt to Equity", debt_to_equity)

        # 👉 UPGRADED Health Score (keeps your base logic but enhances it)
        health_score = 0

        if isinstance(pe_ratio, float):
            if pe_ratio < 15:
                health_score += 30
            elif pe_ratio < 25:
                health_score += 20

        if isinstance(profit_margin, float):
            health_score += min(profit_margin * 100, 30)

        if isinstance(debt_to_equity, float):
            health_score += max(0, 40 - debt_to_equity * 0.2)

        health_score = min(int(health_score), 100)

        st.progress(health_score / 100, text=f"Enhanced Fundamental Health Score: {health_score}/100")

        # 👉 NEW: Volatility insight
        returns = stock_data['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)

        st.metric("Annualized Volatility", f"{round(volatility*100,2)}%")

    # ==========================================
    # 4. MODULE B & C: MACRO CORRELATION
    # ==========================================
    with tab2:
        st.header("Module B: Macro & Index Correlation (IHSG / IDR)")
        
        stock_pct = (stock_data['Close'] / stock_data['Close'].iloc[0] - 1) * 100
        ihsg_pct = (ihsg_data['Close'] / ihsg_data['Close'].iloc[0] - 1) * 100
        idr_pct = (idr_data['Close'] / idr_data['Close'].iloc[0] - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_pct.index, y=stock_pct.squeeze(), name=f"{ticker_input}", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=ihsg_pct.index, y=ihsg_pct.squeeze(), name="IHSG", line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=idr_pct.index, y=idr_pct.squeeze(), name="USD/IDR", line=dict(dash='dot')))
        
        fig.update_layout(title="Relative Performance (%)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 👉 NEW: Correlation insight
        combined = pd.concat([stock_pct, ihsg_pct, idr_pct], axis=1).dropna()
        corr = combined.corr()

        st.subheader("Correlation Matrix")
        st.dataframe(corr)

    # ==========================================
    # 5. MODULE C: SENTIMENT ANALYSIS
    # ==========================================
    with tab3:
        st.header("Module C: NLP Sentiment Analysis")
        
        if stock_news:
            sentiments = []
            weights = [1, 0.9, 0.8, 0.7, 0.6]

            for i, article in enumerate(stock_news[:5]):
                headline = article.get('title', '')
                analysis = TextBlob(headline)
                sentiments.append(analysis.sentiment.polarity * weights[i])

            avg_sentiment = sum(sentiments) / sum(weights)

            sentiment_label = "Bullish 🐂" if avg_sentiment > 0.1 else "Bearish 🐻" if avg_sentiment < -0.1 else "Neutral ⚖️"

            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.metric("Weighted Sentiment", sentiment_label, round(avg_sentiment, 2))
            with col_s2:
                with st.expander("View Headlines"):
                    for article in stock_news[:5]:
                        st.write(f"- {article.get('title', 'No Title')}")

        else:
            avg_sentiment = 0
            st.info("No recent news found.")

    # ==========================================
    # 6. MODULE D: SCENARIO ANALYSIS
    # ==========================================
    with tab4:
        st.header("Module D: Scenario Analysis & Valuation Adjustments")
        
        st.latex(r"P_{projected} = P_{current} \times (1 + \Delta Growth) \times (1 - \Delta Risk \cdot \beta)")

        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            growth_slider = st.slider("Expected Growth (%)", -20, 50, 5)
        with col_sc2:
            risk_slider = st.slider("Macro Risk (%)", 0, 30, 5)

        beta = stock_info.get("beta", 1)

        if isinstance(current_price, (int, float)):
            projected_price = current_price * (1 + (growth_slider / 100)) * (1 - (risk_slider / 100) * beta)

            st.success(f"### Adjusted Target Price: {stock_info.get('currency', '')} {round(projected_price, 2)}")
            st.caption(f"Incorporates Beta (Market Sensitivity): {beta}")

    # ==========================================
    # 7. 👉 FINAL CONSULTANT OUTPUT (NEW)
    # ==========================================
    st.markdown("---")
    st.header("📊 Final Investment Signal")

    if health_score > 70 and avg_sentiment > 0:
        decision = "BUY 🟢"
    elif health_score < 40 and avg_sentiment < 0:
        decision = "SELL 🔴"
    else:
        decision = "HOLD 🟡"

    st.subheader(f"Recommendation: {decision}")

except Exception as e:
    st.error(f"An error occurred while fetching data. Please ensure the ticker '{ticker_input}' is correct and try again. System Error: {e}")
