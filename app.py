import streamlit as st
import pandas as pd
import ta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# En express-byggd Dashboard för lite aktiedata.
st.set_page_config(page_title="Aktieinfo",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

# Logging för felsökning
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lite funktioner först
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data):
    # Kontrollera att 'Close' kolumnen finns och är numerisk
    if 'Close' not in data.columns:
        raise ValueError("'Close' column is missing from data")
    if not pd.api.types.is_numeric_dtype(data['Close']):
        raise ValueError("'Close' column must be numeric")
    # Hantera NaN-värden
    if data['Close'].isnull().any():
        data['Close'] = data['Close'].fillna(method='ffill')

    # Logga de första raderna i 'Close' kolumnen
    logger.info(data['Close'].head())

    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data

# Nu kör vi lite dashboard layout
st.title('Aktieinfo | Dashboard')

# En liten sidebar
st.sidebar.header('Graf-parametrar')
ticker = st.sidebar.text_input('Ticker', 'NVDA')
time_period = st.sidebar.selectbox('Tidsperiod', ['1d', '1wk', '1mo', '1y', 'max'], index=3)
chart_type = st.sidebar.selectbox('Graf-typ', ['Candlestick', 'Linje'])
indicators = st.sidebar.multiselect('Tekniska indikatorer', ['SMA 20', 'EMA 20'])

interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# Huvuddelen
if st.sidebar.button('Uppdatera'):
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    if data is not None:
        data = process_data(data)
        if data is not None:
            data = add_technical_indicators(data)
            if data is not None:
                last_close, change, pct_change, high, low, volume = calculate_metrics(data)

                st.metric(label=f'{ticker} Last Price', value=f'{last_close:.2f}', delta=f'{change:.2f} ({pct_change:.2f}%)')

                col1, col2, col3 = st.columns(3)
                col1.metric("High", f"{high:.2f}")
                col2.metric("Low", f"{low:.2f}")
                col3.metric("Volume", f"{volume:.2f}")

                fig = go.Figure()
                if chart_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(x=data['Datetime'],
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close']))
                else:
                    fig = px.line(data, x='Datetime', y='Close')

                # Indikatorer till graf
                for indicator in indicators:
                    if indicator == 'SMA 20':
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
                    elif indicator == 'EMA 20':
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))

                # Graf
                fig.update_layout(title=f'{ticker} {time_period.upper()} Graf',
                                  xaxis_title='Tid',
                                  yaxis_title='Pris',
                                  height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                "---"

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('DF: Historisk data')
                    st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])
                with col2:
                    st.subheader("DF: Tekniska indikatorer")
                    st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']])

        "---"

# Sidebar-priser
# Liten skön sektion med aktiepriser för utvalda aktier...!
st.sidebar.header("Aktiepriser")
aktier = ['AAPL', 'TSLA', 'NVDA', 'META']
for aktie in aktier:
    aktie_data = fetch_stock_data(aktie, '1d', '1m')
    if not aktie_data.empty:
        aktie_data = process_data(aktie_data)
        last_price = aktie_data['Close'].iloc[-1]
        change = last_price - aktie_data['Open'].iloc[0]
        pct_change = (change / aktie_data['Open'].iloc[0]) * 100
        st.sidebar.metric(f"{aktie}", f"{last_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")

st.sidebar.subheader('Om')
st.sidebar.info('Denna dashboard visar aktiedata och tekniska indikatorer för olika tidsperioder')

st.sidebar.divider()
st.sidebar.caption("Gjord av Gabriel 🏖️")
