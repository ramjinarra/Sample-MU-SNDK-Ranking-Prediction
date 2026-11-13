"""
=============================================================================
DOCUMENTATION & THOUGHT PROCESS
=============================================================================
1. Data Structures Inspected:
   - yfinance DataFrames: Evaluated Daily (1D) and Intraday (5m) timeframes for target assets (e.g., SNDK, MULL)[cite: 1] and broader indices (SOXX, NVDA) to capture momentum, volume, and OHLC data.
   - alpaca-py Data Structures: Interrogated the `OptionHistoricalDataClient` dictionary response to access `OptionSnapshot` objects, which contain the OCC symbol and latest quote (bid/ask) data[cite: 1].

2. Bullish Momentum Score (0-100) Translation[cite: 1]:
   The script allocates a maximum of 10 points to each of the 10 technical factors outlined in Scoring_Logic.txt[cite: 1]. 
   - Factor 1 (Multi-Day Streak): +10 if consecutive daily closes accumulate positively.
   - Factor 2 (Prior-Day CLV): +10 if the previous day's CLV >= 0.5.
   - Factor 3 (Korean Memory Lead): +10 if the 5-day correlation > 0.6 and Proxy Close > Proxy 5-SMA.
   - Factor 4 (Sector Sync): +10 if Target, SOXX, and NVDA 5m 9-EMA slopes are all positive.
   - Factor 7 (PDH/PDL Structure): +10 if Current Price > Previous Day High.
   - Factor 9 (VWAP Reclaim): +10 if price trades below VWAP for 6 prior periods, crosses above, and crossing volume > 20-SMA.
   - Factor 12 (Trend Alignment): +10 if Current Price > VWAP and 9 EMA > 21 EMA.
   - Factor 13 (30-Min ORB): +10 if Current Price > 09:30-10:00 AM High.
   - Factor 14 (RSI & VWAP SD): +10 if 5m RSI > 50 and Price sits between the VWAP and the Lower Band.
   - Factor 15 (Seasonality): +10 if historical green-close win rate for the current day-of-week > 50%.

   Thresholds:
   - >= 80: Strong Bullish
   - 60 - 70: Bullish
   - 40 - 50: Neutral
   - <= 30: Bearish

3. Calculations Unsure About & Solutions[cite: 1]:
   - VWAP Standard Deviation Bands: The standard `pandas_ta` VWAP indicator does not natively emit bands. I resolved this by calculating a 14-period rolling standard deviation of the closing price and subtracting `2 * SD` from the baseline VWAP to form the Lower Band.
   - Options Option Chain Iteration: Parsing expirations and strikes directly from the standard OCC string format via regex was necessary to securely extract the ATM and the nearest OTM calls.
=============================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import re
from datetime import datetime

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

# -----------------------------------------------------------------------------
# 1. UI SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Options Terminal: Directional Predictor", layout="wide")
st.title("Options Terminal: Directional Predictor & Spread Ranker")

st.sidebar.header("Data Configuration")
ticker = st.sidebar.text_input("Target Asset (e.g., SNDK, MULL, SPY)", value="SPY").upper()
proxy_ticker = st.sidebar.text_input("Korean Memory Proxy (e.g., EWY)", value="EWY").upper()

st.sidebar.header("Alpaca Credentials")
alpaca_key = st.sidebar.text_input("Alpaca API Key", type="password")
alpaca_secret = st.sidebar.text_input("Alpaca Secret Key", type="password")

# -----------------------------------------------------------------------------
# 2. DATA FETCHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_market_data(target, proxy):
    def flt(df):
        # Flatten yfinance MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            return df.droplevel(1, axis=1)
        return df

    dt = flt(yf.download(target, period="6mo", interval="1d", progress=False))
    dp = flt(yf.download(proxy, period="1mo", interval="1d", progress=False))
    
    it = flt(yf.download(target, period="5d", interval="5m", progress=False))
    isoxx = flt(yf.download("SOXX", period="5d", interval="5m", progress=False))
    invda = flt(yf.download("NVDA", period="5d", interval="5m", progress=False))
    
    return dt, dp, it, isoxx, invda

# -----------------------------------------------------------------------------
# 3. INDICATOR & SCORING CALCULATIONS
# -----------------------------------------------------------------------------
def calc_streak(df):
    if len(df) < 2: return 0
    diff = df['Close'].diff().dropna()
    streak = 0
    current_dir = 1 if diff.iloc[-1] > 0 else -1 if diff.iloc[-1] < 0 else 0
    for val in diff.iloc[::-1]:
        if val > 0 and current_dir == 1: streak += 1
        elif val < 0 and current_dir == -1: streak -= 1
        else: break
    return 10 if streak > 0 else 0

def calc_clv(df):
    if len(df) < 2: return 0
    prev = df.iloc[-2]
    rng = prev['High'] - prev['Low']
    if rng == 0: return 0
    clv = ((prev['Close'] - prev['Low']) - (prev['High'] - prev['Close'])) / rng
    return 10 if clv >= 0.5 else 0

def calc_kor_memory(df_t, df_p):
    if len(df_t) < 5 or len(df_p) < 5: return 0
    ret_t = df_t['Close'].pct_change()
    ret_p = df_p['Close'].pct_change()
    corr = ret_t.rolling(5).corr(ret_p).dropna()
    if len(corr) == 0: return 0
    curr_corr = corr.iloc[-1]
    sma_5 = df_p['Close'].rolling(5).mean().iloc[-1]
    if curr_corr > 0.6 and df_p['Close'].iloc[-1] > sma_5:
        return 10
    return 0

def calc_sync(df_t, df_s, df_n):
    def get_slope(df):
        if len(df) < 10: return 0
        ema = ta.ema(df['Close'], length=9)
        if ema is None or len(ema.dropna()) < 2: return 0
        ema = ema.dropna()
        return ema.iloc[-1] - ema.iloc[-2]
    return 10 if get_slope(df_t) > 0 and get_slope(df_s) > 0 and get_slope(df_n) > 0 else 0

def calc_pdh(df_daily, df_5m):
    if len(df_daily) < 2 or len(df_5m) == 0: return 0
    return 10 if df_5m['Close'].iloc[-1] > df_daily.iloc[-2]['High'] else 0

def calc_vwap_reclaim(df_5m):
    if len(df_5m) < 20: return 0
    df = df_5m.copy()
    df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
    df = df.dropna(subset=['VWAP', 'Vol_SMA'])
    if len(df) < 8: return 0
    
    curr, prev = df.iloc[-1], df.iloc[-2]
    cross_above = (prev['Close'] < prev['VWAP']) and (curr['Close'] > curr['VWAP'])
    
    if cross_above and (curr['Volume'] > curr['Vol_SMA']):
        prior_6 = df.iloc[-7:-1] 
        if (prior_6['Close'] < prior_6['VWAP']).all():
            return 10
    return 0

def calc_trend_alignment(df_5m):
    if len(df_5m) < 22: return 0
    df = df_5m.copy()
    df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['EMA9'] = ta.ema(df['Close'], length=9)
    df['EMA21'] = ta.ema(df['Close'], length=21)
    curr = df.iloc[-1]
    
    if curr['Close'] > curr['VWAP'] and curr['EMA9'] > curr['EMA21'] and curr['Close'] > curr['EMA9']:
        return 10
    return 0

def calc_orb(df_5m):
    if len(df_5m) == 0: return 0
    df_today = df_5m[df_5m.index.date == df_5m.index[-1].date()]
    try:
        orb_data = df_today.between_time('09:30', '10:00')
        if orb_data.empty: return 0
        return 10 if df_5m['Close'].iloc[-1] > orb_data['High'].max() else 0
    except:
        return 0

def calc_rsi_vwap(df_5m):
    if len(df_5m) < 15: return 0
    df = df_5m.copy()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['LBand'] = df['VWAP'] - 2 * df['Close'].rolling(14).std()
    
    curr = df.iloc[-1]
    if curr['RSI'] > 50 and (curr['LBand'] < curr['Close'] < curr['VWAP']):
        return 10
    return 0

def calc_seasonality(df_daily):
    if len(df_daily) == 0: return 0
    curr_day = df_daily.index[-1].dayofweek
    df_day = df_daily.iloc[-90:]
    df_day = df_day[df_day.index.dayofweek == curr_day]
    if len(df_day) == 0: return 0
    
    win_rate = (df_day['Close'] > df_day['Open']).sum() / len(df_day)
    return 10 if win_rate > 0.5 else 0

def aggregate_score(dt, dp, it, isoxx, invda):
    score = 0
    score += calc_streak(dt)
    score += calc_clv(dt)
    score += calc_kor_memory(dt, dp)
    score += calc_sync(it, isoxx, invda)
    score += calc_pdh(dt, it)
    score += calc_vwap_reclaim(it)
    score += calc_trend_alignment(it)
    score += calc_orb(it)
    score += calc_rsi_vwap(it)
    score += calc_seasonality(dt)
    return score

def decode_prediction(score):
    if score >= 80: return "Strong Bullish"
    if score >= 60: return "Bullish"
    if score >= 40: return "Neutral"
    if score >= 20: return "Bearish"
    return "Strong Bearish"

# -----------------------------------------------------------------------------
# 4. ALPACA OPTIONS LOGIC
# -----------------------------------------------------------------------------
def fetch_alpaca_options(ticker, current_price, score):
    client = OptionHistoricalDataClient(alpaca_key, alpaca_secret)
    req = OptionChainRequest(underlying_symbol=ticker)
    chain = client.get_option_chain(req)
    
    calls = []
    for occ, snapshot in chain.items():
        match = re.match(r'^([a-zA-Z]+)(\d{6})([CP])(\d{8})$', occ.replace(' ', ''))
        if match and match.group(3) == 'C':
            bid = snapshot.latest_quote.bid_price if snapshot.latest_quote else 0
            ask = snapshot.latest_quote.ask_price if snapshot.latest_quote else 0
            if bid > 0 and ask > 0:
                calls.append({
                    'Expiration': match.group(2),
                    'Strike': int(match.group(4)) / 1000.0,
                    'Bid': bid,
                    'Ask': ask,
                    'Liquidity_Spread': ask - bid
                })
                
    df = pd.DataFrame(calls)
    spreads, naked = [], []
    
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    for exp, group in df.groupby('Expiration'):
        group = group.sort_values('Strike').reset_index(drop=True)
        atm_idx = (group['Strike'] - current_price).abs().idxmin()
        atm = group.loc[atm_idx]
        
        naked.append({
            'Directional_Score': score,
            'Expiration': exp,
            'Strike': atm['Strike'],
            'Bid': atm['Bid'],
            'Ask': atm['Ask'],
            'Bid-Ask Liquidity': atm['Liquidity_Spread']
        })
        
        otm_group = group[group['Strike'] > atm['Strike']]
        if not otm_group.empty:
            otm = otm_group.iloc[0]
            cost = atm['Ask'] - otm['Bid']
            width = otm['Strike'] - atm['Strike']
            spreads.append({
                'Directional_Score': score,
                'Expiration': exp,
                'Long_Strike_ATM': atm['Strike'],
                'Short_Strike_OTM': otm['Strike'],
                'Spread_Width': width,
                'Debit_Cost': cost,
                'Max_Profit': width - cost,
                'Bid-Ask Liquidity': atm['Liquidity_Spread'] + otm['Liquidity_Spread']
            })
            
    df_sp = pd.DataFrame(spreads).sort_values(['Debit_Cost', 'Bid-Ask Liquidity']) if spreads else pd.DataFrame()
    df_nk = pd.DataFrame(naked).sort_values(['Bid-Ask Liquidity']) if naked else pd.DataFrame()
    
    return df_sp, df_nk

# -----------------------------------------------------------------------------
# 5. STREAMLIT EXECUTION
# -----------------------------------------------------------------------------
if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching Data and Computing Momentum..."):
        dt, dp, it, isoxx, invda = fetch_market_data(ticker, proxy_ticker)
        
        if len(it) == 0:
            st.error("No 5m data found for target ticker. Verify market hours and symbol.")
            st.stop()
            
        current_price = it['Close'].iloc[-1]
        score = aggregate_score(dt, dp, it, isoxx, invda)
        prediction = decode_prediction(score)
        
        st.metric("Aggregated Bullish Momentum Score", f"{score}/100")
        
        if score >= 60:
            st.success(f"Directional Prediction: {prediction}")
            if not alpaca_key or not alpaca_secret:
                st.warning("Prediction is Bullish. Provide Alpaca API keys to rank Options spreads.")
            else:
                with st.spinner("Fetching and Ranking Options Chain..."):
                    df_sp, df_nk = fetch_alpaca_options(ticker, current_price, score)
                    
                    st.subheader("Top Ranked Call Debit Spreads")
                    st.dataframe(df_sp, use_container_width=True) if not df_sp.empty else st.info("No viable spreads found.")
                    
                    st.subheader("Top Ranked Naked Calls (ATM)")
                    st.dataframe(df_nk, use_container_width=True) if not df_nk.empty else st.info("No viable naked calls found.")
        else:
            if score >= 40:
                st.warning(f"Directional Prediction: {prediction}")
            else:
                st.error(f"Directional Prediction: {prediction}")
            st.info("Prediction is not strongly bullish. Options spread generation aborted to preserve capital.")
