import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Price Prediction | Quant Finance",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #0f3460;border-radius:12px;padding:16px;color:white;text-align:center;}
.metric-value {font-size:28px;font-weight:bold;color:#e94560;}
.metric-label {font-size:13px;color:#aaa;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📈 Quant Dashboard")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Technical Analysis",
    "🤖 ML Prediction",
    "🧠 LSTM-Style Prediction",
    "⚖️ Risk Analytics"
])

st.sidebar.markdown("---")
use_live = st.sidebar.checkbox("Use Live Data (yfinance)", value=False)
ticker = "NIFTY50"
period = "2y"
if use_live:
    ticker = st.sidebar.text_input("Ticker Symbol", value="RELIANCE.NS")
    period = st.sidebar.selectbox("Period", ["1y","2y","5y"], index=1)

@st.cache_data
def load_data(live=False, ticker="RELIANCE.NS", period="2y"):
    if live:
        try:
            import yfinance as yf
            df = yf.download(ticker, period=period, progress=False)
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df[['Date','Open','High','Low','Close','Volume']]
        except:
            st.warning("Live data failed. Using sample data.")
    return pd.read_csv("NIFTY50_data.csv", parse_dates=['Date'])

@st.cache_data
def compute_indicators(df):
    d = df.copy()
    d['SMA_20']  = d['Close'].rolling(20).mean()
    d['SMA_50']  = d['Close'].rolling(50).mean()
    d['SMA_200'] = d['Close'].rolling(200).mean()
    d['EMA_12']  = d['Close'].ewm(span=12).mean()
    d['EMA_26']  = d['Close'].ewm(span=26).mean()
    d['MACD']        = d['EMA_12'] - d['EMA_26']
    d['MACD_Signal'] = d['MACD'].ewm(span=9).mean()
    d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain/loss))
    d['BB_Mid']   = d['Close'].rolling(20).mean()
    bb_std        = d['Close'].rolling(20).std()
    d['BB_Upper'] = d['BB_Mid'] + 2*bb_std
    d['BB_Lower'] = d['BB_Mid'] - 2*bb_std
    d['ATR'] = np.maximum(d['High']-d['Low'],
               np.maximum(abs(d['High']-d['Close'].shift()),
                          abs(d['Low']-d['Close'].shift()))).rolling(14).mean()
    d['Daily_Return']  = d['Close'].pct_change()
    d['Log_Return']    = np.log(d['Close']/d['Close'].shift())
    d['Volatility_20'] = d['Daily_Return'].rolling(20).std()*np.sqrt(252)
    d['Volatility_60'] = d['Daily_Return'].rolling(60).std()*np.sqrt(252)
    d['Volume_MA20']   = d['Volume'].rolling(20).mean()
    d['OBV'] = (np.sign(d['Close'].diff())*d['Volume']).fillna(0).cumsum()
    d['Golden_Cross'] = (d['SMA_20']>d['SMA_50'])&(d['SMA_20'].shift()<=d['SMA_50'].shift())
    d['Death_Cross']  = (d['SMA_20']<d['SMA_50'])&(d['SMA_20'].shift()>=d['SMA_50'].shift())
    return d

df_raw = load_data(live=use_live, ticker=ticker, period=period)
df = compute_indicators(df_raw)

def dark_fig(figsize=(12,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for s in ax.spines.values(): s.set_edgecolor('#333')
    ax.grid(True, alpha=0.15, color='white')
    return fig, ax

def dark_fig2(figsize=(12,6), rows=2):
    fig, axes = plt.subplots(rows, 1, figsize=figsize)
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for s in ax.spines.values(): s.set_edgecolor('#333')
        ax.grid(True, alpha=0.15, color='white')
    return fig, axes

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📈 Stock Price Prediction")
    st.caption("Quantitative Finance | ML + LSTM-Style + Technical Analysis")
    st.markdown("---")

    latest = df.dropna().iloc[-1]
    prev   = df.dropna().iloc[-2]
    chg_pct = (latest['Close']-prev['Close'])/prev['Close']*100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Last Close",  f"₹{latest['Close']:,.2f}", f"{chg_pct:+.2f}%")
    c2.metric("52W High",    f"₹{df['Close'].tail(252).max():,.2f}")
    c3.metric("52W Low",     f"₹{df['Close'].tail(252).min():,.2f}")
    c4.metric("Avg Volume",  f"{int(df['Volume'].tail(20).mean()/1e6):.1f}M")
    c5.metric("Annual Vol",  f"{latest['Volatility_20']*100:.1f}%")

    st.markdown("---")
    recent = df.tail(500)
    fig, ax = dark_fig(figsize=(13,5))
    ax.plot(recent['Date'], recent['Close'],   color='#00d4ff', lw=1.5, label='Close')
    ax.plot(recent['Date'], recent['SMA_20'],  color='#ffa500', lw=1,   label='SMA 20',  ls='--')
    ax.plot(recent['Date'], recent['SMA_50'],  color='#ff6b6b', lw=1,   label='SMA 50',  ls='--')
    ax.plot(recent['Date'], recent['SMA_200'], color='#a29bfe', lw=1,   label='SMA 200', ls='--')
    gc = recent[recent['Golden_Cross']]
    dc = recent[recent['Death_Cross']]
    ax.scatter(gc['Date'], gc['Close'], color='gold',    s=80, zorder=5, label='Golden Cross ⬆')
    ax.scatter(dc['Date'], dc['Close'], color='#ff4757', s=80, zorder=5, label='Death Cross ⬇')
    ax.set_title("Price Chart with Moving Averages", color='white', fontsize=14)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Recent Data")
        show = df[['Date','Open','High','Low','Close','Volume']].tail(10).copy()
        show['Date'] = show['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(show.set_index('Date').style.format({
            'Open':'{:.2f}','High':'{:.2f}','Low':'{:.2f}',
            'Close':'{:.2f}','Volume':'{:,.0f}'}), use_container_width=True)
    with col2:
        st.subheader("📊 Return Distribution")
        fig, ax = dark_fig(figsize=(6,4))
        ret = df['Daily_Return'].dropna()
        ax.hist(ret, bins=60, color='#00d4ff', alpha=0.7, edgecolor='none')
        ax.axvline(ret.mean(), color='gold',    lw=2, ls='--', label=f'Mean: {ret.mean():.4f}')
        ax.axvline(ret.std(),  color='#ff4757', lw=2, ls='--', label=f'Std: {ret.std():.4f}')
        ax.set_title("Daily Return Distribution", color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        st.pyplot(fig); plt.close()

# ── TECHNICAL ANALYSIS ────────────────────────────────────────────────────────
elif page == "📊 Technical Analysis":
    st.title("📊 Technical Analysis Dashboard")
    st.markdown("---")
    recent = df.tail(300)

    fig, ax = dark_fig(figsize=(13,4))
    ax.plot(recent['Date'], recent['Close'],    color='#00d4ff', lw=1.5, label='Close')
    ax.plot(recent['Date'], recent['BB_Upper'], color='#ff6b6b', lw=1,   label='BB Upper', ls='--')
    ax.plot(recent['Date'], recent['BB_Mid'],   color='#ffa500', lw=1,   label='BB Mid',   ls='--')
    ax.plot(recent['Date'], recent['BB_Lower'], color='#6bcb77', lw=1,   label='BB Lower', ls='--')
    ax.fill_between(recent['Date'], recent['BB_Upper'], recent['BB_Lower'], alpha=0.07, color='#00d4ff')
    ax.set_title("Bollinger Bands (20,2)", color='white', fontsize=13)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        fig, axes = dark_fig2(figsize=(7,5), rows=2)
        axes[0].plot(recent['Date'], recent['Close'], color='#00d4ff', lw=1.2)
        axes[0].set_title("Price", color='white')
        axes[1].plot(recent['Date'], recent['RSI'], color='#ffa500', lw=1.2)
        axes[1].axhline(70, color='#ff4757', lw=1, ls='--', label='Overbought')
        axes[1].axhline(30, color='#6bcb77', lw=1, ls='--', label='Oversold')
        axes[1].fill_between(recent['Date'], recent['RSI'], 70, where=recent['RSI']>=70, alpha=0.2, color='#ff4757')
        axes[1].fill_between(recent['Date'], recent['RSI'], 30, where=recent['RSI']<=30, alpha=0.2, color='#6bcb77')
        axes[1].set_title("RSI (14)", color='white')
        axes[1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        axes[1].set_ylim(0,100)
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, axes = dark_fig2(figsize=(7,5), rows=2)
        axes[0].plot(recent['Date'], recent['Close'], color='#00d4ff', lw=1.2)
        axes[0].set_title("Price", color='white')
        axes[1].plot(recent['Date'], recent['MACD'],        color='#00d4ff', lw=1.2, label='MACD')
        axes[1].plot(recent['Date'], recent['MACD_Signal'], color='#ff6b6b', lw=1.2, label='Signal')
        axes[1].bar(recent['Date'], recent['MACD_Hist'].clip(lower=0), color='#6bcb77', alpha=0.6, width=1)
        axes[1].bar(recent['Date'], recent['MACD_Hist'].clip(upper=0), color='#ff4757', alpha=0.6, width=1)
        axes[1].axhline(0, color='white', lw=0.5, ls='--')
        axes[1].set_title("MACD (12,26,9)", color='white')
        axes[1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("📉 Realized Volatility")
    fig, ax = dark_fig(figsize=(13,3))
    ax.plot(recent['Date'], recent['Volatility_20']*100, color='#ffa500', lw=1.2, label='20-day')
    ax.plot(recent['Date'], recent['Volatility_60']*100, color='#a29bfe', lw=1.2, label='60-day')
    ax.fill_between(recent['Date'], recent['Volatility_20']*100, alpha=0.15, color='#ffa500')
    ax.set_ylabel("Volatility (%)", color='white')
    ax.set_title("Annualized Realized Volatility", color='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    st.pyplot(fig); plt.close()

    st.subheader("🚦 Current Signals")
    latest = df.dropna().iloc[-1]
    sig_df = pd.DataFrame({
        'Indicator': ['RSI (14)', 'MACD (12,26,9)', 'SMA Cross (20/50)', 'Bollinger Bands'],
        'Value': [f"{latest['RSI']:.1f}", f"{latest['MACD']:.1f}",
                  f"{latest['SMA_20']:.0f} / {latest['SMA_50']:.0f}",
                  f"Close: {latest['Close']:.0f}"],
        'Signal': [
            "🔴 Overbought" if latest['RSI']>70 else ("🟢 Oversold" if latest['RSI']<30 else "🟡 Neutral"),
            "🟢 Bullish" if latest['MACD']>latest['MACD_Signal'] else "🔴 Bearish",
            "🟢 Bullish" if latest['SMA_20']>latest['SMA_50'] else "🔴 Bearish",
            "🔴 Overbought" if latest['Close']>latest['BB_Upper'] else (
             "🟢 Oversold" if latest['Close']<latest['BB_Lower'] else "🟡 Inside Band")
        ]
    })
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ── ML PREDICTION ─────────────────────────────────────────────────────────────
elif page == "🤖 ML Prediction":
    st.title("🤖 ML-Based Price Prediction")
    st.caption("Random Forest & Gradient Boosting Ensemble")
    st.markdown("---")

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    @st.cache_resource
    def train_ml(df):
        d = df.copy().dropna()
        for lag in [1,2,3,5,10,20]:
            d[f'Close_lag_{lag}']  = d['Close'].shift(lag)
            d[f'Return_lag_{lag}'] = d['Daily_Return'].shift(lag)
        d['momentum_5']       = d['Close']/d['Close'].shift(5)-1
        d['momentum_20']      = d['Close']/d['Close'].shift(20)-1
        d['HL_ratio']         = d['High']/d['Low']
        d['CO_ratio']         = d['Close']/d['Open']
        d['Volume_change']    = d['Volume'].pct_change()
        d['Target'] = d['Close'].shift(-1)
        d = d.dropna()

        feats = [c for c in d.columns if c not in
                 ['Date','Open','High','Low','Close','Volume','Target',
                  'Golden_Cross','Death_Cross']]
        X, y = d[feats], d['Target']
        split = int(len(X)*0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]
        dates_te   = d['Date'].iloc[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        gb.fit(X_tr_sc, y_tr)

        rf_p  = rf.predict(X_te_sc)
        gb_p  = gb.predict(X_te_sc)
        ens_p = 0.5*rf_p + 0.5*gb_p

        return rf, gb, sc, feats, rf_p, gb_p, ens_p, y_te.values, dates_te.values, X_te_sc

    with st.spinner("Training models..."):
        rf, gb, sc, feats, rf_p, gb_p, ens_p, y_te, dates_te, X_te_sc = train_ml(df)

    def mets(yt, yp, name):
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        mape = np.mean(np.abs((yt-yp)/yt))*100
        return {'Model':name,'RMSE':f'{rmse:.2f}','R²':f'{r2:.4f}','MAPE':f'{mape:.2f}%'}

    st.subheader("📊 Model Performance")
    st.dataframe(pd.DataFrame([
        mets(y_te, rf_p,  'Random Forest'),
        mets(y_te, gb_p,  'Gradient Boosting'),
        mets(y_te, ens_p, 'Ensemble (RF+GB)')
    ]), use_container_width=True, hide_index=True)

    st.subheader("📈 Actual vs Predicted")
    fig, ax = dark_fig(figsize=(13,5))
    ax.plot(dates_te, y_te,   color='#00d4ff', lw=1.5, label='Actual')
    ax.plot(dates_te, ens_p,  color='#ffa500', lw=1.2, label='Ensemble', ls='--')
    ax.plot(dates_te, rf_p,   color='#6bcb77', lw=0.8, label='RF',       alpha=0.6)
    ax.plot(dates_te, gb_p,   color='#ff6b6b', lw=0.8, label='GB',       alpha=0.6)
    ax.set_title("Actual vs Predicted Next-Day Close", color='white', fontsize=13)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    st.pyplot(fig); plt.close()

    st.subheader("🔍 Top 15 Feature Importances")
    fi = pd.Series(rf.feature_importances_, index=feats).nlargest(15)
    fig, ax = dark_fig(figsize=(10,5))
    ax.barh(fi.index[::-1], fi.values[::-1], color='#00d4ff', alpha=0.8)
    ax.set_title("Feature Importances (Random Forest)", color='white')
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🔮 Next Trading Day Forecast")
    d2 = df.copy().dropna()
    for lag in [1,2,3,5,10,20]:
        d2[f'Close_lag_{lag}']  = d2['Close'].shift(lag)
        d2[f'Return_lag_{lag}'] = d2['Daily_Return'].shift(lag)
    d2['momentum_5']    = d2['Close']/d2['Close'].shift(5)-1
    d2['momentum_20']   = d2['Close']/d2['Close'].shift(20)-1
    d2['HL_ratio']      = d2['High']/d2['Low']
    d2['CO_ratio']      = d2['Close']/d2['Open']
    d2['Volume_change'] = d2['Volume'].pct_change()
    d2 = d2.dropna()

    row    = d2[feats].iloc[[-1]]
    row_sc = sc.transform(row)
    rf_nx  = rf.predict(row_sc)[0]
    gb_nx  = gb.predict(row_sc)[0]
    en_nx  = 0.5*rf_nx + 0.5*gb_nx
    cur    = df['Close'].iloc[-1]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Current Close",     f"₹{cur:,.2f}")
    c2.metric("RF Forecast",       f"₹{rf_nx:,.2f}", f"{(rf_nx-cur)/cur*100:+.2f}%")
    c3.metric("GB Forecast",       f"₹{gb_nx:,.2f}", f"{(gb_nx-cur)/cur*100:+.2f}%")
    c4.metric("Ensemble Forecast", f"₹{en_nx:,.2f}", f"{(en_nx-cur)/cur*100:+.2f}%")

# ── LSTM-STYLE ────────────────────────────────────────────────────────────────
elif page == "🧠 LSTM-Style Prediction":
    st.title("🧠 LSTM-Style Sequential Prediction")
    st.caption("Sliding window sequence model — same concept as LSTM, no TensorFlow needed")
    st.markdown("---")

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import GradientBoostingRegressor

    LOOKBACK = st.slider("Lookback Window (days)", 10, 90, 30)
    MODEL    = st.selectbox("Sequence Model", ["Ridge Regression", "Gradient Boosting"])

    @st.cache_resource
    def train_seq(lookback, model_name):
        close  = df['Close'].values.astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close.reshape(-1,1)).flatten()

        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        split   = int(len(X)*0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        dates_te   = df['Date'].values[lookback+split:]

        if model_name == "Ridge Regression":
            m = Ridge(alpha=0.1)
        else:
            m = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)
        m.fit(X_tr, y_tr)
        pred_sc = m.predict(X_te)
        pred    = scaler.inverse_transform(pred_sc.reshape(-1,1)).flatten()
        actual  = scaler.inverse_transform(y_te.reshape(-1,1)).flatten()

        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2   = r2_score(actual, pred)
        mape = np.mean(np.abs((actual-pred)/actual))*100

        # next day
        last_seq = scaled[-lookback:]
        next_sc  = m.predict(last_seq.reshape(1,-1))[0]
        next_p   = scaler.inverse_transform([[next_sc]])[0][0]
        return pred, actual, dates_te, rmse, r2, mape, next_p

    with st.spinner("Training sequence model..."):
        pred, actual, dates_te, rmse, r2, mape, next_p = train_seq(LOOKBACK, MODEL)

    c1,c2,c3 = st.columns(3)
    c1.metric("RMSE", f"₹{rmse:.2f}")
    c2.metric("R²",   f"{r2:.4f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    fig, ax = dark_fig(figsize=(13,5))
    ax.plot(dates_te, actual, color='#00d4ff', lw=1.5, label='Actual')
    ax.plot(dates_te, pred,   color='#ffa500', lw=1.5, label=f'{MODEL} Predicted', ls='--')
    ax.fill_between(dates_te, actual, pred, alpha=0.1, color='#ffa500')
    ax.set_title(f"Sequential Prediction — Lookback={LOOKBACK} days | {MODEL}", color='white', fontsize=13)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🔮 Next Day Forecast")
    cur = df['Close'].iloc[-1]
    c1,c2,c3 = st.columns(3)
    c1.metric("Current Price", f"₹{cur:,.2f}")
    c2.metric("Next Day Forecast", f"₹{next_p:,.2f}", f"{(next_p-cur)/cur*100:+.2f}%")
    c3.metric("Direction", "📈 UP" if next_p>cur else "📉 DOWN")

    st.info("💡 This page uses a sliding-window sequence approach — the same core idea as LSTM. "
            "The lookback window feeds past prices as features, identical to how LSTM processes time sequences.")

# ── RISK ANALYTICS ────────────────────────────────────────────────────────────
elif page == "⚖️ Risk Analytics":
    st.title("⚖️ Risk Analytics")
    st.caption("VaR | CVaR | Sharpe | Sortino | Drawdown | Monte Carlo")
    st.markdown("---")

    returns = df['Daily_Return'].dropna()
    conf    = st.slider("VaR Confidence Level", 90, 99, 95)
    VaR     = np.percentile(returns, 100-conf)
    CVaR    = returns[returns <= VaR].mean()
    sharpe  = (returns.mean()*252)/(returns.std()*np.sqrt(252))
    sortino = (returns.mean()*252)/(returns[returns<0].std()*np.sqrt(252))
    cum_ret = (1+returns).cumprod()
    drawdown= (cum_ret - cum_ret.cummax())/cum_ret.cummax()
    max_dd  = drawdown.min()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(f"VaR ({conf}%)",  f"{VaR*100:.2f}%")
    c2.metric("CVaR (ES)",       f"{CVaR*100:.2f}%")
    c3.metric("Sharpe Ratio",    f"{sharpe:.2f}")
    c4.metric("Sortino Ratio",   f"{sortino:.2f}")
    c5.metric("Max Drawdown",    f"{max_dd*100:.2f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = dark_fig(figsize=(7,4))
        ax.hist(returns, bins=80, color='#00d4ff', alpha=0.6, edgecolor='none')
        ax.axvline(VaR,  color='#ff4757', lw=2, ls='--', label=f'VaR: {VaR*100:.2f}%')
        ax.axvline(CVaR, color='#ffa500', lw=2, ls='--', label=f'CVaR: {CVaR*100:.2f}%')
        ax.set_title("Return Distribution with VaR", color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = dark_fig(figsize=(7,4))
        dd_dates = df['Date'].iloc[1:len(drawdown)+1]
        ax.fill_between(dd_dates, drawdown.values*100, 0, color='#ff4757', alpha=0.5)
        ax.plot(dd_dates, drawdown.values*100, color='#ff4757', lw=1)
        ax.set_title("Drawdown (%)", color='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=30)
        st.pyplot(fig); plt.close()

    st.subheader("🎲 Monte Carlo Price Simulation (252 days)")
    N_SIM = st.slider("Simulations", 100, 1000, 500, step=100)
    mu, sigma = returns.mean(), returns.std()
    S0 = df['Close'].iloc[-1]
    np.random.seed(42)
    sims = S0 * np.cumprod(1 + np.random.normal(mu, sigma, (252, N_SIM)), axis=0)

    fig, ax = dark_fig(figsize=(13,5))
    for i in range(min(200, N_SIM)):
        ax.plot(sims[:,i], color='#00d4ff', alpha=0.04, lw=0.5)
    ax.plot(np.percentile(sims,  5, axis=1), color='#ff4757', lw=2, ls='--', label='5th %ile (Bear)')
    ax.plot(np.percentile(sims, 50, axis=1), color='#ffa500', lw=2, ls='--', label='Median')
    ax.plot(np.percentile(sims, 95, axis=1), color='#6bcb77', lw=2, ls='--', label='95th %ile (Bull)')
    ax.axhline(S0, color='white', lw=1, ls=':', label=f'Current ₹{S0:,.0f}')
    ax.set_title(f"Monte Carlo — {N_SIM} paths × 252 trading days", color='white', fontsize=13)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.set_xlabel("Trading Days", color='white')
    ax.set_ylabel("Price (₹)",    color='white')
    st.pyplot(fig); plt.close()

    p5  = np.percentile(sims[-1],  5)
    p50 = np.percentile(sims[-1], 50)
    p95 = np.percentile(sims[-1], 95)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Current",       f"₹{S0:,.2f}")
    c2.metric("Bear (5%)",     f"₹{p5:,.2f}",  f"{(p5-S0)/S0*100:+.1f}%")
    c3.metric("Base (50%)",    f"₹{p50:,.2f}", f"{(p50-S0)/S0*100:+.1f}%")
    c4.metric("Bull (95%)",    f"₹{p95:,.2f}", f"{(p95-S0)/S0*100:+.1f}%")
