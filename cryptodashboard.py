# crypto_dashboard_final.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import random
import warnings

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title=" 📊 Crypto Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Neon-dark CSS (clean)
# ---------------------------
NEON_BG = "#050814"
NEON_PANEL = "#0b1220"
NEON_ACCENT = "#00f0ff"
NEON_ACCENT2 = "#7CFF6A"
NEON_TEXT = "#E6F7FA"

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {NEON_BG};
        --panel: {NEON_PANEL};
        --accent: {NEON_ACCENT};
        --accent2: {NEON_ACCENT2};
        --text: {NEON_TEXT};
    }}
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
    }}
    .block-container {{
        padding-top: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }}
    .stSidebar {{
        background: radial-gradient(circle at top left,#101827,#050814);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}
    .neon-card {{
        background: linear-gradient(180deg, rgba(13,23,40,0.95), rgba(6,12,24,0.9));
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.75);
        border: 1px solid rgba(0,255,255,0.16);
        margin-bottom: 12px;
    }}
    h1,h2,h3 {{
        color: var(--text) !important;
        margin-bottom: 0.3rem;
    }}
    .neon-title {{
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0.6px;
        margin-bottom: 4px;
        font-size: 1rem;
    }}
    .small-muted {{
        color: rgba(230,247,250,0.7);
        font-size: 12px;
    }}
    .metric-box {{
        background: linear-gradient(135deg, rgba(0,240,255,0.10), rgba(124,255,106,0.05));
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 6px;
    }}
    .metric-label {{
        font-size: 11px;
        color: rgba(230,247,250,0.75);
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 700;
        color: {NEON_ACCENT};
    }}
    .jsplotly-plot .plot-container {{
        background: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly template
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        template="plotly_dark",
        paper_bgcolor=NEON_BG,
        plot_bgcolor=NEON_PANEL,
        font=dict(color=NEON_TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0.2)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

def apply_plotly_layout(fig, title=None):
    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            font=dict(color=NEON_ACCENT, size=18),
        ),
        paper_bgcolor=PLOTLY_TEMPLATE["layout"].paper_bgcolor,
        plot_bgcolor=PLOTLY_TEMPLATE["layout"].plot_bgcolor,
        font=PLOTLY_TEMPLATE["layout"].font,
        legend=PLOTLY_TEMPLATE["layout"].legend,
        margin=PLOTLY_TEMPLATE["layout"].margin,
        colorway=[NEON_ACCENT, NEON_ACCENT2, "#6EC0FF", "#FFA75C", "#C792EA"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(ttl=600)
def fetch_coin_data(coin_id, days=60):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(
            url,
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame(columns=["date", "price"])
        df = pd.DataFrame(prices, columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "price"])

@st.cache_data(ttl=120)
def get_live_prices(coins):
    try:
        if not coins:
            return {}
        ids = ",".join(coins)
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ids,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        out = {}
        for c in coins:
            try:
                out[c] = float(data.get(c, {}).get("usd", 0.0))
            except Exception:
                out[c] = 0.0
        return out
    except Exception:
        return {c: 0.0 for c in coins}

def SMA(series, window=5):
    return series.rolling(window).mean()

def EMA(series, window=5):
    return series.ewm(span=window, adjust=False).mean()

def RSI(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean().replace(0, np.nan)
    rsi = 100 - 100 / (1 + (ma_up / ma_down))
    return rsi.fillna(0)

@st.cache_data(ttl=1800)
def forecast_arima(series, days=7):
    if series is None or len(series.dropna()) < 10:
        start = pd.Timestamp.today()
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )
    try:
        model = ARIMA(series, order=(2, 1, 2)).fit()
        fc = model.forecast(days)
        start = series.index[-1] + pd.Timedelta(days=1)
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": fc.values}
        )
    except Exception:
        start = (
            series.index[-1] + pd.Timedelta(days=1)
            if hasattr(series, "index") and len(series) > 0
            else pd.Timestamp.today()
        )
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )

@st.cache_data(ttl=1800)
def forecast_sarimax(series, days=7):
    if series is None or len(series.dropna()) < 10:
        start = pd.Timestamp.today()
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )
    try:
        model = SARIMAX(
            series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)
        ).fit(disp=False)
        fc = model.forecast(days)
        start = series.index[-1] + pd.Timedelta(days=1)
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": fc.values}
        )
    except Exception:
        start = (
            series.index[-1] + pd.Timedelta(days=1)
            if hasattr(series, "index") and len(series) > 0
            else pd.Timestamp.today()
        )
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )

@st.cache_data(ttl=1800)
def forecast_lstm(series, days=7, epochs=5):
    if not TF_AVAILABLE:
        start = (
            series.index[-1] + pd.Timedelta(days=1)
            if hasattr(series, "index") and len(series) > 0
            else pd.Timestamp.today()
        )
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )

    if series is None or len(series.dropna()) < 10:
        start = (
            series.index[-1] + pd.Timedelta(days=1)
            if hasattr(series, "index") and len(series) > 0
            else pd.Timestamp.today()
        )
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )

    try:
        scaler = MinMaxScaler()
        vals = series.values.reshape(-1, 1)
        scaled = scaler.fit_transform(vals)

        lookback = 3
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback : i, 0])
            y.append(scaled[i, 0])
        X = np.array(X)
        y = np.array(y)
        if len(X) == 0:
            start = series.index[-1] + pd.Timedelta(days=1)
            return pd.DataFrame(
                {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
            )

        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)

        inp = scaled[-lookback:].reshape(1, lookback, 1)
        preds_scaled = []
        for _ in range(days):
            p = model.predict(inp, verbose=0)[0, 0]
            preds_scaled.append(p)
            next_inp = np.concatenate(
                [inp[:, 1:, :], np.array(p).reshape(1, 1, 1)], axis=1
            )
            inp = next_inp

        preds = scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        )[:, 0]
        start = series.index[-1] + pd.Timedelta(days=1)
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": preds}
        )
    except Exception:
        start = (
            series.index[-1] + pd.Timedelta(days=1)
            if hasattr(series, "index") and len(series) > 0
            else pd.Timestamp.today()
        )
        return pd.DataFrame(
            {"date": pd.date_range(start, periods=days), "price": [np.nan] * days}
        )

def portfolio_value(holdings, prices):
    return sum([holdings.get(c, 0) * float(prices.get(c, 0)) for c in holdings])

# ---------------------------
# Small UI helpers
# ---------------------------
def section(title, subtitle=None):
    st.markdown(
        f"<div class='neon-card'><div class='neon-title'>{title}</div>"
        f"<div class='small-muted'>{subtitle or ''}</div></div>",
        unsafe_allow_html=True,
    )

def metric_box(label, value):
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------------------
# Sidebar header + FX
# ---------------------------
with st.sidebar:
    st.markdown(
        "<div style='padding:10px 4px 2px 4px;'>"
        "<h3 style='color:#00f0ff;margin-bottom:2px;'>Crypto Dashboard</h3>"
        "<div style='color:rgba(230,247,250,0.7);font-size:12px'>Neon dark UI</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    show_inr = st.checkbox("Show values in INR", value=True)
    fx_rate = st.slider(
        "USD → INR rate",
        min_value=60.0,
        max_value=95.0,
        value=83.0,
        step=0.5,
        help="Approximate FX used for INR display",
    )

def fmt_money(amount, show_inr=True, fx=83.0):
    if show_inr:
        return f"₹{amount*fx:,.2f}"
    return f"${amount:,.2f}"

# ---------------------------
# Pages
# ---------------------------
def page_1_overview():
    section("Overview — Executive KPIs", "Real-time prices and portfolio snapshot")

    top_l, top_r = st.columns([2, 1])
    with top_l:
        coins = ["bitcoin", "ethereum", "binancecoin", "ripple"]
        selected = st.multiselect(
            "Select coins to monitor",
            coins,
            default=["bitcoin", "ethereum"],
        )
    with top_r:
        st.caption("Tip: Toggle INR in the sidebar and adjust FX rate.")

    if not selected:
        st.warning("Select at least one coin to monitor.")
        return

    holdings = {
        c: st.number_input(
            f"Holding {c}",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key=f"h_{c}",
        )
        for c in selected
    }

    prices = get_live_prices(selected)
    total_value_usd = portfolio_value(holdings, prices)

    cols = st.columns(len(selected))
    for i, c in enumerate(selected):
        p = prices.get(c, 0.0)
        val_usd = holdings.get(c, 0.0) * p
        with cols[i]:
            metric_box(
                f"{c.upper()} Price",
                fmt_money(p, show_inr, fx_rate),
            )
            metric_box(
                "Holding Value",
                fmt_money(val_usd, show_inr, fx_rate),
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.metric(
        f"Portfolio Value ({'INR' if show_inr else 'USD'})",
        fmt_money(total_value_usd, show_inr, fx_rate),
    )

    rows = []
    for c in selected:
        rows.append(
            {
                "coin": c.upper(),
                "price (USD)": prices.get(c, 0.0),
                "holding": holdings.get(c, 0.0),
                "value (USD)": holdings.get(c, 0.0) * prices.get(c, 0.0),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("coin"))
    st.markdown("</div>", unsafe_allow_html=True)

def page_2_price_explorer():
    section("Price Explorer & Candlesticks", "Interactive candlestick viewer")
    coins = ["bitcoin", "ethereum", "binancecoin"]
    coin = st.selectbox("Select coin", coins, index=0)
    days = st.number_input(
        "History (days)",
        min_value=30,
        max_value=365,
        value=120,
        step=1,
    )
    df = fetch_coin_data(coin, days=int(days))
    if df.empty:
        st.warning("No data.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    df["open"] = df["price"].shift(1).fillna(df["price"])
    df["high"] = df[["price", "open"]].max(axis=1) * 1.002
    df["low"] = df[["price", "open"]].min(axis=1) * 0.998
    df["close"] = df["price"]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    apply_plotly_layout(fig, title=f"{coin.upper()} Candlestick ({int(days)}d)")
    st.plotly_chart(fig, use_container_width=True)

def page_3_forecast_uncertainty():
    section("Forecast & Uncertainty", "ARIMA / SARIMAX / LSTM forecasts (demo)")

    coins = ["bitcoin", "ethereum"]
    c1, c2 = st.columns(2)
    with c1:
        coin = st.selectbox("Coin for forecast", coins, index=0)
    with c2:
        horizon = st.number_input(
            "Forecast horizon (days)",
            min_value=3,
            max_value=60,
            value=14,
            step=1,
            help="Enter number of days (3–60)",
        )

    df = fetch_coin_data(coin, days=180)
    if df.empty:
        st.warning("No data.")
        return

    series = df.set_index("date")["price"].astype(float)

    f_arima = forecast_arima(series, int(horizon))
    f_sarimax = forecast_sarimax(series, int(horizon))
    f_lstm = forecast_lstm(series, int(horizon))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            name="Historical",
            line=dict(width=2),
        )
    )
    if not f_arima["price"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=f_arima["date"],
                y=f_arima["price"],
                name="ARIMA",
                line=dict(width=1.8),
            )
        )
    if not f_sarimax["price"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=f_sarimax["date"],
                y=f_sarimax["price"],
                name="SARIMAX",
                line=dict(width=1.8),
            )
        )
    if not f_lstm["price"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=f_lstm["date"],
                y=f_lstm["price"],
                name="LSTM",
                line=dict(width=1.8),
            )
        )

    apply_plotly_layout(fig, title=f"{coin.upper()} Forecasts")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast summary table (last point per model)
    table_rows = []
    if not f_arima["price"].isna().all():
        table_rows.append(
            {
                "Model": "ARIMA",
                "Last date": f_arima["date"].iloc[-1].date(),
                "Last forecast (USD)": round(float(f_arima["price"].iloc[-1]), 2),
            }
        )
    if not f_sarimax["price"].isna().all():
        table_rows.append(
            {
                "Model": "SARIMAX",
                "Last date": f_sarimax["date"].iloc[-1].date(),
                "Last forecast (USD)": round(float(f_sarimax["price"].iloc[-1]), 2),
            }
        )
    if not f_lstm["price"].isna().all():
        table_rows.append(
            {
                "Model": "LSTM",
                "Last date": f_lstm["date"].iloc[-1].date(),
                "Last forecast (USD)": round(float(f_lstm["price"].iloc[-1]), 2),
            }
        )

    if table_rows:
        st.markdown("##### Forecast summary")
        st.table(pd.DataFrame(table_rows))

def page_4_sentiment_news():
    section("Sentiment & News Impact", "Simple rule-based demo")
    st.info(
        "Demo sentiment only. For production, plug in a real NLP sentiment API."
    )
    txt = st.text_area("Paste news / tweet / headline", height=120)

    def simple_sentiment(text):
        pos = [
            "gain",
            "surge",
            "rise",
            "bull",
            "optimis",
            "beat",
            "record",
            "soar",
            "adopt",
        ]
        neg = [
            "drop",
            "fall",
            "plunge",
            "bear",
            "crash",
            "hack",
            "fraud",
            "regulat",
            "ban",
            "sell-off",
        ]
        t = text.lower()
        score = sum(t.count(w) for w in pos) - sum(t.count(w) for w in neg)
        return 0.5 + np.tanh(score / 4) / 2

    coins = ["bitcoin", "ethereum", "binancecoin"]
    if not txt.strip():
        scores = {c: random.uniform(0.2, 0.9) for c in coins}
    else:
        scores = {c: round(simple_sentiment(txt + " " + c), 3) for c in coins}
    df = pd.DataFrame({"coin": list(scores.keys()), "sentiment": list(scores.values())})
    fig = px.bar(df, x="coin", y="sentiment", range_y=[0, 1], text="sentiment")
    apply_plotly_layout(fig, title="Sentiment Scores")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

def page_5_volatility_risk():
    section("Volatility & Risk", "Rolling volatility + Monte Carlo VaR")
    coins = ["bitcoin", "ethereum", "binancecoin"]
    selected = st.selectbox("Select coin", coins, index=0)
    window = st.slider("Rolling window (days)", 3, 60, 14)
    df = fetch_coin_data(selected, days=365)
    if df.empty:
        st.warning("No data.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    df["returns"] = df["price"].pct_change().fillna(0)
    df["volatility"] = df["returns"].rolling(window).std() * np.sqrt(365)
    fig = px.line(
        df,
        x="date",
        y="volatility",
        title=f"{selected.upper()} Annualized Volatility (rolling {window}d)",
    )
    apply_plotly_layout(
        fig, title=f"{selected.upper()} Annualized Volatility (rolling {window}d)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # simple VaR metrics
    last = df["price"].iloc[-1]
    mu = df["returns"].mean() * 365
    sigma = df["returns"].std() * np.sqrt(365)
    sims = 200
    mc = np.zeros((sims, 30))
    for s in range(sims):
        prices = [last]
        for d in range(30):
            shock = np.random.normal(
                (mu - 0.5 * sigma**2) * (1 / 365), sigma * np.sqrt(1 / 365)
            )
            prices.append(prices[-1] * np.exp(shock))
        mc[s, :] = prices[1:]
    q5 = np.percentile(mc[:, -1], 5)
    c1, c2 = st.columns(2)
    with c1:
        metric_box("Current Price", fmt_money(last, show_inr, fx_rate))
    with c2:
        metric_box("30d 5% VaR (price)", fmt_money(q5, show_inr, fx_rate))

def page_6_indicators():
    section("Indicators Dashboard", "SMA / EMA / RSI")
    coins = ["bitcoin", "ethereum"]
    selected = st.selectbox("Select coin", coins, index=0)
    df = fetch_coin_data(selected, days=180)
    if df.empty:
        st.warning("No data.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    df["SMA20"] = SMA(df["price"], 20)
    df["EMA20"] = EMA(df["price"], 20)
    df["RSI14"] = RSI(df["price"], 14)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_box("Current Price", fmt_money(df["price"].iloc[-1], show_inr, fx_rate))
    with c2:
        metric_box("SMA20", fmt_money(df["SMA20"].iloc[-1], show_inr, fx_rate))
    with c3:
        metric_box("RSI14", f"{df['RSI14'].iloc[-1]:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name="Price"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["EMA20"], name="EMA20"))
    apply_plotly_layout(fig, title=f"{selected.upper()} Indicators")
    st.plotly_chart(fig, use_container_width=True)

def page_7_correlations():
    section("Correlations & Market Structure", "Return correlations and PCA")
    coins = ["bitcoin", "ethereum", "binancecoin", "ripple"]
    selected = st.multiselect(
        "Select coins", coins, default=["bitcoin", "ethereum"]
    )
    if not selected:
        st.warning("Choose at least one coin.")
        return
    price_series = {}
    for c in selected:
        df = fetch_coin_data(c, days=120)
        if not df.empty:
            price_series[c] = df.set_index("date")["price"]
    if not price_series:
        st.warning("No data for selected coins.")
        return
    dfp = pd.DataFrame(price_series).dropna().sort_index()
    corr = dfp.pct_change().corr()
    fig = px.imshow(corr, text_auto=True, title="Return Correlations")
    apply_plotly_layout(fig, title="Return Correlations")
    st.plotly_chart(fig, use_container_width=True)

def page_8_feature_importance():
    section("Feature Importance & Explainability", "Permutation importance")
    coins = ["bitcoin", "ethereum", "binancecoin"]
    selected = st.selectbox("Select coin", coins, index=0)
    df = fetch_coin_data(selected, days=365)
    if df.empty:
        st.warning("No data.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    df["ret1"] = df["price"].pct_change(1)
    df["SMA7"] = SMA(df["price"], 7)
    df["SMA21"] = SMA(df["price"], 21)
    df["EMA9"] = EMA(df["price"], 9)
    df["RSI14"] = RSI(df["price"], 14)
    df["vol7"] = df["ret1"].rolling(7).std()
    df = df.dropna().reset_index(drop=True)
    if len(df) < 30:
        st.warning("Not enough data for feature importance. Try a longer history.")
        return
    X = df[["SMA7", "SMA21", "EMA9", "RSI14", "vol7"]].fillna(0)
    y = df["price"].shift(-1).dropna()
    X = X.iloc[:-1, :]
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        r = permutation_importance(
            rf, X, y, n_repeats=20, random_state=42, n_jobs=1
        )
        imp = (
            pd.DataFrame(
                {"feature": X.columns, "importance": r.importances_mean}
            )
            .sort_values("importance", ascending=False)
        )
        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            orientation="h",
            title=f"{selected.upper()} Permutation Importance",
        )
        apply_plotly_layout(
            fig, title=f"{selected.upper()} Permutation Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp.style.format({"importance": "{:.4f}"}))
    except Exception as e:
        st.error("Feature importance failed: " + str(e))

def page_9_strategy_backtest():
    section("Strategy Backtest & Performance", "Simple SMA crossover backtest")
    coins = ["bitcoin", "ethereum"]
    selected = st.selectbox("Select coin to backtest", coins, index=0)
    df = fetch_coin_data(selected, days=365)
    if df.empty:
        st.warning("No data.")
        return
    short = st.slider("Short SMA", 5, 30, 10)
    long = st.slider("Long SMA", 20, 120, 50)
    try:
        df = df.sort_values("date").reset_index(drop=True)
        df["SMA_short"] = SMA(df["price"], short)
        df["SMA_long"] = SMA(df["price"], long)
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            st.warning("Not enough data after smoothing.")
            return
        df["signal"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
        df["position"] = df["signal"].shift(1).fillna(0)
        df["returns"] = df["price"].pct_change().fillna(0) * df["position"]
        df["equity"] = (1 + df["returns"]).cumprod()
        back = df
        st.line_chart(back.set_index("date")["equity"], height=300)
        total_return = (
            back["equity"].iloc[-1] - back["equity"].iloc[0]
        ) / back["equity"].iloc[0]
        st.metric("Total Return (strategy)", f"{total_return*100:.2f}%")
    except Exception as e:
        st.error("Backtest failed: " + str(e))

def page_10_interactive_explorer():
    section("Interactive Explorer", "Slice, plot & download ranged data")
    coins = ["bitcoin", "ethereum", "binancecoin"]
    selected = st.selectbox("Select coin", coins, index=0)
    hist_days = st.number_input(
        "History days",
        min_value=30,
        max_value=720,
        value=180,
        step=1,
    )
    df = fetch_coin_data(selected, days=int(hist_days))
    if df.empty:
        st.warning("No data.")
        return
    df = df.sort_values("date").reset_index(drop=True)
    start_date = st.date_input(
        "Start date",
        value=df["date"].iloc[0].date(),
        min_value=df["date"].iloc[0].date(),
        max_value=df["date"].iloc[-1].date(),
    )
    end_date = st.date_input(
        "End date",
        value=df["date"].iloc[-1].date(),
        min_value=start_date,
        max_value=df["date"].iloc[-1].date(),
    )
    mask = (df["date"].dt.date >= start_date) & (
        df["date"].dt.date <= end_date
    )
    sub = df[mask]
    if sub.empty:
        st.warning("No data in the chosen range.")
        return
    st.line_chart(sub.set_index("date")["price"], height=350)
    csv = sub.to_csv(index=False).encode()
    st.download_button(
        "Download slice CSV",
        data=csv,
        file_name=f"{selected}_slice.csv",
        mime="text/csv",
    )
    show_sma = st.checkbox("Show SMA(20)", value=True)
    show_ema = st.checkbox("Show EMA(9)", value=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["price"], name="Price"))
    if show_sma:
        fig.add_trace(
            go.Scatter(x=sub["date"], y=SMA(sub["price"], 20), name="SMA20")
        )
    if show_ema:
        fig.add_trace(
            go.Scatter(x=sub["date"], y=EMA(sub["price"], 9), name="EMA9")
        )
    apply_plotly_layout(fig, title=f"{selected.upper()} Explorer")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page Dictionary and sidebar nav
# ---------------------------
page_dict = {
    "Overview": page_1_overview,
    "Price Explorer": page_2_price_explorer,
    "Forecasts": page_3_forecast_uncertainty,
    "Sentiment": page_4_sentiment_news,
    "Volatility & Risk": page_5_volatility_risk,
    "Indicators": page_6_indicators,
    "Correlations": page_7_correlations,
    "Feature Importance": page_8_feature_importance,
    "Backtest": page_9_strategy_backtest,
    "Interactive Explorer": page_10_interactive_explorer,
}

st.sidebar.markdown("---")
selected_page = st.sidebar.radio("Navigation", list(page_dict.keys()), index=0)
st.sidebar.markdown(
    "<div style='padding:4px 4px 8px 4px;'>"
    "<div style='font-size:11px;color:rgba(230,247,250,0.7)'>Theme</div>"
    "<div style='font-weight:700;color:#7CFF6A'>Neon Dark</div>"
    "</div>",
    unsafe_allow_html=True,
)

page_dict[selected_page]()