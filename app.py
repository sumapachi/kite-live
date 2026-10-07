import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from kite_data import fetch_daily_history, fetch_nse_instruments, generate_session, get_kite, get_live_feed, load_histories, validate_kite_access

st.set_page_config(page_title="Midcap + Smallcap Kite Research Scanner", page_icon="📈", layout="wide")

APP_VERSION = "3.1.0"
NSE_INDEX_URLS = {
    "Nifty Midcap 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}
MAX_ALLOCATION = 10_000
STOP_PCT = 0.15
MIN_PRICE = 20.0
MIN_AVG_VALUE = 2_000_000


def normalise_symbols(df, segment):
    cols = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = next((cols[x] for x in ["symbol", "ticker", "scrip", "stock", "company symbol"] if x in cols), None)
    if symbol_col is None:
        raise ValueError("Constituent CSV does not contain a usable Symbol column.")
    out = pd.DataFrame({"Symbol": df[symbol_col].astype(str).str.strip().str.upper()})
    out = out[~out["Symbol"].isin(["", "NAN", "NONE", "SYMBOL"])].copy()
    out["Segment"] = segment
    return out.drop_duplicates("Symbol")


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_nse_constituents():
    frames, errors = [], []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36",
        "Accept": "text/csv,application/octet-stream,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    for segment, url in NSE_INDEX_URLS.items():
        try:
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            frames.append(normalise_symbols(pd.read_csv(io.BytesIO(r.content)), segment))
        except Exception as exc:
            errors.append(f"{segment}: {exc}")
    if not frames:
        raise RuntimeError("Could not refresh NSE Indices constituents. " + " | ".join(errors))
    universe = pd.concat(frames, ignore_index=True)
    grouped = universe.groupby("Symbol", as_index=False).agg(
        Segment=("Segment", lambda s: " + ".join(sorted(set(s))))
    )
    return grouped


def validate_uploaded(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = next((cols[x] for x in ["symbol", "ticker", "scrip", "stock"] if x in cols), None)
    if symbol_col is None:
        raise ValueError("CSV must contain a Symbol column.")
    out = pd.DataFrame({"Symbol": df[symbol_col].astype(str).str.strip().str.upper()})
    out = out[~out.Symbol.isin(["", "NAN", "NONE"])].drop_duplicates()
    out["Segment"] = "Custom"
    return out


def map_to_kite(universe, instruments):
    ins = instruments.copy()
    ins["tradingsymbol"] = ins["tradingsymbol"].astype(str).str.upper()
    out = universe.copy()
    out = out.merge(
        ins[["instrument_token", "tradingsymbol", "name"]],
        left_on="Symbol", right_on="tradingsymbol", how="left",
    )
    out["KiteOK"] = out.instrument_token.notna()
    return out


def calculate(x):
    if x is None or x.empty:
        return None
    needed = ["open", "high", "low", "close", "volume"]
    if any(c not in x.columns for c in needed):
        return None
    x = x[needed].copy()
    x.index = pd.to_datetime(x.index).tz_localize(None) if getattr(pd.to_datetime(x.index), "tz", None) is not None else pd.to_datetime(x.index)
    for c in needed:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["close", "volume"])
    if len(x) < 270:
        return None
    x["SMA20"] = x.close.rolling(20).mean()
    x["SMA50"] = x.close.rolling(50).mean()
    x["SMA150"] = x.close.rolling(150).mean()
    x["EMA220"] = x.close.ewm(span=220, adjust=False, min_periods=220).mean()
    x["ATR14"] = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1).rolling(14).mean()
    x["Prev52WLow"] = x.low.rolling(252).min().shift(1)
    x["Prev52WHighClose"] = x.close.rolling(252).max().shift(1)
    x["Return5D"] = x.close.pct_change(5) * 100
    x["Return20D"] = x.close.pct_change(20) * 100
    x["Return60D"] = x.close.pct_change(60) * 100
    x["AvgVol20"] = x.volume.rolling(20).mean()
    x["RVOL20"] = x.volume / x.AvgVol20
    x["AvgValue20"] = (x.close * x.volume).rolling(20).mean()
    x["Dip90"] = (x.low < x.EMA220).shift(1).astype(float).rolling(90).sum().fillna(0)
    x["C1"] = x.SMA150 > x.EMA220
    x["C2"] = x.close > x.SMA50
    x["C3"] = x.SMA50 > x.SMA150
    x["C4"] = x.close > 1.25 * x.Prev52WLow
    x["C5"] = x.Dip90 > 0
    x["Breakout"] = x.close > x.Prev52WHighClose
    x["ConditionsMet"] = x[["C1", "C2", "C3", "C4", "C5", "Breakout"]].sum(axis=1)
    x["ConfirmedSignal"] = x.ConditionsMet == 6
    x["PctAboveEMA220"] = (x.close / x.EMA220 - 1) * 100
    x["PctAbove52WLow"] = (x.close / x.Prev52WLow - 1) * 100
    x["ATRpct"] = x.ATR14 / x.close * 100
    return x


def percentile(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return 50.0 if s.empty or pd.isna(value) else float((s <= value).mean() * 100)


def scan(universe, frames, min_price, min_avg_value):
    rows, failed = [], []
    for _, r in universe.iterrows():
        x = calculate(frames.get(r.Symbol))
        if x is None:
            failed.append(r.Symbol)
            continue
        a = x.iloc[-1]
        if any(pd.isna(a[c]) for c in ["EMA220", "Prev52WLow", "SMA50", "SMA150", "ATR14"]):
            failed.append(r.Symbol)
            continue
        liquid = bool(a.close >= min_price and a.AvgValue20 >= min_avg_value and a.volume > 0)
        trend = (10 if a.C1 else 0) + (10 if a.C2 else 0) + (10 if a.C3 else 0)
        momentum = percentile(x.Return20D, a.Return20D) * 0.15 + percentile(x.Return60D, a.Return60D) * 0.15
        breakout_volume = (10 if a.Breakout else 0) + min(max((a.RVOL20 - 1) * 10, 0), 15)
        score = trend + momentum + breakout_volume + (15 if liquid else 0)
        rows.append({
            "Symbol": r.Symbol, "Segment": r.Segment, "Date": x.index[-1].date(),
            "Close": a.close, "SMA20": a.SMA20, "SMA50": a.SMA50, "SMA150": a.SMA150, "EMA220": a.EMA220,
            "ATR14": a.ATR14, "ATRpct": a.ATRpct, "Return5D": a.Return5D, "Return20D": a.Return20D, "Return60D": a.Return60D,
            "RVOL20": a.RVOL20, "AvgValue20": a.AvgValue20, "Prev52WLow": a.Prev52WLow,
            "Prev52WHighClose": a.Prev52WHighClose, "ConditionsMet": int(a.ConditionsMet), "Breakout": bool(a.Breakout),
            "C1_150SMA>220EMA": bool(a.C1), "C2_Close>50SMA": bool(a.C2), "C3_50SMA>150SMA": bool(a.C3),
            "C4_Close>1.25x52WLow": bool(a.C4), "C5_DippedBelowEMA220_90D": bool(a.C5), "ConfirmedSignal": bool(a.ConfirmedSignal),
            "Liquid": liquid, "PctAboveEMA220": a.PctAboveEMA220, "PctAbove52WLow": a.PctAbove52WLow, "ResearchScore": score,
        })
    return pd.DataFrame(rows), failed


def live_overlay(scanner, universe, feed):
    if scanner.empty:
        return scanner
    ticks = feed.snapshot()
    if not ticks:
        return scanner
    token_map = universe.set_index("Symbol")["instrument_token"].to_dict()
    z = scanner.copy()
    z["Live"] = False
    z["LTP"] = np.nan
    z["DayChange%"] = np.nan
    z["VWAP"] = np.nan
    z["VsVWAP%"] = np.nan
    z["DayHigh"] = np.nan
    z["DayLow"] = np.nan
    z["DayVolume"] = np.nan
    z["LiveMomentum"] = np.nan
    for idx, row in z.iterrows():
        token = token_map.get(row.Symbol)
        tick = ticks.get(int(token)) if pd.notna(token) else None
        if not tick:
            continue
        ltp = float(tick.get("last_price", np.nan))
        ohlc = tick.get("ohlc") or {}
        prev_close = float(ohlc.get("close", np.nan))
        vwap = float(tick.get("average_traded_price", np.nan))
        z.at[idx, "Live"] = True
        z.at[idx, "LTP"] = ltp
        z.at[idx, "DayChange%"] = ((ltp / prev_close) - 1) * 100 if prev_close else np.nan
        z.at[idx, "VWAP"] = vwap
        z.at[idx, "VsVWAP%"] = ((ltp / vwap) - 1) * 100 if vwap else np.nan
        z.at[idx, "DayHigh"] = float(ohlc.get("high", np.nan))
        z.at[idx, "DayLow"] = float(ohlc.get("low", np.nan))
        z.at[idx, "DayVolume"] = float(tick.get("volume_traded", np.nan))
        z.at[idx, "LiveMomentum"] = z.at[idx, "DayChange%"] + max(z.at[idx, "VsVWAP%"], 0) * 0.5
    z["LiveResearchScore"] = z["ResearchScore"] + z["LiveMomentum"].fillna(0).clip(-10, 10)
    z = z.sort_values(["LiveResearchScore", "ResearchScore", "ConditionsMet"], ascending=False).reset_index(drop=True)
    z["ResearchRank"] = np.arange(1, len(z) + 1)
    return z


st.title("📈 Midcap + Smallcap — Kite Live Research Scanner")
st.caption(f"Research-grade universe + broker-grade live feed | v{APP_VERSION}")

st.sidebar.header("🔐 Kite authentication")
api_key = st.sidebar.text_input("Kite API key", value=st.secrets.get("KITE_API_KEY", ""), type="password")
api_secret = st.sidebar.text_input("Kite API secret", value=st.secrets.get("KITE_API_SECRET", ""), type="password")
request_token = st.sidebar.text_input("Morning request token", type="password", help="Paste the request_token from your Kite Connect redirect. The app exchanges it for the access token.")
access_token_manual = st.sidebar.text_input("Or paste access token", type="password")

if request_token and api_key and api_secret:
    if st.sidebar.button("🔑 Authenticate with request token"):
        try:
            session_data = generate_session(api_key, api_secret, request_token)
            st.session_state["kite_access_token"] = session_data["access_token"]
            st.session_state["kite_login_time"] = str(session_data.get("login_time", ""))
            st.sidebar.success("Authenticated. Access token loaded for this Streamlit session.")
        except Exception as exc:
            st.sidebar.error(f"Authentication failed: {exc}")

if access_token_manual:
    st.session_state["kite_access_token"] = access_token_manual.strip()

access_token = st.session_state.get("kite_access_token", "")
if not api_key or not access_token:
    st.info("Enter your Kite API key and paste either the morning request token or an access token in the sidebar.")
    st.stop()

# Validate credentials before spending ~2 minutes on hundreds of historical requests.
try:
    profile = validate_kite_access(api_key, access_token)
    st.sidebar.success(f"Kite authenticated: {profile.get('user_name', profile.get('user_id', 'OK'))}")
except Exception as exc:
    st.error(f"Kite authentication failed before historical-data scan: {type(exc).__name__}: {exc}")
    st.warning("Generate a fresh access token today and paste that token. Do not paste the API secret into the access-token field.")
    st.stop()

try:
    instruments = fetch_nse_instruments(api_key, access_token)
except Exception as exc:
    st.error(f"Could not load Kite NSE instruments: {type(exc).__name__}: {exc}")
    st.stop()

st.sidebar.header("Universe")
universe_mode = st.sidebar.selectbox("Universe", ["Combined — Midcap 150 + Smallcap 250", "Nifty Midcap 150", "Nifty Smallcap 250", "Custom CSV"])
if universe_mode == "Custom CSV":
    uploaded = st.sidebar.file_uploader("Upload universe CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV containing a Symbol column.")
        st.stop()
    universe = validate_uploaded(pd.read_csv(uploaded))
else:
    full = fetch_nse_constituents()
    if universe_mode == "Nifty Midcap 150":
        universe = full[full.Segment.str.contains("Nifty Midcap 150")].copy()
    elif universe_mode == "Nifty Smallcap 250":
        universe = full[full.Segment.str.contains("Nifty Smallcap 250")].copy()
    else:
        universe = full.copy()
universe = map_to_kite(universe, instruments)
missing_kite = universe[~universe.KiteOK].Symbol.tolist()
universe = universe[universe.KiteOK].copy()

st.sidebar.metric("Kite-mapped universe", len(universe))
min_price = st.sidebar.number_input("Minimum price (₹)", min_value=0.0, value=MIN_PRICE, step=5.0)
min_avg_value = st.sidebar.number_input("Minimum 20D avg traded value (₹)", min_value=0.0, value=float(MIN_AVG_VALUE), step=500_000.0, format="%.0f")

if st.sidebar.button("🧹 Clear historical cache"):
    fetch_daily_history.clear()
    st.rerun()

with st.spinner(f"Building daily research history for {len(universe)} stocks… first run can take a few minutes because Kite historical data is rate-limited."):
    frames, history_failed = load_histories(universe, api_key, access_token, years_back=2)

with st.spinner("Calculating technical research factors…"):
    scanner, failed = scan(universe, frames, min_price, min_avg_value)

if scanner.empty:
    st.error("No valid securities were returned.")
    if history_failed:
        st.warning(f"Historical-data failures: {len(history_failed)}. The first few Kite responses are shown below.")
        st.code("\n".join(history_failed[:15]))
    else:
        st.warning(f"Kite mapped {len(universe)} securities, but none had enough usable daily history for the research model.")
    st.stop()

scanner["PlannedEntry"] = scanner.Close
scanner["ATRStop"] = scanner.PlannedEntry - 2 * scanner.ATR14
scanner["15pctStop"] = scanner.PlannedEntry * (1 - STOP_PCT)
scanner["SuggestedStop"] = np.maximum(scanner.ATRStop, scanner["15pctStop"])
scanner["Shares"] = np.floor(MAX_ALLOCATION / scanner.PlannedEntry).astype(int)
scanner["CapitalRequired"] = scanner.Shares * scanner.PlannedEntry
scanner["RiskToSuggestedStop"] = scanner.Shares * (scanner.PlannedEntry - scanner.SuggestedStop).clip(lower=0)
scanner = scanner.sort_values(["ConfirmedSignal", "ResearchScore", "ConditionsMet", "RVOL20"], ascending=[False, False, False, False]).reset_index(drop=True)
scanner["ResearchRank"] = np.arange(1, len(scanner) + 1)

st.sidebar.header("⚡ Live feed")
if st.sidebar.button("▶ Start / restart live feed"):
    tokens = tuple(universe.instrument_token.astype(int).tolist())
    feed = get_live_feed(api_key, access_token, tokens)
    feed.start()
    st.session_state["live_started"] = True

feed = None
if st.session_state.get("live_started"):
    tokens = tuple(universe.instrument_token.astype(int).tolist())
    feed = get_live_feed(api_key, access_token, tokens)
    live_scanner = live_overlay(scanner, universe, feed)
else:
    live_scanner = scanner.copy()

confirmed = live_scanner[live_scanner.ConfirmedSignal].copy()
near = live_scanner[live_scanner.ConditionsMet >= 5].copy()
liquid = live_scanner[live_scanner.Liquid].copy()

c = st.columns(7)
c[0].metric("Scanned", len(live_scanner))
c[1].metric("Midcap", int(live_scanner.Segment.str.contains("Midcap").sum()))
c[2].metric("Smallcap", int(live_scanner.Segment.str.contains("Smallcap").sum()))
c[3].metric("6/6", len(confirmed))
c[4].metric("≥5/6", len(near))
c[5].metric("Liquid", len(liquid))
c[6].metric("Live ticks", int(live_scanner.Live.sum()) if "Live" in live_scanner else 0)

if feed:
    st.success(f"Live feed: {feed.status} | ticks received: {len(feed.snapshot())}")
    if feed.last_error:
        st.warning(feed.last_error)
else:
    st.info("Live feed is off. Start it from the sidebar after authentication.")

if missing_kite:
    st.warning(f"{len(missing_kite)} universe symbols are not currently mapped to NSE EQ instruments in Kite.")

st.info(f"Daily research date: {live_scanner.Date.max()} | Live layer uses Kite WebSocket ticks. Kite's WebSocket is the appropriate mechanism for continuous market data; REST quotes/historical endpoints are rate-limited. citeturn0search0turn1search10")

DISPLAY = ["ResearchRank", "Symbol", "Segment", "Date", "Close", "LTP", "DayChange%", "VWAP", "VsVWAP%", "LiveResearchScore", "ResearchScore", "ConditionsMet", "Breakout", "Return20D", "Return60D", "RVOL20", "ATRpct", "AvgValue20", "SuggestedStop", "Shares", "CapitalRequired"]
DISPLAY = [x for x in DISPLAY if x in live_scanner.columns]

def show(df, limit=None):
    z = df[DISPLAY].copy()
    if limit:
        z = z.head(limit)
    st.dataframe(z.round(2), use_container_width=True, hide_index=True)


t1, t2, t3, t4, t5 = st.tabs(["🟢 Confirmed", "🟡 Near Signal", "🔥 Live Research Leaders", "📋 Full Universe", "🧩 Conditions"])
with t1:
    st.subheader("Confirmed 6/6 signals")
    show(confirmed)
with t2:
    st.subheader("5/6 or better")
    show(near)
with t3:
    st.subheader("Highest-quality research candidates")
    show(live_scanner[live_scanner.Liquid], 50)
with t4:
    st.subheader("Complete Midcap + Smallcap scan")
    show(live_scanner)
with t5:
    matrix = ["ResearchRank", "Symbol", "Segment", "ConditionsMet", "C1_150SMA>220EMA", "C2_Close>50SMA", "C3_50SMA>150SMA", "C4_Close>1.25x52WLow", "C5_DippedBelowEMA220_90D", "Breakout", "Liquid", "ConfirmedSignal"]
    st.dataframe(live_scanner[matrix], use_container_width=True, hide_index=True)

st.subheader("Top 25 research candidates")
st.dataframe(live_scanner.head(25)[DISPLAY].round(2), use_container_width=True, hide_index=True)

csv_bytes = live_scanner.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download full research CSV", csv_bytes, file_name=f"kite_midcap_smallcap_scan_{live_scanner.Date.max()}.csv", mime="text/csv")

all_failed = sorted(set(failed + history_failed))
if all_failed:
    st.warning(f"{len(all_failed)} symbols could not be calculated/downloaded. Review before treating the scan as exhaustive.")
    with st.expander("Failed symbols"):
        st.write("\n".join(all_failed[:300]))

st.caption(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Daily research via Kite historical candles | Live market layer via KiteTicker WebSocket | No order-placement code is included.")
