import time
from datetime import datetime, timedelta
from threading import Lock

import pandas as pd
import streamlit as st
from kiteconnect import KiteConnect, KiteTicker


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


@st.cache_resource(show_spinner=False)
def get_kite(api_key: str, access_token: str):
    return KiteConnect(api_key=api_key, access_token=access_token)


def generate_session(api_key: str, api_secret: str, request_token: str):
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token.strip(), api_secret=api_secret.strip())
    return data


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_nse_instruments(api_key: str, access_token: str):
    kite = get_kite(api_key, access_token)
    rows = kite.instruments("NSE")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Kite returned no NSE instruments.")
    df = df[df.exchange == "NSE"].copy()
    df = df[df.instrument_type == "EQ"].copy()
    return df[["instrument_token", "exchange_token", "tradingsymbol", "name", "segment", "exchange"]]


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_daily_history(instrument_token: int, from_date: str, to_date: str, access_token: str, api_key: str):
    kite = get_kite(api_key, access_token)
    # Kite historical REST API is rate-limited; keep a conservative cadence.
    time.sleep(0.36)
    rows = kite.historical_data(
        int(instrument_token), from_date, to_date, "day", continuous=False, oi=False
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")


def validate_kite_access(api_key: str, access_token: str):
    """Fail fast with the broker's actual authentication error."""
    kite = get_kite(api_key, access_token)
    return kite.profile()


def load_histories(universe_df: pd.DataFrame, api_key: str, access_token: str, years_back=2):
    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=int(365.25 * years_back))
    frames = {}
    failed = []
    for _, row in universe_df.iterrows():
        symbol = str(row.tradingsymbol)
        try:
            df = fetch_daily_history(
                int(row.instrument_token), from_date.isoformat(), to_date.isoformat(),
                access_token, api_key
            )
            if df is None or df.empty:
                failed.append(f"{symbol}: empty historical response")
            else:
                frames[symbol] = df
        except Exception as exc:
            failed.append(f"{symbol}: {type(exc).__name__}: {exc}")
    return frames, failed


class LiveFeed:
    def __init__(self, api_key, access_token, tokens):
        self.api_key = api_key
        self.access_token = access_token
        self.tokens = [int(x) for x in tokens]
        self.ticks = {}
        self.status = "created"
        self.last_error = ""
        self.lock = Lock()
        self.kws = KiteTicker(api_key, access_token, debug=False)
        self.kws.on_connect = self._on_connect
        self.kws.on_ticks = self._on_ticks
        self.kws.on_error = self._on_error
        self.kws.on_close = self._on_close
        self.kws.on_reconnect = self._on_reconnect

    def _on_connect(self, ws, response):
        ws.subscribe(self.tokens)
        ws.set_mode(ws.MODE_FULL, self.tokens)
        self.status = "connected"

    def _on_ticks(self, ws, ticks):
        with self.lock:
            for tick in ticks:
                self.ticks[int(tick["instrument_token"])] = tick

    def _on_error(self, ws, code, reason):
        self.last_error = f"{code}: {reason}"
        self.status = "error"

    def _on_close(self, ws, code, reason):
        self.status = "closed"

    def _on_reconnect(self, ws, attempts_count):
        self.status = f"reconnecting ({attempts_count})"

    def start(self):
        if self.status in {"connected", "connecting"}:
            return
        self.status = "connecting"
        self.kws.connect(threaded=True)

    def stop(self):
        try:
            self.kws.close()
        finally:
            self.status = "stopped"

    def snapshot(self):
        with self.lock:
            return dict(self.ticks)


@st.cache_resource(show_spinner=False)
def get_live_feed(api_key: str, access_token: str, tokens_tuple: tuple):
    return LiveFeed(api_key, access_token, tokens_tuple)
