"""ORB Command Center - run with:  streamlit run app.py"""
from __future__ import annotations

import os
from datetime import date as _date, time as _time, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import engine as E
import fno as F
import journal as J
import leaders as L
import store
import ui as U
import auth
import pwa

st.set_page_config(page_title="ORB Command Center", page_icon="📈", layout="wide")
pwa.inject()
store.init()
auth.require_login()


# ----------------------------------------------------------------------------- cached loaders
@st.cache_data(ttl=3600, show_spinner=False)
def load_uni(name):
    return E.load_universe(name)


@st.cache_data(ttl=600, show_spinner=False)
def load_daily(symbols: tuple):
    return E.download_batch(list(symbols), "6mo", "1d")


@st.cache_data(ttl=10, show_spinner=False)
def load_1m(sym: str):
    return E.download_batch([sym], "1d", "1m").get(sym)


@st.cache_data(ttl=15, show_spinner=False)
def load_1m_5d(sym: str):
    """5 sessions of 1-minute bars: source for 3m/5m/15m charts (warms up EMAs)."""
    return E.download_batch([sym], "5d", "1m").get(sym)


@st.cache_data(ttl=300, show_spinner=False)
def nifty_ret20():
    return E.index_ret20()


@st.cache_data(ttl=90, show_spinner=False)
def load_breadth(name, _daily, _industry):
    return E.compute_breadth(_daily, _industry)


# ---- F&O universe loaders (used by Breakout Leaders / Trend Ignition / Gaps / Replay)
@st.cache_data(ttl=3600, show_spinner=False)
def load_fno():
    return F.load_fno()


@st.cache_data(ttl=600, show_spinner=False)
def load_fno_daily(symbols: tuple):
    return E.download_batch(list(symbols), "6mo", "1d")


@st.cache_data(ttl=20, show_spinner=False)
def load_fno_1m(symbols: tuple):
    return E.download_batch(list(symbols), "1d", "1m", chunk=40)


@st.cache_data(ttl=300, show_spinner=False)
def load_fno_5d(symbols: tuple):
    """Up to 5 sessions of 1-minute bars (yfinance keeps ~7 days of 1m history) - source for replaying a past day."""
    return E.download_batch(list(symbols), "5d", "1m", chunk=40)


def beep():
    components.html(
        "<script>try{const c=new (window.AudioContext||window.webkitAudioContext)();"
        "[880,1175].forEach((f,i)=>{const o=c.createOscillator();o.frequency.value=f;o.connect(c.destination);"
        "o.start(c.currentTime+i*0.25);o.stop(c.currentTime+i*0.25+0.2);});}catch(e){}</script>", height=0)


def parse_nums(txt, default, cast=float):
    try:
        v = tuple(cast(x) for x in txt.replace(" ", "").split(",") if x)
        return v if len(v) == 3 else default
    except Exception:
        return default


def pick(options, val, fallback=0):
    return options.index(val) if val in options else fallback


# ----------------------------------------------------------------------------- sidebar / config
saved = E.load_cfg()
with st.sidebar:
    st.title("ORB Command Center")
    auth.sidebar_controls()
    universe = st.selectbox("Universe", list(E.NSE_LISTS), index=2)
    with st.expander("Risk", expanded=True):
        capital = st.number_input("Capital (₹)", 10000.0, 1e9, float(saved.capital), 10000.0)
        risk_pct = st.number_input("Risk per trade (%)", 0.05, 5.0, float(saved.risk_pct), 0.05)
        leverage = st.number_input("Max leverage (MIS)", 1.0, 20.0, float(saved.leverage), 0.5)
        rr = parse_nums(st.text_input("Targets T1,T2,T3 (R multiples)", ",".join(f"{x:g}" for x in saved.rr)), saved.rr)
        book = parse_nums(st.text_input("Booking % at T1,T2,T3", ",".join(f"{x:g}" for x in saved.book_pct)), saved.book_pct, int)
    with st.expander("ORB and 1-minute confirmation"):
        orb_minutes = st.selectbox("ORB window (minutes)", [5, 15, 30], index=pick([5, 15, 30], saved.orb_minutes, 1))
        confirm_bars = st.selectbox("Closes beyond ORB required", [1, 2, 3], index=pick([1, 2, 3], saved.confirm_bars))
        vol_mult = st.slider("Signal volume vs 20-bar avg (x)", 1.0, 3.0, float(saved.vol_mult), 0.1)
        body_ratio = st.slider("Min candle body / range", 0.3, 0.9, float(saved.body_ratio), 0.05)
        min_orb, max_orb = st.slider("Tradable ORB range (% of price)", 0.1, 4.0, (float(saved.min_orb_pct), float(saved.max_orb_pct)), 0.05)
        last_entry = st.text_input("No new entries after (HH:MM)", saved.last_entry)
    with st.expander("Trailing stop and exits"):
        modes = ["Structure", "EMA9", "ATR", "None"]
        trail_mode = st.selectbox("Trail mode (starts after T1)", modes, index=pick(modes, saved.trail_mode))
        trail_n = st.slider("Structure: last N 1-min bars", 2, 15, int(saved.trail_n))
        atr_k = st.slider("ATR trail multiple", 1.0, 6.0, float(saved.atr_k), 0.5)
        eod_exit = st.text_input("Square-off time (HH:MM)", saved.eod_exit)
    with st.expander("Scanner"):
        min_score = st.slider("Strong-buy min score", 50, 95, int(saved.min_score))
        top_k = st.slider("Stocks sent to 1-min stage", 10, 100, int(saved.top_k), 5)
        fresh_min = st.slider("Alert only if signal is newer than (min)", 2, 60, int(saved.fresh_min))
        min_turn = st.number_input("Min avg turnover (₹ Cr)", 0.0, 500.0, float(saved.min_turnover_cr), 5.0)
    with st.expander("Breakout Leaders and F&O"):
        ld_min_price = st.number_input("Junk filter: min price (₹)", 0.0, 5000.0, float(saved.ld_min_price), 10.0)
        ld_min_turn = st.number_input("Illiquid filter: min avg turnover (₹ Cr)", 0.0, 1000.0, float(saved.ld_min_turnover_cr), 5.0)
        ld_spurt = st.slider("⚡ SPURT: volume jump ≥ (x)", 1.0, 3.0, float(saved.ld_spurt_vol), 0.1)
        ld_strong = st.slider("🔥 STRONG: volume jump ≥ (x)", 1.5, 6.0, float(saved.ld_strong_vol), 0.1)
        ld_expl = st.slider("💥 EXPLOSIVE: volume jump ≥ (x)", 2.0, 10.0, float(saved.ld_explosive_vol), 0.1)
        ld_strong_score = st.slider("STRONG min strength score", 40, 90, int(saved.ld_strong_score))
        ld_expl_score = st.slider("EXPLOSIVE min strength score", 50, 95, int(saved.ld_explosive_score))
        ld_max_age = st.slider("Late-entry filter: drop leaders older than (min)", 10, 180, int(saved.ld_max_age), 5)
        ld_late_ext = st.slider("Late-entry filter: drop if already this many ORB-widths past the breakout", 0.25, 3.0,
                                float(saved.ld_late_ext), 0.25)
        ld_shorts = st.checkbox("Include breakdown (short / PE) leaders", bool(saved.ld_shorts))
        ld_strike = st.selectbox("Option strike suggestion", ["ATM", "ITM-1", "OTM-1"], index=pick(["ATM", "ITM-1", "OTM-1"], saved.ld_strike_pref))
        _wd = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        expiry_weekday = st.selectbox("Monthly stock-option expiry weekday (verify with NSE)", list(range(5)),
                                      index=min(int(saved.expiry_weekday), 4), format_func=lambda i: _wd[i])
    with st.expander("Trend Ignition (open = low / open = high)"):
        ig_window = st.slider("Ignition candle within first N minutes", 1, 60, int(saved.ig_window))
        ig_vol = st.slider("Ignition candle volume vs avg minute (x)", 0.5, 10.0, float(saved.ig_vol), 0.5)
        ig_open_tol = st.slider("Open within % of the candle's low/high", 0.0, 0.5, float(saved.ig_open_tol), 0.01)
        ig_body = st.slider("Min body / range", 0.3, 0.9, float(saved.ig_body), 0.05)
        ig_min_move = st.slider("Min body as % of price", 0.05, 2.0, float(saved.ig_min_move), 0.05)
        ig_max_risk = st.slider("Skip if candle wider than (% of price)", 0.3, 5.0, float(saved.ig_max_risk_pct), 0.1)
    with st.expander("Telegram alerts"):
        tok = st.text_input("Bot token", os.environ.get("TELEGRAM_BOT_TOKEN", ""), type="password")
        chat = st.text_input("Chat ID", os.environ.get("TELEGRAM_CHAT_ID", ""))
        if tok and chat:
            os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"] = tok, chat
        tele = st.checkbox("Send alerts to Telegram", bool(tok and chat))
    cfg = E.Cfg(capital=capital, risk_pct=risk_pct, leverage=leverage, rr=rr, book_pct=book, orb_minutes=orb_minutes,
                confirm_bars=confirm_bars, vol_mult=vol_mult, body_ratio=body_ratio, min_orb_pct=min_orb, max_orb_pct=max_orb,
                last_entry=last_entry, trail_mode=trail_mode, trail_n=trail_n, atr_k=atr_k, eod_exit=eod_exit,
                min_score=min_score, top_k=top_k, fresh_min=fresh_min, min_turnover_cr=min_turn,
                ld_min_price=ld_min_price, ld_min_turnover_cr=ld_min_turn, ld_spurt_vol=ld_spurt, ld_strong_vol=ld_strong,
                ld_explosive_vol=ld_expl, ld_strong_score=ld_strong_score, ld_explosive_score=ld_expl_score, ld_max_age=ld_max_age,
                ld_late_ext=ld_late_ext, ld_shorts=ld_shorts, ld_strike_pref=ld_strike, expiry_weekday=expiry_weekday,
                ig_window=ig_window, ig_vol=ig_vol, ig_open_tol=ig_open_tol, ig_body=ig_body, ig_min_move=ig_min_move,
                ig_max_risk_pct=ig_max_risk)
    if st.button("Save settings for backend monitor", use_container_width=True):
        E.save_cfg(cfg)
        st.success("Saved. monitor.py picks this up automatically.")
    refresh_s = st.select_slider("Live refresh (seconds)", [5, 10, 15, 20, 30, 60], value=15)

# ----------------------------------------------------------------------------- header
now = E.now_ist()
h1, h2, h3, h4 = st.columns([3, 1, 1, 1.4])
h1.markdown("### ORB Command Center")
h2.metric("IST", now.strftime("%H:%M"))
h3.metric("Market", "Open" if E.market_open(now) else "Closed")
h4.metric("Backend", "monitor.py running" if store.monitor_alive() else "Dashboard-managed")

symbols, industry, src = load_uni(universe)
st.caption(f"{src}. Data: yfinance (unofficial, can lag - verify prices on your broker before placing orders).")

(tab_ld, tab_ig, tab_gap, tab_rp, tab_jr, tab_wl, tab_br, tab_sc, tab_desk, tab_tm) = st.tabs(
    ["💥 Breakout Leaders", "⚡ Trend Ignition", "🌅 Pre-Open Gaps", "▶ Day Replay", "📓 Journal",
     "1. Watchlist", "2. Market breadth", "3. Strong-buy scanner", "4. Live desk", "5. Trade manager"])


# ----------------------------------------------------------------------------- chart
def make_chart(ind, orb, sigs, lines, focus, title="", height=650, tf="1m"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.02)
    x = ind.index
    fig.add_trace(go.Candlestick(x=x, open=ind.Open, high=ind.High, low=ind.Low, close=ind.Close, name=tf,
                                 increasing_line_color="#2ec4a6", decreasing_line_color="#ef5b5b"), 1, 1)
    if ind["vwap"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=ind["vwap"], name="VWAP", line=dict(color="#f5b83d", width=1.8)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ind["ema9"], name="EMA9", line=dict(color="#6ea8fe", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ind["ema21"], name="EMA21", line=dict(color="#c792ea", width=1)), 1, 1)
    if orb:
        fig.add_hrect(y0=orb["low"], y1=orb["high"], fillcolor="rgba(110,168,254,0.10)", line_width=0, row=1, col=1)
        for nm, v in (("ORB high", orb["high"]), ("ORB low", orb["low"])):
            fig.add_hline(y=v, line=dict(color="#6ea8fe", dash="dash", width=1), annotation_text=f"{nm} {v:.2f}",
                          annotation_position="left", row=1, col=1)
    for s in sigs:
        pos = ind.index.searchsorted(s["ts"], side="right") - 1
        if pos < 0:
            continue
        up = s["side"] == "LONG"
        bar = ind.iloc[pos]
        fig.add_trace(go.Scatter(x=[ind.index[pos]], y=[bar["Low"] * 0.999 if up else bar["High"] * 1.001],
                                 mode="markers+text", text=[s["side"]], textposition="bottom center" if up else "top center",
                                 marker=dict(symbol="triangle-up" if up else "triangle-down", size=14,
                                             color="#2ec4a6" if up else "#ef5b5b"), name=f"{s['side']} signal",
                                 showlegend=False), 1, 1)
    for nm, (v, col) in lines.items():
        fig.add_hline(y=v, line=dict(color=col, dash="dot", width=1.2), annotation_text=f"{nm} {v:.2f}",
                      annotation_position="right", row=1, col=1)
    colors = np.where(ind.Close >= ind.Open, "#2ec4a6", "#ef5b5b")
    fig.add_trace(go.Bar(x=x, y=ind.Volume, marker_color=colors, name="Volume", showlegend=False), 2, 1)
    lo = min([ind.Low.min()] + focus)
    hi = max([ind.High.max()] + focus)
    pad = (hi - lo) * 0.04
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    fig.update_layout(template="plotly_dark", height=height, title=title, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=90, t=40, b=10), legend=dict(orientation="h", y=1.04))
    return fig


# ----------------------------------------------------------------------------- 1. watchlist
with tab_wl:
    st.markdown("Rank the universe on the last completed session, mark levels, and pin 10-15 names before 9:15.")
    top_n = st.slider("Names per side", 5, 40, 15, key="topn")
    if st.button("Build watchlist", type="primary"):
        with st.spinner(f"Downloading daily charts for {len(symbols)} stocks (first run on Nifty 500 takes a minute or two)..."):
            daily = load_daily(tuple(symbols))
            st.session_state.wl = E.build_watchlist(daily, nifty_ret20(), cfg)
    wl = st.session_state.get("wl")
    if wl is None or wl.empty:
        st.info("Click Build watchlist to score the universe.")
    else:
        st.caption(f"Levels are from the session of {wl['asof'].max()}. Score = trend + momentum + proximity to 20-day extreme + volume + relative strength + tradable volatility.")
        cols = ["symbol", "score", "close", "pdh", "pdl", "pivot", "r1", "s1", "hh20", "atr_pct", "rsi", "vol_x", "rs20"]
        longs = wl.sort_values("long_score", ascending=False).head(top_n).rename(columns={"long_score": "score"})
        shorts = wl.sort_values("short_score", ascending=False).head(top_n).rename(columns={"short_score": "score"})
        st.markdown("**Long bias**")
        st.dataframe(longs[cols].round(2), hide_index=True, use_container_width=True)
        st.markdown("**Short bias**")
        st.dataframe(shorts[cols].rename(columns={"hh20": "hh20"}).round(2), hide_index=True, use_container_width=True)
        options = sorted(wl.symbol)
        default = [s for s in (st.session_state.get("pins") or store.load_watchlist() or list(longs.symbol.head(10))) if s in options]
        pins = st.multiselect("Pinned watchlist", options, default=default)
        if st.button("Save watchlist"):
            store.save_watchlist(pins)
            st.session_state.pins = pins
            st.success(f"Saved {len(pins)} symbols.")

# ----------------------------------------------------------------------------- 2. breadth
with tab_br:
    st.markdown("Check breadth before trusting any ORB. Bearish blocks longs, Bullish blocks shorts.")
    if st.button("Refresh breadth", type="primary"):
        with st.spinner("Computing breadth..."):
            st.session_state.breadth = load_breadth(universe, load_daily(tuple(symbols)), industry)
    b = st.session_state.get("breadth")
    if not b:
        st.info("Click Refresh breadth. The scanner also refreshes it automatically.")
    else:
        msg = (f"**{b['verdict']}** (score {b['score']:+d}) - long ORBs: {'allowed' if b['long_gate'] else 'blocked'}, "
               f"short ORBs: {'allowed' if b['short_gate'] else 'blocked'}. Data as of {b['asof']}.")
        {"Bullish": st.success, "Bearish": st.error}.get(b["verdict"], st.warning)(msg)
        c = st.columns(5)
        c[0].metric("Advance / Decline", f"{b['adv']} / {b['dec']}")
        c[1].metric("Above EMA20", f"{b['pct_above20']:.0f}%")
        c[2].metric("Nifty 50 above VWAP", "n/a" if np.isnan(b["n50_above_vwap"]) else f"{b['n50_above_vwap']:.0f}%")
        c[3].metric("20-day highs / lows", f"{b['new_highs']} / {b['new_lows']}")
        c[4].metric("Up-volume share", f"{b['up_vol_pct']:.0f}%")
        for n in b["notes"]:
            st.write("-", n)
        if len(b["index"]):
            st.dataframe(b["index"], hide_index=True, use_container_width=True)
        sec = b["sector"]
        if len(sec) > 1:
            fig = go.Figure(go.Bar(x=sec.avg_chg, y=sec.industry, orientation="h",
                                   marker_color=np.where(sec.avg_chg >= 0, "#2ec4a6", "#ef5b5b")))
            fig.update_layout(template="plotly_dark", height=max(300, 22 * len(sec)), title="Average change by industry (%)",
                              yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- 3. scanner
with tab_sc:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    c1.button("Run scan now", type="primary", on_click=lambda: st.session_state.update(scan_now=True))
    auto = c2.toggle("Auto-scan", False)
    every = c3.number_input("Every (sec)", 30, 900, 120, 30)
    force = c4.checkbox("Alert even when market is closed (testing)", False)

    def scan_once():
        daily = load_daily(tuple(symbols))
        b = load_breadth(universe, daily, industry)
        st.session_state.breadth = b
        res = E.run_scan(symbols, industry, cfg, daily=daily, nifty_ret20=nifty_ret20(), breadth=b)
        st.session_state.scan = res
        return res

    def fire_alerts(res):
        if res is None or res.empty or not (force or E.market_open()):
            return
        seen = st.session_state.setdefault("alerted", set())
        new = [r for _, r in res[res.status == "STRONG BUY"].iterrows() if f"{r.symbol}|{r.sig_time}" not in seen]
        for r in new:
            seen.add(f"{r.symbol}|{r.sig_time}")
            text = E.strong_buy_message(r)
            st.toast(text, icon="🚀")
            if tele:
                E.notify(text)
        if new:
            beep()

    def scanner_panel():
        if auto or st.session_state.pop("scan_now", False):
            with st.spinner("Scanning: daily structure, then 1-minute ORB confirmation..."):
                res = scan_once()
            fire_alerts(res)
        res = st.session_state.get("scan")
        b = st.session_state.get("breadth")
        if b:
            st.caption(f"Breadth: {b['verdict']} ({b['score']:+d}). Long signals are {'allowed' if b['long_gate'] else 'blocked'}.")
        if res is None:
            st.info("Run a scan. Stage 1 ranks the whole universe on daily structure; stage 2 checks the top names on 1-minute candles.")
            return
        if res.empty:
            st.warning("No results - check the market is open, or that yfinance returned data.")
            return
        n = int((res.status == "STRONG BUY").sum())
        (st.success if n else st.info)(f"{n} strong buy | {len(res)} names checked on 1-minute data | last scan {E.now_ist().strftime('%H:%M:%S')}")

        icons = {"STRONG BUY": "🟢 STRONG BUY", "BUY (late/weak)": "🟡 BUY (late/weak)",
                 "NEAR ORB HIGH": "🔵 NEAR ORB HIGH", "WATCH": "⚪ WATCH"}
        show = res.copy()
        show["status"] = show["status"].map(icons).fillna(show["status"])
        st.caption("🟢 Strong buy · 🟡 Signal exists but late or weak · 🔵 Near ORB high, waiting for breakout · ⚪ Watch only")
        st.dataframe(show, hide_index=True, use_container_width=True, height=560,
                     column_order=["symbol", "status", "score", "price", "day_chg", "entry", "sl", "t1", "t2", "t3", "qty",
                                   "risk_amt", "rvol", "sig_time", "age_min", "vwap", "orb_high", "orb_low",
                                   "daily_score", "intraday_score", "industry"])

    st.fragment(run_every=int(every) if auto else None)(scanner_panel)()

# ----------------------------------------------------------------------------- 4. live desk
VIEWS = {"1m": ["1m"], "3m": ["3m"], "5m": ["5m"], "15m": ["15m"],
         "1m + 5m": ["1m", "5m"], "5m + 15m": ["5m", "15m"], "1m + 5m + 15m": ["1m", "5m", "15m"],
         "All 4": ["1m", "3m", "5m", "15m"]}
TREND_ICON = {"Bullish": "🟢", "Bearish": "🔴", "Mixed": "🟡"}

with tab_desk:
    scan_res = st.session_state.get("scan")
    strong = list(scan_res[scan_res.status == "STRONG BUY"].symbol) if isinstance(scan_res, pd.DataFrame) and not scan_res.empty else []
    pins = st.session_state.get("pins") or store.load_watchlist()
    opts = list(dict.fromkeys(strong + list(pins) + symbols))
    d1, d2, d3 = st.columns([2, 3, 1])
    sym = d1.selectbox("Symbol", opts, help="Strong buys first, then pinned watchlist, then the whole universe.")
    view = d2.radio("Timeframes", list(VIEWS), index=4, horizontal=True,
                    help="Entries are confirmed on 1-minute closes. Higher timeframes give trend context.")
    live = d3.toggle("Live refresh", E.market_open())

    def desk_body(sym, view):
        df = load_1m(sym)
        if df is None or df.empty:
            st.warning("No 1-minute data for this symbol right now.")
            return
        ind = E.add_intraday_indicators(E.session_df(df))          # signals always come from 1-minute candles
        orb = E.orb_levels(ind, cfg)
        sigs = E.find_signals(ind, orb, cfg)

        frames = {"1m": ind}
        raw5 = load_1m_5d(sym)
        for name, mins in E.TIMEFRAMES.items():
            if name == "1m":
                continue
            try:
                src = raw5 if raw5 is not None and not raw5.empty else df
                frames[name] = E.tf_frame(src, mins)
            except Exception:
                pass
        states = E.mtf_state(frames)

        b = st.session_state.get("breadth")
        trade = store.open_for(sym)
        sig = sigs[-1] if sigs else None
        lines, focus = {}, []
        if trade:
            lines = {"Entry": (trade["entry"], "#6ea8fe"), "SL": (trade["sl"], "#ef5b5b"),
                     **{f"T{i + 1}": (v, "#2ec4a6") for i, v in enumerate(trade["targets"])}}
            focus = [trade["sl"], trade["targets"][0], trade["entry"]]
        elif sig:
            lines = {"Entry": (sig["entry"], "#6ea8fe"), "SL": (sig["sl"], "#ef5b5b"),
                     **{f"T{i + 1}": (v, "#2ec4a6") for i, v in enumerate(sig["targets"])}}
            focus = [sig["sl"], sig["targets"][0], sig["entry"]]

        # trend strip across timeframes
        if states:
            cols = st.columns(len(states))
            for c, stt in zip(cols, states):
                c.metric(f"{stt['tf']} trend", f"{TREND_ICON[stt['trend']]} {stt['trend']}", f"RSI {stt['rsi']:.0f}", delta_color="off")
            st.caption("Trend = price vs VWAP and EMA9 vs EMA21 on that timeframe. Bullish needs both up, Bearish needs both down.")

        chart, side = st.columns([3, 1.1])
        tfs = [t for t in VIEWS[view] if t in frames]

        def show(container, tf, height):
            f = frames[tf]
            f = f[f.index.date == f.index[-1].date()]                # display latest session only
            container.plotly_chart(make_chart(f, orb, sigs, lines, focus, f"{sym}  -  {tf}", height=height, tf=tf),
                                   use_container_width=True, key=f"chart_{sym}_{tf}_{view}")

        with chart:
            if len(tfs) == 1:
                show(chart, tfs[0], 650)
            elif len(tfs) == 4:
                for row in (tfs[:2], tfs[2:]):
                    cc = st.columns(2)
                    for c, tf in zip(cc, row):
                        show(c, tf, 380)
            else:
                for tf in tfs:
                    show(chart, tf, 430 if len(tfs) == 2 else 340)

        last = ind.iloc[-1]
        with side:
            st.metric("LTP", f"{last.Close:.2f}", f"{(last.Close / ind.Open.iloc[0] - 1) * 100:+.2f}% from open")
            st.write(f"VWAP **{last.vwap:.2f}** ({'above' if last.Close > last.vwap else 'below'})" if pd.notna(last.vwap) else "VWAP n/a")
            if orb is None:
                st.info("Waiting for the opening range.")
            elif not orb["formed"]:
                st.info(f"ORB forming until {9 + (15 + cfg.orb_minutes) // 60:02d}:{(15 + cfg.orb_minutes) % 60:02d}. Range so far {orb['low']:.2f} - {orb['high']:.2f}.")
            elif not orb["range_ok"]:
                st.warning(f"ORB range {orb['range_pct']:.2f}% is outside your tradable band. Skip.")
            else:
                st.write(f"ORB **{orb['low']:.2f} - {orb['high']:.2f}** ({orb['range_pct']:.2f}%)")
            if trade:
                un, net = E.trade_pnl(trade)
                st.success(f"Tracking {trade['side']} | SL {trade['sl']} | stage T{trade['stage']} | net ₹{net:,.0f}")
            elif sig:
                gate_ok = (b is None) or (b["long_gate"] if sig["side"] == "LONG" else b["short_gate"])
                (st.success if gate_ok else st.warning)(
                    f"{sig['side']} confirmed {sig['ts'].strftime('%H:%M')} (volume {sig['vol_ratio']}x)"
                    + ("" if gate_ok else " - blocked by market breadth"))
                n_ok, n_tot = E.mtf_agrees(states, sig["side"])
                if n_tot:
                    (st.success if n_ok == n_tot else st.warning)(
                        f"5m and 15m trend: {n_ok}/{n_tot} agree with this {sig['side'].lower()}"
                        + ("" if n_ok == n_tot else " - higher timeframe disagrees, size down or skip"))
                st.table(pd.DataFrame({"Level": ["Entry", "Stop loss", "T1", "T2", "T3", "ORB projection", "Qty", "Risk ₹", "Capital used ₹"],
                                       "Value": [sig["entry"], sig["sl"], *sig["targets"], sig["orb_target"], sig["qty"],
                                                 f"{sig['risk_amt']:,.0f}", f"{sig['capital_used']:,.0f}"]}).astype(str))
                if sig["sl_capped"]:
                    st.caption("Stop was capped to the ORB width. Size accordingly.")
                if st.button(f"Track this {sig['side'].lower()} trade", key=f"track_{sym}"):
                    store.add_trade(E.new_trade(sym, sig, cfg))
                    st.toast("Trade added - see the Trade manager tab.", icon="✅")
            elif orb and orb["formed"] and orb["range_ok"]:
                st.info(f"Inside the range. Long trigger above {orb['high']:.2f}, short below {orb['low']:.2f}, on a 1-minute close.")

    st.fragment(run_every=refresh_s if live else None)(desk_body)(sym, view)

# ----------------------------------------------------------------------------- 5. trade manager
with tab_tm:
    with st.expander("Add a trade manually"):
        m1, m2, m3, m4, m5 = st.columns(5)
        ms = m1.text_input("Symbol", "RELIANCE")
        mside = m2.selectbox("Side", ["LONG", "SHORT"])
        me = m3.number_input("Entry", 0.0, 1e6, 0.0, 0.05)
        msl = m4.number_input("Stop loss", 0.0, 1e6, 0.0, 0.05)
        if m5.button("Add"):
            if me > 0 and msl > 0 and me != msl and ((mside == "LONG") == (msl < me)):
                store.add_trade(E.manual_trade(ms.strip().upper(), mside, me, msl, cfg))
                st.success("Added.")
            else:
                st.error("Stop must be below entry for LONG, above entry for SHORT.")

    def trades_panel():
        trades = store.open_trades()
        if trades and not store.monitor_alive():        # dashboard manages trades only if monitor.py is not running
            data = E.download_batch(sorted({t["symbol"] for t in trades}), "1d", "1m", chunk=40)
            for t in trades:
                t, events = E.manage_trade(t, data.get(t["symbol"]))
                store.update_trade(t)
                for e in events:
                    store.add_event(e)
                    st.toast(e)
                    if tele:
                        E.notify(e)
            trades = store.open_trades()
        st.markdown(f"**Open trades: {len(trades)}**  ·  "
                    + ("managed by monitor.py" if store.monitor_alive() else "managed by this dashboard while it is open (run monitor.py to keep going after you close it)"))
        if trades:
            rows = []
            for t in trades:
                un, net = E.trade_pnl(t)
                rows.append({"ID": t["id"], "Symbol": t["symbol"], "Side": t["side"], "Entry": t["entry"], "LTP": round(t["last_price"], 2),
                             "SL": t["sl"], "Stage": "Initial SL" if t["stage"] == 0 else f"T{t['stage']} done",
                             "T1": t["targets"][0], "T2": t["targets"][1], "T3": t["targets"][2],
                             "Open qty": t["qty_open"], "Booked ₹": round(t["realized"]), "Open ₹": round(un), "Net ₹": round(net),
                             "R": round(net / (t["risk_ps"] * t["qty"]), 2) if t["risk_ps"] and t["qty"] else 0})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            x1, x2 = st.columns([2, 1])
            pick_id = x1.selectbox("Exit at last price", [t["id"] for t in trades], format_func=lambda i: next(f"#{t['id']} {t['symbol']}" for t in trades if t["id"] == i))
            if x2.button("Exit now"):
                t = next(t for t in trades if t["id"] == pick_id)
                t, ev = E.close_trade(t, t["last_price"], "MANUAL")
                store.update_trade(t)
                for e in ev:
                    store.add_event(e)
                st.rerun()
        ev = store.recent_events(25)
        if ev:
            st.markdown("**Event log**")
            for ts, text in ev:
                st.write(f"`{ts}`  {text}")

    st.fragment(run_every=refresh_s)(trades_panel)()

    hist = store.closed_trades()
    if hist:
        st.markdown("**Closed trades**")
        hd = pd.DataFrame([{"ID": t["id"], "Symbol": t["symbol"], "Side": t["side"], "Entry": t["entry"], "Exit": t["exit_price"],
                            "Reason": t["close_reason"], "P&L ₹": t["pnl"], "R": t.get("r_multiple", 0)} for t in hist])
        k1, k2, k3 = st.columns(3)
        k1.metric("Net P&L", f"₹{hd['P&L ₹'].sum():,.0f}")
        k2.metric("Win rate", f"{(hd['P&L ₹'] > 0).mean() * 100:.0f}%")
        k3.metric("Avg R", f"{hd['R'].mean():.2f}")
        st.dataframe(hd, hide_index=True, use_container_width=True)


# =============================================================================================================
# NEW: Breakout Leaders, Option strikes, Market mood + alerts, Trend Ignition, Pre-Open Gaps, Day Replay, Journal
# =============================================================================================================
fno_syms, fno_lots, fno_steps, fno_note = load_fno()
fno_key = tuple(fno_syms)
MOOD_COLOR = {"Bullish": "#2ec4a6", "Bearish": "#ef5b5b", "Neutral": "#f5b83d", "Unknown": "#8a8f98"}


def _n(x, default=None):
    """NaN / None safe value."""
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else x


def mood_strip(b):
    """Advance/decline breadth in one look."""
    html_, h = U.breadth_html(b)
    components.html(html_, height=h, scrolling=True)


# ----------------------------------------------------------------------------- 1-3. breakout leaders
def run_leader_scan():
    daily = load_fno_daily(fno_key)
    data = load_fno_1m(fno_key)
    b = load_breadth("F&O", daily, {s: "NA" for s in fno_syms})
    board, stats = L.scan_leaders(fno_syms, cfg, daily, data, nifty_ret20(), b, fno_lots, fno_steps, industry)
    st.session_state.update(ld_board=board, ld_stats=stats, ld_breadth=b, ld_time=E.now_ist())
    return board


def fire_leader_alerts(board, force):
    """Telegram + toast + beep when a NEW, fresh, mood-compatible EXPLOSIVE leader appears."""
    if board is None or board.empty or not (force or E.market_open()):
        return
    seen = st.session_state.setdefault("ld_alerted", set())
    done = set(store.get_meta("ld_alerted", []) or [])          # shared with monitor.py so nothing is sent twice
    day, new = E.now_ist().date().isoformat(), False
    for r in board[(board.tier == "EXPLOSIVE") & board.mood_ok & (board.age_min <= cfg.fresh_min)].to_dict("records"):
        k = f"{day}|{r['symbol']}|{r['side']}|{r['tier']}"
        if k in seen or k in done:
            continue
        seen.add(k)
        done.add(k)
        new = True
        text = L.leader_message(r)
        st.toast(text, icon="💥")
        store.add_event(f"💥 EXPLOSIVE {r['symbol']} {r['side']} score {r['score']:.0f}")
        if tele and not store.monitor_alive():
            E.notify(text)
    store.set_meta("ld_alerted", sorted(done)[-500:])
    if new:
        beep()


with tab_ld:
    st.caption(f"{fno_note}. Option premiums are Black-Scholes estimates from recent realised volatility (yfinance has no NSE option chain) - "
               "read the live premium and exact strike list from your broker.")
    a1, a2, a3, a4 = st.columns([1, 1.6, 1, 1.6])
    a1.button("Scan now", type="primary", key="ld_now_btn", on_click=lambda: st.session_state.update(ld_now=True))
    ld_auto = a2.toggle("Live auto-scan (starts 09:16)", E.market_open(), key="ld_auto")
    ld_every = a3.number_input("Every (sec)", 15, 300, 30, 15, key="ld_every")
    ld_force = a4.checkbox("Test mode: ignore market hours", False, key="ld_force")
    ld_max = st.slider("Cards per tier", 3, 20, 8, key="ld_max")

    def leaders_panel():
        after_916 = E.now_ist().hour * 60 + E.now_ist().minute >= 556
        active = ld_auto and (ld_force or (E.market_open() and after_916))
        if active or st.session_state.pop("ld_now", False):
            with st.spinner(f"Scanning {len(fno_syms)} F&O stocks..."):
                board = run_leader_scan()
            fire_leader_alerts(board, ld_force)
        board, stt, b = st.session_state.get("ld_board"), st.session_state.get("ld_stats"), st.session_state.get("ld_breadth")
        mood_strip(b)
        if board is None:
            st.info("Click **Scan now** (or switch on Live auto-scan). The first scan downloads 1-minute data for the F&O universe, which takes a little while.")
            return
        counts = {t: int((board.tier == t).sum()) for t in ("SPURT", "STRONG", "EXPLOSIVE")} if not board.empty else {t: 0 for t in ("SPURT", "STRONG", "EXPLOSIVE")}
        tb_html, tb_h = U.leaders_toolbar_html(stt, b, int(ld_every), ld_auto, st.session_state["ld_time"].strftime("%H:%M:%S"),
                                               counts, cfg.ld_spurt_vol, cfg.confirm_bars * 4 + 3)
        components.html(tb_html, height=tb_h, scrolling=True)
        if board.empty:
            st.warning("No leaders right now. Either nothing has broken out cleanly, or the market is closed / data is missing.")
            return
        bd_html, bd_h = U.leaders_board_html(board, ld_max)
        components.html(bd_html, height=bd_h, scrolling=True)
        with st.expander("Track a leader as a trade"):
            pick = st.selectbox("Symbol", board.symbol.tolist(), key="ld_track_pick")
            if st.button("Track trade", key="ld_track_btn"):
                r = board[board.symbol == pick].iloc[0].to_dict()
                store.add_trade(E.new_trade(r["symbol"], L.sig_from_row(r), cfg))
                st.toast(f"{r['symbol']} added to Trade manager.", icon="✅")

    st.fragment(run_every=int(ld_every) if ld_auto else None)(leaders_panel)()


# ----------------------------------------------------------------------------- 4. trend ignition
with tab_ig:
    st.markdown("**Catch the trend the minute it is born.** Stocks that do not break a range - they open *at* the day's low (or high) "
                "on a strong, high-volume candle and never look back. Entry is printed from that very candle; a structure trail shows where the trade exits.")
    i1, i2, i3 = st.columns([1, 1.6, 1])
    i1.button("Scan now", type="primary", key="ig_now_btn", on_click=lambda: st.session_state.update(ig_now=True))
    ig_auto = i2.toggle("Live auto-scan", E.market_open(), key="ig_auto")
    ig_every = i3.number_input("Every (sec)", 15, 300, 30, 15, key="ig_every")

    def ignition_panel():
        if ig_auto or st.session_state.pop("ig_now", False):
            with st.spinner("Looking for open-at-low / open-at-high candles..."):
                st.session_state.ig = L.ignition_scan(fno_syms, cfg, load_fno_daily(fno_key), load_fno_1m(fno_key), nifty_ret20())
                st.session_state.ig_time = E.now_ist()
        res = st.session_state.get("ig")
        if res is None:
            st.info("Click **Scan now**. Detections start from the 09:15 candle - newest always on top.")
            return
        if res.empty:
            st.warning("No ignition candles today (yet). They only form in the first minutes after the open.")
            return
        k = st.columns(4)
        k[0].metric("Detections", len(res))
        k[1].metric("Still running", int((res.status == "RUNNING").sum()))
        k[2].metric("Exited", int(res.status.str.startswith("EXITED").sum()))
        k[3].metric("Average P&L", f"{res.pnl_pct.mean():+.2f}%")
        best = res.sort_values("pnl_pct").iloc[-1]
        ex = f"EXIT ₹{best['exit']}" if _n(best["exit"]) else f"LTP ₹{best['last']}"
        st.success(f"Best so far: **{best.symbol}** found at {best.detect} → {'BUY' if best.side == 'LONG' else 'SELL'} ₹{best.entry}, "
                   f"{ex} → **{best.pnl_pct:+.2f}%**")
        ig_html, ig_h = U.ignition_html(res)
        components.html(ig_html, height=ig_h, scrolling=True)
        st.caption("Entry = close of the detection candle. Trail = lowest low (highest high for shorts) of the last "
                   f"{cfg.trail_n} bars. Simulated on 1-minute closes; real fills will differ. Last scan "
                   f"{st.session_state['ig_time'].strftime('%H:%M:%S')}.")
        with st.expander("Full table"):
            show = res.copy()
            show["side"] = show.side.map({"LONG": "▲ BUY", "SHORT": "▼ SELL"})
            st.dataframe(show, hide_index=True, use_container_width=True, height=420,
                         column_order=["detect", "symbol", "side", "entry", "init_sl", "sl", "status", "exit", "exit_time", "last",
                                       "pnl_pct", "mfe_pct", "r_mult", "vol_x"],
                         column_config={"detect": "Detected", "init_sl": "Initial SL", "sl": "Trail SL now", "exit": "Exit ₹",
                                        "exit_time": "Exit time", "last": "LTP", "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f"),
                                        "mfe_pct": st.column_config.NumberColumn("Best %", format="%.2f"),
                                        "r_mult": st.column_config.NumberColumn("R", format="%.1f"), "vol_x": "Vol x"})

    st.fragment(run_every=int(ig_every) if ig_auto else None)(ignition_panel)()


# ----------------------------------------------------------------------------- 5. pre-open gaps
with tab_gap:
    st.markdown("**Read the auction before the first candle.** The exchange's 09:00-09:15 pre-open auction sets the opening price; "
                "the gap below is that open versus the previous close and is frozen at the bell. Trap flags mark where the crowd gets caught.")
    g1, g2 = st.columns([1, 1])
    gap_min = g1.slider("List gaps of at least (%)", 0.1, 3.0, 0.3, 0.1)
    if g2.button("Load / refresh gaps", type="primary"):
        load_fno_daily.clear()
        st.session_state.gaps_on = True
    if st.session_state.get("gaps_on"):
        with st.spinner("Loading daily data for the F&O universe..."):
            dly = load_fno_daily(fno_key)
        gt, asof = L.gap_table(dly, fno_syms, cfg, nifty_ret20(), None, gap_min, st.session_state.get("gaps_preopen"))
        today_iso = E.now_ist().date().isoformat()
        if gt.empty:
            st.warning("No gap data available.")
        else:
            if asof != today_iso:
                st.warning(f"Latest daily bar is {asof}, not today - the market has not opened yet (or is on holiday), so this shows that session's gaps.")
            elif E.now_ist().hour * 60 + E.now_ist().minute >= 555 and L.freeze_gaps(gt, asof):
                st.toast("Opening gaps frozen for today.", icon="🌅")
                if tele and not store.monitor_alive():
                    E.notify(L.gap_summary_message(gt, asof))
            snap = store.get_meta(f"gaps:{asof}")
            if snap:
                st.caption(f"Frozen snapshot saved {snap['saved'][:16].replace('T', ' ')} IST for session {asof}.")
            po = st.session_state.get("gaps_preopen") or {}
            gp_html, gp_h = U.gaps_html(gt, gap_min, E.now_ist().strftime("%H:%M"), asof, has_bs=bool(po))
            components.html(gp_html, height=gp_h, scrolling=True)
            with st.expander("Full table"):
                sel = gt[gt.gap_pct.abs() >= gap_min]
                cc = ["symbol", "gap_pct", "prev_close", "open", "ltp", "live_pct", "from_open_pct", "trap", "trap_state", "reason"]
                cfgc = {"gap_pct": st.column_config.NumberColumn("Gap % (frozen)", format="%.2f"),
                        "live_pct": st.column_config.NumberColumn("Live %", format="%.2f"),
                        "from_open_pct": st.column_config.NumberColumn("From open %", format="%.2f"),
                        "prev_close": "Prev close", "ltp": "LTP", "trap": "Trap flag", "trap_state": "State", "reason": "Why"}
                st.dataframe(sel.sort_values("gap_pct", ascending=False)[cc], hide_index=True, use_container_width=True, column_config=cfgc)
            st.caption("LTP / live % come from yfinance daily data (cached up to 10 min - press Load / refresh to update).")
    else:
        st.info("Click **Load / refresh gaps** after 09:15 (the open is frozen at the bell). Before the open you will see the previous session.")
    with st.expander("Optional: NSE indicative pre-open prices (09:00-09:08)"):
        st.caption("Experimental and untested against live NSE - NSE often blocks scripted requests. If it fails, the 09:15 open above is what you get.")
        if st.button("Try NSE pre-open feed"):
            po = L.fetch_nse_preopen()
            if not po:
                st.warning("NSE did not return pre-open data (blocked, or outside the auction window).")
            else:
                st.session_state.gaps_preopen = po
                pdf = pd.DataFrame([{"symbol": k, **v} for k, v in po.items() if k in set(fno_syms)]).sort_values("pct", ascending=False)
                st.dataframe(pdf.round(2), hide_index=True, use_container_width=True)
                st.success("B/S chip will now show on the gap table above (re-open this tab / click Load again).")


# ----------------------------------------------------------------------------- 7. day replay
def mini_chart(c):
    long = c["side"] == "LONG"
    col = "#2ec4a6" if long else "#ef5b5b"
    xs = [L.hhmm(m) for m in c["m"]]
    y = c["series"]
    fig = go.Figure(go.Scatter(x=xs, y=y, mode="lines", line=dict(width=1.8, color=col), showlegend=False))
    for v in (c["orb_high"], c["orb_low"]):
        fig.add_hline(y=v, line=dict(color="#6ea8fe", dash="dot", width=1))
    if c["first_m"] in c["m"]:
        i = c["m"].index(c["first_m"])
        fig.add_trace(go.Scatter(x=[xs[i]], y=[y[i]], mode="markers", showlegend=False,
                                 marker=dict(symbol="triangle-up" if long else "triangle-down", size=11, color=col)))
    lo, hi = min(y + [c["orb_low"]]), max(y + [c["orb_high"]])
    pad = (hi - lo) * 0.08 or 1
    fig.update_layout(template="plotly_dark", height=150, margin=dict(l=2, r=2, t=2, b=2), showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False, range=[lo - pad, hi + pad]))
    return fig


with tab_rp:
    st.markdown("**Rewatch the day exactly as it ran live.** Every leader pops back in at the exact minute it fired, with the badge and mini chart it had "
                "*then*. Signals are recomputed bar by bar from data available at that minute (no hindsight), so yesterday becomes a zero-risk practice ground.")
    _days, _d = [], E.now_ist().date()
    while len(_days) < 6:
        if _d.weekday() < 5:
            _days.append(_d.isoformat())
        _d -= timedelta(days=1)
    r1, r2, r3 = st.columns([1.2, 1.6, 1.2])
    build_day = r1.selectbox("Session to build", _days, help="yfinance keeps about 7 days of 1-minute history.")
    if r2.button("Build replay pack (downloads F&O 1-minute data)", type="primary"):
        with st.spinner("Downloading 1-minute bars and replaying the session bar by bar..."):
            _data = load_fno_5d(fno_key)
            _day = _date.fromisoformat(build_day)
            if not any((df.index.date == _day).any() for df in _data.values()):
                st.error("No 1-minute data for that date (holiday, or outside yfinance's 1-minute window).")
            else:
                _pk = L.build_pack(_data, load_fno_daily(fno_key), fno_syms, cfg, _day, nifty_ret20(), fno_lots, fno_steps)
                st.session_state["rp_pick"] = build_day
                st.session_state["rp_t"] = 570
                st.success(f"Pack ready: {len(_pk['leaders'])} leaders.")
    packs = sorted(L.pack_dates(), reverse=True)
    if not packs:
        st.info("No pack yet. Build one for a recent session - or run `python monitor.py --leaders`, which saves today's pack automatically after the close.")
    else:
        pick_day = r3.selectbox("Saved pack", packs, key="rp_pick")
        pack = L.load_pack(pick_day)
        st.caption(f"Pack for **{pack['date']}**: {len(pack['leaders'])} leaders · {len(pack['leaders'])} mini charts · arrivals minute-exact "
                   f"· built {pack['built'][:16].replace('T', ' ')}")
        MINS = list(range(555, 930))
        st.session_state.setdefault("rp_t", 570)
        st.session_state.setdefault("rp_running", False)
        b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 2.2, 1.4])
        b1.button("▶ Play", key="rp_play", on_click=lambda: st.session_state.update(rp_running=True))
        b2.button("⏸ Pause", key="rp_pause", on_click=lambda: st.session_state.update(rp_running=False))
        b3.button("⏮ Restart", key="rp_restart", on_click=lambda: st.session_state.update(rp_t=555, rp_running=False))
        rp_speed = b4.select_slider("Speed (market minutes per second)", [1, 2, 5, 10, 15], value=5, key="rp_speed")
        rp_n = b5.slider("Cards shown", 3, 24, 9, key="rp_n")
        _running = bool(st.session_state.get("rp_running"))

        def replay_panel(pack):
            finished = False
            if st.session_state.get("rp_running"):
                nt = st.session_state.rp_t + int(st.session_state.get("rp_speed", 5))
                if nt >= MINS[-1]:
                    nt, finished = MINS[-1], True
                    st.session_state.rp_running = False
                st.session_state.rp_t = nt
            t = st.select_slider("Replay clock", MINS, key="rp_t", format_func=L.hhmm)
            cards = L.replay_cards(pack, t)
            n_charts = sum(1 for c in cards if len(c["series"]) > 1)
            st.markdown(f"**▶ Day Replay — the day as it ran live**  \nPacked {pack['built'][11:16]} · {len(pack['leaders'])} leaders · {n_charts} charts")
            rp_html, rp_h = U.replay_html(cards, per_col=rp_n, step=max(int(st.session_state.get("rp_speed", 5)) // 2, 1))
            components.html(rp_html, height=rp_h, scrolling=True)
            with st.expander("Arrivals feed", expanded=False):
                for m, txt in L.replay_feed(pack, t, 15):
                    st.write(f"`{L.hhmm(m)}`  {txt}")
            if finished:
                st.rerun()

        st.fragment(run_every=1 if _running else None)(replay_panel)(pack)


# ----------------------------------------------------------------------------- 6. journal
with tab_jr:
    st.markdown("**Every trade logged, every lesson remembered.** Log a trade in seconds; stats, equity curve, calendar and insights build themselves.")
    jkey = st.text_input("Access key", type="password", key="jr_key",
                         help="Your journal is filed under a hash of this key and is never shown under any other key. "
                              "To use the same journal on phone and PC, open this same app (one server) on both and enter the same key.")
    if not jkey.strip():
        st.info("Enter an access key to open your private journal (choose anything you will remember - there is no recovery).")
    else:
        owner = J.owner_hash(jkey)
        with st.expander("➕ Log a trade", expanded=True):
            c = st.columns(5)
            j_date = c[0].date_input("Date", E.now_ist().date(), key="j_date")
            j_sym = c[1].text_input("Symbol", key="j_sym").strip().upper()
            j_inst = c[2].radio("Instrument", ["Equity", "Option"], horizontal=True, key="j_inst")
            j_side = c[3].selectbox("Side", ["LONG", "SHORT"], key="j_side", help="For options, LONG = you bought the premium.")
            j_setup = c[4].selectbox("Setup", J.SETUPS, key="j_setup")
            c = st.columns(5)
            j_entry = c[0].number_input("Entry", 0.0, 1e7, 0.0, 0.05, key="j_entry")
            j_sl = c[1].number_input("Stop-loss", 0.0, 1e7, 0.0, 0.05, key="j_sl")
            j_tgt = c[2].number_input("Target", 0.0, 1e7, 0.0, 0.05, key="j_tgt")
            j_exit = c[3].number_input("Exit", 0.0, 1e7, 0.0, 0.05, key="j_exit")
            j_charges = c[4].number_input("Charges ₹", 0.0, 1e6, 0.0, 10.0, key="j_charges")
            c = st.columns(5)
            lot_known = _n(fno_lots.get(j_sym))
            if j_inst == "Option":
                j_strike = c[0].text_input("Option strike", placeholder="e.g. 2400 CE", key="j_strike")
                j_lots = c[1].number_input("Lots", 1, 100000, 1, key="j_lots")
                if lot_known:
                    lot_size = int(lot_known)
                    c[2].markdown(f"Lot size **{lot_size}** (auto)")
                else:
                    lot_size = int(c[2].number_input("Lot size", 1, 1000000, 1, key="j_lotsize",
                                                     help="Not found for this symbol - enter it, or add it to cache/fno_lots.csv."))
                j_qty = int(j_lots) * lot_size
                c[3].markdown(f"Quantity **{j_qty:,}**  \n= {j_lots} lot × {lot_size}")
            else:
                j_strike, j_lots, lot_size = "", 0, 0
                j_qty = int(c[0].number_input("Quantity", 1, 10_000_000, 1, key="j_qty"))
            j_tin = c[4].time_input("Entry time", _time(9, 30), key="j_tin")
            c = st.columns([1, 1, 3])
            j_tout = c[0].time_input("Exit time", _time(10, 0), key="j_tout")
            j_mist = c[1].selectbox("Mistake (if any)", J.MISTAKES, key="j_mist")
            j_note = c[2].text_input("Note", key="j_note")
            if st.button("Save trade", type="primary", key="j_save"):
                if not j_sym or j_entry <= 0 or j_exit <= 0 or j_qty <= 0:
                    st.error("Symbol, entry, exit and quantity are required.")
                else:
                    store.add_journal(owner, dict(
                        date=j_date.isoformat(), symbol=j_sym, side=j_side, instrument=j_inst, setup=j_setup, entry=j_entry, sl=j_sl,
                        target=j_tgt, exit=j_exit, qty=j_qty, lots=int(j_lots), lot_size=int(lot_size), strike=j_strike,
                        entry_time=j_tin.strftime("%H:%M"), exit_time=j_tout.strftime("%H:%M"), mistake=j_mist,
                        charges=j_charges, note=j_note))
                    st.toast("Trade logged.", icon="📓")
                    st.rerun()
        if st.button("Import closed trades from the Trade manager"):
            done = set(store.get_meta(f"jr_imported:{owner}", []) or [])
            added = 0
            for t in store.closed_trades(500):
                if t["id"] in done or t.get("exit_price") is None:
                    continue
                oa, ca = pd.Timestamp(t["opened_at"]), pd.Timestamp(t["closed_at"])
                store.add_journal(owner, dict(
                    date=ca.date().isoformat(), symbol=t["symbol"], side=t["side"], instrument="Equity", setup="ORB scanner (strong buy)",
                    entry=t["entry"], sl=t["init_sl"], target=t["targets"][1], exit=t["exit_price"], qty=t["qty"],
                    entry_time=oa.strftime("%H:%M"), exit_time=ca.strftime("%H:%M"), mistake="None", charges=0.0,
                    pnl_override=t["pnl"], note=f"imported trade #{t['id']} ({t['close_reason']})"))
                done.add(t["id"])
                added += 1
            store.set_meta(f"jr_imported:{owner}", sorted(done))
            st.success(f"Imported {added} trade(s).") if added else st.info("Nothing new to import.")
            if added:
                st.rerun()

        jdf = J.to_frame(store.journal_list(owner))
        if jdf.empty:
            st.info("No trades logged yet.")
        else:
            S = J.stats(jdf)
            cards = J.insight_cards(jdf)
            ym = sorted({(d.year, d.month) for d in jdf.date}, reverse=True)
            jc1, jc2 = st.columns([1, 5])
            pick_m = jc1.selectbox("Calendar month", ym, format_func=lambda x: f"{x[0]}-{x[1]:02d}", key="jr_month")
            jr_html, jr_h = U.journal_html(jdf, jdf, S, cards, pick_m[0], pick_m[1], E.now_ist().date())
            components.html(jr_html, height=jr_h, scrolling=True)
            x1, x2, x3 = st.columns([2, 1, 1])
            del_id = x1.selectbox("Delete entry", list(jdf["id"]), format_func=lambda i: f"#{i} " + str(jdf[jdf['id'] == i].iloc[0].symbol))
            if x2.button("Delete"):
                store.journal_delete(owner, int(del_id))
                st.rerun()
            x3.download_button("Export CSV", J.csv_bytes(jdf), "trading_journal.csv", "text/csv")
