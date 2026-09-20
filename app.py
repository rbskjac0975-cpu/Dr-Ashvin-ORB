"""ORB Command Center - run with:  streamlit run app.py"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import engine as E
import store

st.set_page_config(page_title="ORB Command Center", page_icon="📈", layout="wide")
store.init()


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


@st.cache_data(ttl=300, show_spinner=False)
def nifty_ret20():
    return E.index_ret20()


@st.cache_data(ttl=90, show_spinner=False)
def load_breadth(name, _daily, _industry):
    return E.compute_breadth(_daily, _industry)


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
    with st.expander("Telegram alerts"):
        tok = st.text_input("Bot token", os.environ.get("TELEGRAM_BOT_TOKEN", ""), type="password")
        chat = st.text_input("Chat ID", os.environ.get("TELEGRAM_CHAT_ID", ""))
        if tok and chat:
            os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"] = tok, chat
        tele = st.checkbox("Send alerts to Telegram", bool(tok and chat))
    cfg = E.Cfg(capital=capital, risk_pct=risk_pct, leverage=leverage, rr=rr, book_pct=book, orb_minutes=orb_minutes,
                confirm_bars=confirm_bars, vol_mult=vol_mult, body_ratio=body_ratio, min_orb_pct=min_orb, max_orb_pct=max_orb,
                last_entry=last_entry, trail_mode=trail_mode, trail_n=trail_n, atr_k=atr_k, eod_exit=eod_exit,
                min_score=min_score, top_k=top_k, fresh_min=fresh_min, min_turnover_cr=min_turn)
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

tab_wl, tab_br, tab_sc, tab_desk, tab_tm = st.tabs(
    ["1. Watchlist", "2. Market breadth", "3. Strong-buy scanner", "4. Live desk", "5. Trade manager"])


# ----------------------------------------------------------------------------- chart
def make_chart(ind, orb, sigs, lines, focus, title=""):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.02)
    x = ind.index
    fig.add_trace(go.Candlestick(x=x, open=ind.Open, high=ind.High, low=ind.Low, close=ind.Close, name="1m",
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
        up = s["side"] == "LONG"
        fig.add_trace(go.Scatter(x=[s["ts"]], y=[ind.loc[s["ts"], "Low" if up else "High"] * (0.999 if up else 1.001)],
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
    fig.update_layout(template="plotly_dark", height=650, title=title, xaxis_rangeslider_visible=False,
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

        def hl(row):
            col = {"STRONG BUY": "background-color:#c9f2dc;color:#0b3d24;font-weight:600",
                   "BUY (late/weak)": "background-color:#fff0bf;color:#5a4200",
                   "NEAR ORB HIGH": "background-color:#d9e6ff;color:#12305c"}.get(row["status"], "")
            return [col] * len(row)
        st.dataframe(res.style.apply(hl, axis=1), hide_index=True, use_container_width=True, height=560)

    st.fragment(run_every=int(every) if auto else None)(scanner_panel)()

# ----------------------------------------------------------------------------- 4. live desk
with tab_desk:
    scan_res = st.session_state.get("scan")
    strong = list(scan_res[scan_res.status == "STRONG BUY"].symbol) if isinstance(scan_res, pd.DataFrame) and not scan_res.empty else []
    pins = st.session_state.get("pins") or store.load_watchlist()
    opts = list(dict.fromkeys(strong + list(pins) + symbols))
    d1, d2 = st.columns([3, 1])
    sym = d1.selectbox("Symbol", opts, help="Strong buys first, then pinned watchlist, then the whole universe.")
    live = d2.toggle("Live refresh", E.market_open())

    def desk_body(sym):
        df = load_1m(sym)
        if df is None or df.empty:
            st.warning("No 1-minute data for this symbol right now.")
            return
        ind = E.add_intraday_indicators(E.session_df(df))
        orb = E.orb_levels(ind, cfg)
        sigs = E.find_signals(ind, orb, cfg)
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
        chart, side = st.columns([3, 1.1])
        chart.plotly_chart(make_chart(ind, orb, sigs, lines, focus, f"{sym}  -  1 minute"), use_container_width=True)
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

    st.fragment(run_every=refresh_s if live else None)(desk_body)(sym)

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
