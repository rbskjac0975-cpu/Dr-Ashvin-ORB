"""
engine.py - data, indicators, market breadth, ORB + 1-minute confirmation signals,
risk plan, scanner and trade management.

Pure logic (no Streamlit) so the same code powers the dashboard AND the background monitor.
Data source: yfinance (unofficial, may be delayed - always verify against your broker feed).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = "Asia/Kolkata"
TICK = 0.05
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "cache")
CONFIG_FILE = os.path.join(BASE, "config.json")


# --------------------------------------------------------------------------- config
@dataclass
class Cfg:
    # risk
    capital: float = 500000.0
    risk_pct: float = 0.5            # % of capital risked per trade
    leverage: float = 5.0            # max intraday leverage (MIS)
    rr: tuple = (1.0, 2.0, 3.0)      # T1/T2/T3 as R multiples
    book_pct: tuple = (50, 30, 20)   # % of qty booked at T1/T2/T3
    # ORB + confirmation
    orb_minutes: int = 15
    confirm_bars: int = 1            # consecutive 1m closes beyond ORB needed
    vol_mult: float = 1.3            # signal bar volume vs 20-bar average
    body_ratio: float = 0.5          # candle body / range
    min_orb_pct: float = 0.25        # ORB range must be within this band (% of price)
    max_orb_pct: float = 2.0
    last_entry: str = "13:30"
    # trailing / exits
    trail_mode: str = "Structure"    # Structure | EMA9 | ATR | None
    trail_n: int = 5
    atr_k: float = 3.0
    eod_exit: str = "15:15"
    # scanner
    min_score: int = 75
    top_k: int = 40                  # stage-2 (1-min) candidates
    fresh_min: int = 10              # alert only if signal is this fresh
    min_turnover_cr: float = 10.0    # avg daily turnover filter (Rs crore)


def save_cfg(cfg: Cfg) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_cfg() -> Cfg:
    try:
        with open(CONFIG_FILE) as f:
            d = json.load(f)
        d = {k: v for k, v in d.items() if k in Cfg.__dataclass_fields__}
        for k in ("rr", "book_pct"):
            if k in d:
                d[k] = tuple(d[k])
        return Cfg(**d)
    except Exception:
        return Cfg()


# --------------------------------------------------------------------------- time helpers
def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def tmin(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def market_open(now: Optional[pd.Timestamp] = None) -> bool:
    now = now if now is not None else now_ist()
    return now.weekday() < 5 and 555 <= now.hour * 60 + now.minute <= 930  # 09:15-15:30


def rt(x: float) -> float:
    return round(round(float(x) / TICK) * TICK, 2)


# --------------------------------------------------------------------------- universe
NSE_LISTS = {
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
}
NSE_BASES = [
    "https://nsearchives.nseindia.com/content/indices/",
    "https://archives.nseindia.com/content/indices/",
]
# Offline fallback only (approximate Nifty 50). The live NSE list is used whenever reachable.
NIFTY50_FALLBACK = (
    "ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BEL BHARTIARTL "
    "CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO "
    "HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC "
    "ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA TATACONSUMER TATAMOTORS TATASTEEL TCS "
    "TECHM TITAN TRENT ULTRACEMCO WIPRO"
).split()


def load_universe(name: str, max_age_days: int = 7) -> Tuple[List[str], Dict[str, str], str]:
    """Returns (symbols, {symbol: industry}, source note). Cached on disk in ./cache.
    You can also drop the official NSE CSV into ./cache/ manually."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, NSE_LISTS[name])
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path) < max_age_days * 86400)
    if not fresh:
        for base in NSE_BASES:
            try:
                r = requests.get(base + NSE_LISTS[name], headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if r.status_code == 200 and "Symbol" in r.text[:500]:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(r.text)
                    break
            except Exception:
                continue
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "Symbol" in df.columns:
                ind = df["Industry"] if "Industry" in df.columns else pd.Series("NA", index=df.index)
                syms = [str(s).strip() for s in df["Symbol"]]
                return syms, dict(zip(syms, ind)), f"NSE list: {name} ({len(syms)} stocks)"
        except Exception:
            pass
    syms = list(NIFTY50_FALLBACK)
    return syms, {s: "NA" for s in syms}, (
        f"NSE list unreachable -> using built-in Nifty 50 fallback ({len(syms)} stocks). "
        f"Download {NSE_LISTS[name]} from NSE and place it in ./cache/ for the full {name}."
    )


# --------------------------------------------------------------------------- data (yfinance)
def to_yf(sym: str) -> str:
    return sym if sym.startswith("^") or sym.endswith(".NS") else f"{sym}.NS"


def _extract(raw: pd.DataFrame, tkr: str, single: bool) -> Optional[pd.DataFrame]:
    try:
        df = raw
        if isinstance(raw.columns, pd.MultiIndex):
            if tkr in raw.columns.get_level_values(0):
                df = raw[tkr]
            elif tkr in raw.columns.get_level_values(1):
                df = raw.xs(tkr, axis=1, level=1)
            elif single:
                df = raw.droplevel(1, axis=1)
            else:
                return None
        need = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in need):
            return None
        df = df[need].dropna(subset=["Close"])
        return df if len(df) else None
    except Exception:
        return None


def _fix_index(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    if intraday:
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        df.index = idx.tz_convert(IST)
    else:
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        df.index = idx.normalize()
    return df


def download_batch(symbols: List[str], period: str, interval: str, chunk: int = 80) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    intraday = interval.endswith("m")
    symbols = list(dict.fromkeys(symbols))
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        yfs = [to_yf(s) for s in part]
        raw = None
        for _ in range(2):
            try:
                raw = yf.download(yfs, period=period, interval=interval, group_by="ticker",
                                  auto_adjust=False, threads=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                time.sleep(1)
        if raw is None or raw.empty:
            continue
        for s, y in zip(part, yfs):
            df = _extract(raw, y, len(yfs) == 1)
            if df is not None:
                out[s] = _fix_index(df, intraday)
    return out


def session_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Keep only the latest trading session, regular hours."""
    if df is None or df.empty:
        return df
    d = df[df.index.date == df.index[-1].date()]
    return d.between_time("09:15", "15:29")


def completed_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's still-forming daily bar (until 15:35 IST)."""
    now = now_ist()
    if len(df) and df.index[-1].date() == now.date() and now.time() < dtime(15, 35):
        return df.iloc[:-1]
    return df


# --------------------------------------------------------------------------- indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + ru / rd.replace(0, np.nan))).fillna(100)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    day = d.index.date
    pv = (tp * d["Volume"]).groupby(day).cumsum()
    vv = d["Volume"].groupby(day).cumsum()
    d["vwap"] = pv / vv.replace(0, np.nan)
    d["ema9"] = ema(d["Close"], 9)
    d["ema21"] = ema(d["Close"], 21)
    d["atr"] = atr(d, 14)
    d["vol_ma"] = d["Volume"].rolling(20, min_periods=10).mean().shift(1)
    return d


# --------------------------------------------------------------------------- multi-timeframe
TIMEFRAMES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15}


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute bars to N-minute bars aligned to the 09:15 market open."""
    if minutes <= 1:
        return df
    o = df.resample(f"{minutes}min", origin="start_day", offset=pd.Timedelta(minutes=555)).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return o.dropna(subset=["Close"])


def tf_frame(raw_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample multi-day 1-minute data and add VWAP (per session), EMA9/21, ATR, RSI.
    Using several days of history warms up the EMAs so higher timeframes are meaningful."""
    d = resample_ohlcv(raw_1m.between_time("09:15", "15:29"), minutes)
    d = add_intraday_indicators(d)
    d["rsi"] = rsi(d["Close"])
    return d


def mtf_state(frames: Dict[str, pd.DataFrame]) -> List[dict]:
    out = []
    for name, d in frames.items():
        if d is None or d.empty:
            continue
        l = d.iloc[-1]
        has_vw = bool(pd.notna(l["vwap"]))
        above = has_vw and l["Close"] > l["vwap"]
        below = has_vw and l["Close"] < l["vwap"]
        e_up = bool(l["ema9"] > l["ema21"])
        trend = "Bullish" if (above and e_up) else "Bearish" if (below and not e_up) else "Mixed"
        rsi_v = float(l["rsi"]) if "rsi" in d.columns and pd.notna(l["rsi"]) else float(rsi(d["Close"]).iloc[-1])
        out.append(dict(tf=name, trend=trend, rsi=round(rsi_v, 1), close=float(l["Close"]),
                        vwap=float(l["vwap"]) if has_vw else np.nan, ema_bull=e_up))
    return out


def mtf_agrees(states: List[dict], side: str, tfs=("5m", "15m")) -> Tuple[int, int]:
    want = "Bullish" if side == "LONG" else "Bearish"
    use = [s for s in states if s["tf"] in tfs]
    return sum(s["trend"] == want for s in use), len(use)


# --------------------------------------------------------------------------- daily setup / watchlist
def score_side(m: dict, side: str) -> int:
    sc = 0
    if side == "LONG":
        sc += 10 * int(m["close"] > m["ema20"]) + 10 * int(m["ema20"] > m["ema50"]) + 5 * int(m["ema50"] > m["ema50_5"])
        r = m["rsi"]
        sc += 20 if 55 <= r <= 75 else 10 if (50 <= r < 55 or 75 < r <= 80) else 0
        d = m["dist_hh"]
        sc += 20 if d <= 2 else 10 if d <= 5 else 0
        sc += 10 * int(m["rs20"] > 0)
    else:
        sc += 10 * int(m["close"] < m["ema20"]) + 10 * int(m["ema20"] < m["ema50"]) + 5 * int(m["ema50"] < m["ema50_5"])
        r = m["rsi"]
        sc += 20 if 25 <= r <= 45 else 10 if (20 <= r < 25 or 45 < r <= 50) else 0
        d = m["dist_ll"]
        sc += 20 if d <= 2 else 10 if d <= 5 else 0
        sc += 10 * int(m["rs20"] < 0)
    vx = m["vol_x"]
    sc += 15 if vx >= 1.5 else 8 if vx >= 1.2 else 0
    ap = m["atr_pct"]
    sc += 10 if 1 <= ap <= 4 else 5 if (0.7 <= ap < 1 or 4 < ap <= 5) else 0
    return int(sc)


def daily_metrics(df: pd.DataFrame, nifty_ret20: float = 0.0) -> Optional[dict]:
    h = completed_daily(df)
    if h is None or len(h) < 60:
        return None
    c, hi, lo, v = h["Close"], h["High"], h["Low"], h["Volume"]
    e20, e50 = ema(c, 20), ema(c, 50)
    close, pdh, pdl = float(c.iloc[-1]), float(hi.iloc[-1]), float(lo.iloc[-1])
    a = float(atr(h, 14).iloc[-1])
    hh20, ll20 = float(hi.iloc[-20:].max()), float(lo.iloc[-20:].min())
    avgv = float(v.iloc[-21:-1].mean())
    p = (pdh + pdl + close) / 3
    ret20 = (close / float(c.iloc[-21]) - 1) * 100
    m = dict(
        asof=h.index[-1].date().isoformat(), close=close, pdh=pdh, pdl=pdl,
        pivot=p, r1=2 * p - pdl, s1=2 * p - pdh, r2=p + (pdh - pdl), s2=p - (pdh - pdl),
        hh20=hh20, ll20=ll20, ema20=float(e20.iloc[-1]), ema50=float(e50.iloc[-1]), ema50_5=float(e50.iloc[-6]),
        rsi=float(rsi(c).iloc[-1]), atr=a, atr_pct=a / close * 100,
        vol_x=float(v.iloc[-1]) / avgv if avgv > 0 else 0.0, avg_vol=avgv,
        ret20=ret20, rs20=ret20 - nifty_ret20,
        dist_hh=(hh20 - close) / close * 100, dist_ll=(close - ll20) / close * 100,
        turnover_cr=close * float(v.iloc[-20:].mean()) / 1e7,
        day_chg=(close / float(c.iloc[-2]) - 1) * 100,
    )
    m["long_score"] = score_side(m, "LONG")
    m["short_score"] = score_side(m, "SHORT")
    m["bias"] = "LONG" if m["long_score"] >= m["short_score"] else "SHORT"
    return m


def build_watchlist(daily: Dict[str, pd.DataFrame], nifty_ret20: float, cfg: Cfg) -> pd.DataFrame:
    rows = []
    for sym, df in daily.items():
        m = daily_metrics(df, nifty_ret20)
        if m and m["turnover_cr"] >= cfg.min_turnover_cr:
            rows.append({"symbol": sym, **m})
    return pd.DataFrame(rows)


def index_ret20() -> float:
    try:
        d = download_batch(["^NSEI"], "6mo", "1d").get("^NSEI")
        h = completed_daily(d)
        return float((h["Close"].iloc[-1] / h["Close"].iloc[-21] - 1) * 100)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- market breadth
def index_snapshot() -> pd.DataFrame:
    names = {"^NSEI": "Nifty 50", "^NSEBANK": "Bank Nifty", "^INDIAVIX": "India VIX"}
    try:
        dd = download_batch(list(names), "5d", "1d")
        ii = download_batch(list(names), "1d", "5m")
    except Exception:
        return pd.DataFrame()
    out = []
    for k, n in names.items():
        d, i = dd.get(k), ii.get(k)
        if d is None or len(d) < 2:
            continue
        has_i = i is not None and len(i) > 0
        last = float(i["Close"].iloc[-1]) if has_i else float(d["Close"].iloc[-1])
        sess = i.index[-1].date() if has_i else d.index[-1].date()
        prev = float(d["Close"].iloc[-2]) if d.index[-1].date() == sess else float(d["Close"].iloc[-1])
        opn = float(session_df(i)["Open"].iloc[0]) if has_i and len(session_df(i)) else last
        out.append(dict(name=n, ltp=round(last, 2), chg_pct=round((last / prev - 1) * 100, 2),
                        from_open_pct=round((last / opn - 1) * 100, 2)))
    return pd.DataFrame(out)


def compute_breadth(daily: Dict[str, pd.DataFrame], industry: Dict[str, str], fetch_index: bool = True) -> dict:
    rows = []
    for sym, df in daily.items():
        if len(df) < 25:
            continue
        c, last, prev = df["Close"], df.iloc[-1], df.iloc[-2]
        rows.append(dict(
            symbol=sym, chg=(last["Close"] / prev["Close"] - 1) * 100,
            above20=bool(last["Close"] > ema(c, 20).iloc[-1]), above50=bool(last["Close"] > ema(c, 50).iloc[-1]),
            new_high=bool(last["High"] >= df["High"].iloc[-21:-1].max()),
            new_low=bool(last["Low"] <= df["Low"].iloc[-21:-1].min()),
            vol=float(last["Volume"]), industry=industry.get(sym, "NA"),
        ))
    b = pd.DataFrame(rows)
    if b.empty:
        return {"verdict": "Unknown", "score": 0, "long_gate": True, "short_gate": True,
                "notes": ["No data"], "sector": pd.DataFrame(), "index": pd.DataFrame(), "asof": ""}
    adv, dec = int((b.chg > 0).sum()), int((b.chg < 0).sum())
    ad = adv / max(adv + dec, 1)
    p20, p50 = b.above20.mean() * 100, b.above50.mean() * 100
    upvol = b.loc[b.chg > 0, "vol"].sum() / max(b.vol.sum(), 1) * 100

    vwap_pct, n_v = np.nan, 0
    idx = pd.DataFrame()
    if fetch_index:
        try:
            n50 = load_universe("Nifty 50")[0]
            intr = download_batch(n50, "1d", "5m", chunk=60)
            above = 0
            for s, df in intr.items():
                d = session_df(df)
                if d is None or len(d) < 3 or d["Volume"].sum() <= 0:
                    continue
                tp = (d["High"] + d["Low"] + d["Close"]) / 3
                vw = (tp * d["Volume"]).cumsum() / d["Volume"].cumsum().replace(0, np.nan)
                above += int(d["Close"].iloc[-1] > vw.iloc[-1])
                n_v += 1
            vwap_pct = above / n_v * 100 if n_v else np.nan
        except Exception:
            pass
        idx = index_snapshot()

    score, notes = 0, []
    if ad >= 0.6: score += 2; notes.append(f"Advancers dominate ({adv} vs {dec})")
    elif ad <= 0.4: score -= 2; notes.append(f"Decliners dominate ({dec} vs {adv})")
    if p20 >= 60: score += 1; notes.append(f"{p20:.0f}% above EMA20")
    elif p20 <= 40: score -= 1; notes.append(f"Only {p20:.0f}% above EMA20")
    if not np.isnan(vwap_pct):
        if vwap_pct >= 60: score += 2; notes.append(f"{vwap_pct:.0f}% of Nifty 50 above VWAP")
        elif vwap_pct <= 40: score -= 2; notes.append(f"Only {vwap_pct:.0f}% of Nifty 50 above VWAP")
    if len(idx):
        n = idx[idx.name == "Nifty 50"]
        if len(n):
            r = n.iloc[0]
            score += int(r.chg_pct > 0) - int(r.chg_pct < 0) + int(r.from_open_pct > 0) - int(r.from_open_pct < 0)
            notes.append(f"Nifty {r.chg_pct:+.2f}% on day, {r.from_open_pct:+.2f}% from open")
        v = idx[idx.name == "India VIX"]
        if len(v):
            if v.iloc[0].chg_pct > 3: score -= 1; notes.append("VIX rising - caution")
            elif v.iloc[0].chg_pct < -3: score += 1; notes.append("VIX falling - supportive")

    verdict = "Bullish" if score >= 3 else "Bearish" if score <= -3 else "Neutral"
    sector = (b.groupby("industry").agg(stocks=("symbol", "count"), adv_pct=("chg", lambda x: (x > 0).mean() * 100),
                                        avg_chg=("chg", "mean")).reset_index().sort_values("avg_chg", ascending=False))
    return dict(
        verdict=verdict, score=int(score), adv=adv, dec=dec, ad_ratio=ad, pct_above20=p20, pct_above50=p50,
        n50_above_vwap=vwap_pct, new_highs=int(b.new_high.sum()), new_lows=int(b.new_low.sum()), up_vol_pct=upvol,
        long_gate=verdict != "Bearish", short_gate=verdict != "Bullish", notes=notes,
        sector=sector, index=idx, asof=max(df.index[-1] for df in daily.values()).date().isoformat(),
    )


# --------------------------------------------------------------------------- ORB + signals + risk
def orb_levels(ind: pd.DataFrame, cfg: Cfg) -> Optional[dict]:
    if ind is None or len(ind) == 0:
        return None
    start, end = 555, 555 + int(cfg.orb_minutes)
    m = np.asarray(ind.index.hour * 60 + ind.index.minute)
    op = ind[(m >= start) & (m < end)]
    if op.empty:
        return None
    hi, lo = float(op["High"].max()), float(op["Low"].min())
    mid = (hi + lo) / 2
    pct = (hi - lo) / mid * 100
    return dict(high=hi, low=lo, mid=mid, range=hi - lo, range_pct=pct, end_minute=end,
                formed=bool(m[-1] >= end and len(op) >= max(3, int(cfg.orb_minutes * 0.6))),
                range_ok=bool(cfg.min_orb_pct <= pct <= cfg.max_orb_pct), open=float(op["Open"].iloc[0]))


def _size(side: str, entry: float, sl: float, cfg: Cfg) -> dict:
    s = 1 if side == "LONG" else -1
    entry, sl = rt(entry), rt(sl)
    risk = abs(entry - sl)
    targets = [rt(entry + s * risk * r) for r in cfg.rr]
    qty = int(min(cfg.capital * cfg.risk_pct / 100 / risk, cfg.capital * cfg.leverage / entry)) if risk > 0 else 0
    qty = max(qty, 0)
    return dict(entry=entry, sl=sl, risk=round(risk, 2), targets=targets, qty=qty,
                risk_amt=round(risk * qty, 2), capital_used=round(entry * qty, 2))


def trade_plan(side: str, entry: float, bar_extreme: float, orb: dict, atr_v: float, cfg: Cfg) -> dict:
    s = 1 if side == "LONG" else -1
    buf = max(0.15 * (atr_v if np.isfinite(atr_v) else 0.0), TICK)
    sl = (min(bar_extreme, orb["high"]) - buf) if s == 1 else (max(bar_extreme, orb["low"]) + buf)
    capped = False
    if abs(entry - sl) > orb["range"]:            # never risk more than the opening-range width
        sl, capped = entry - s * orb["range"], True
    if abs(entry - sl) < entry * 0.001:           # floor at 0.1% so size stays sane
        sl = entry - s * entry * 0.001
    p = _size(side, entry, sl, cfg)
    p["sl_capped"] = capped
    p["orb_target"] = rt(orb["high"] + orb["range"]) if s == 1 else rt(orb["low"] - orb["range"])
    return p


def find_signals(ind: pd.DataFrame, orb: Optional[dict], cfg: Cfg) -> List[dict]:
    """First confirmed 1-minute breakout per side. Confirmation = candle CLOSE beyond ORB, strong body,
    above/below VWAP, EMA9 vs EMA21 aligned, volume above average, and not over-extended."""
    if not orb or not orb["formed"] or not orb["range_ok"] or orb["range"] <= 0:
        return []
    hi, lo, rng = orb["high"], orb["low"], orb["range"]
    mins = np.asarray(ind.index.hour * 60 + ind.index.minute)
    last_entry = tmin(cfg.last_entry)
    O, H, L, C, V = (ind[c].to_numpy(float) for c in ("Open", "High", "Low", "Close", "Volume"))
    VW, E9, E21, AT, VM = (ind[c].to_numpy(float) for c in ("vwap", "ema9", "ema21", "atr", "vol_ma"))
    sigs, fired = [], set()
    for i in range(len(ind)):
        if mins[i] < orb["end_minute"] or mins[i] > last_entry or i < cfg.confirm_bars:
            continue
        br = H[i] - L[i]
        if br <= 0 or np.isnan(VM[i]) or np.isnan(VW[i]) or VM[i] <= 0:
            continue
        body, vr = abs(C[i] - O[i]) / br, V[i] / VM[i]
        if body < cfg.body_ratio or vr < cfg.vol_mult:
            continue
        for side in ("LONG", "SHORT"):
            if side in fired:
                continue
            if side == "LONG":
                ok = all(C[i - k] > hi for k in range(cfg.confirm_bars)) and C[i] > O[i] and C[i] > VW[i] and E9[i] > E21[i]
                ext, extreme = C[i] - hi, L[i]
            else:
                ok = all(C[i - k] < lo for k in range(cfg.confirm_bars)) and C[i] < O[i] and C[i] < VW[i] and E9[i] < E21[i]
                ext, extreme = lo - C[i], H[i]
            if not ok or ext > max(0.35 * rng, 1.5 * (AT[i] if np.isfinite(AT[i]) else 0)):
                continue
            plan = trade_plan(side, C[i], extreme, orb, AT[i], cfg)
            sigs.append(dict(ts=ind.index[i], side=side, vol_ratio=round(float(vr), 2),
                             orb_high=hi, orb_low=lo, **plan))
            fired.add(side)
    return sorted(sigs, key=lambda s: s["ts"])


# --------------------------------------------------------------------------- scanner
def intraday_snapshot(df1m: pd.DataFrame, cfg: Cfg, dm: Optional[dict]) -> Optional[dict]:
    d = session_df(df1m)
    if d is None or len(d) < 5:
        return None
    ind = add_intraday_indicators(d)
    orb = orb_levels(ind, cfg)
    longs = [s for s in find_signals(ind, orb, cfg) if s["side"] == "LONG"]
    sig = longs[-1] if longs else None
    last, n = ind.iloc[-1], len(ind)
    avgv = dm.get("avg_vol") if dm else None
    frac = min(1.0, (n / 375) ** 0.65)     # rough intraday volume curve
    rvol = float(ind["Volume"].sum() / (avgv * frac)) if avgv and avgv > 0 else float("nan")
    lo, hi = float(ind["Low"].min()), float(ind["High"].max())
    pos = float((last["Close"] - lo) / (hi - lo)) if hi > lo else 0.5
    above_vwap = bool(pd.notna(last["vwap"]) and last["Close"] > last["vwap"])
    above_orb = bool(orb and last["Close"] > orb["high"])
    score = (20 * above_vwap + 25 * above_orb + (20 if sig else 0)
             + (15 if rvol >= 1.5 else 8 if rvol >= 1.2 else 0) + (10 if pos >= 0.7 else 0)
             + (10 if last["ema9"] > last["ema21"] else 0))
    age = (ind.index[-1] - sig["ts"]).total_seconds() / 60 if sig else None
    return dict(price=float(last["Close"]), vwap=float(last["vwap"]) if pd.notna(last["vwap"]) else np.nan,
                orb_high=orb["high"] if orb else np.nan, orb_low=orb["low"] if orb else np.nan,
                orb_formed=bool(orb and orb["formed"]), rvol=rvol, above_vwap=above_vwap, above_orb=above_orb,
                intraday_score=int(score), sig=sig, age_min=age, last_ts=ind.index[-1],
                day_chg=float((last["Close"] / float(d["Open"].iloc[0]) - 1) * 100))


_STATUS_ORDER = {"STRONG BUY": 0, "BUY (late/weak)": 1, "NEAR ORB HIGH": 2, "WATCH": 3}


def run_scan(symbols: List[str], industry: Dict[str, str], cfg: Cfg, daily: Optional[Dict[str, pd.DataFrame]] = None,
             nifty_ret20: float = 0.0, breadth: Optional[dict] = None) -> pd.DataFrame:
    """Stage 1: daily structure on the whole universe. Stage 2: 1-minute ORB confirmation on the top-K."""
    if daily is None:
        daily = download_batch(symbols, "6mo", "1d")
    wl = build_watchlist(daily, nifty_ret20, cfg)
    if wl.empty:
        return pd.DataFrame()
    cand = wl.sort_values("long_score", ascending=False).head(cfg.top_k)
    intr = download_batch(list(cand.symbol), "1d", "1m", chunk=40)
    gate = breadth["long_gate"] if breadth else True
    rows = []
    for _, r in cand.iterrows():
        snap = intraday_snapshot(intr.get(r.symbol), cfg, r.to_dict()) if r.symbol in intr else None
        if snap is None:
            continue
        sig = snap["sig"]
        final = round(0.4 * r.long_score + 0.6 * snap["intraday_score"], 1)
        rv_ok = bool(snap["rvol"] >= 1.2) if np.isfinite(snap["rvol"]) else False
        fresh = sig is not None and snap["age_min"] is not None and snap["age_min"] <= cfg.fresh_min
        if final >= cfg.min_score and sig and snap["above_vwap"] and rv_ok and gate and fresh:
            status = "STRONG BUY"
        elif sig and snap["above_vwap"]:
            status = "BUY (late/weak)"
        elif snap["orb_formed"] and np.isfinite(snap["orb_high"]) and 0 <= (snap["orb_high"] - snap["price"]) / snap["price"] * 100 <= 0.3:
            status = "NEAR ORB HIGH"
        else:
            status = "WATCH"
        rows.append(dict(
            symbol=r.symbol, status=status, score=final, daily_score=int(r.long_score), intraday_score=snap["intraday_score"],
            price=round(snap["price"], 2), day_chg=round(snap["day_chg"], 2), vwap=round(snap["vwap"], 2) if np.isfinite(snap["vwap"]) else None,
            orb_high=round(snap["orb_high"], 2) if np.isfinite(snap["orb_high"]) else None,
            orb_low=round(snap["orb_low"], 2) if np.isfinite(snap["orb_low"]) else None,
            rvol=round(snap["rvol"], 2) if np.isfinite(snap["rvol"]) else None,
            sig_time=sig["ts"].strftime("%H:%M") if sig else "", age_min=round(snap["age_min"], 1) if sig else None,
            entry=sig["entry"] if sig else None, sl=sig["sl"] if sig else None,
            t1=sig["targets"][0] if sig else None, t2=sig["targets"][1] if sig else None, t3=sig["targets"][2] if sig else None,
            qty=sig["qty"] if sig else None, risk_amt=sig["risk_amt"] if sig else None,
            industry=industry.get(r.symbol, "NA"),
        ))
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["_o"] = res.status.map(_STATUS_ORDER)
    return res.sort_values(["_o", "score"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- trade lifecycle
def new_trade(symbol: str, sig: dict, cfg: Cfg) -> dict:
    ts = sig["ts"].isoformat()
    return dict(
        symbol=symbol, side=sig["side"], entry=sig["entry"], sl=sig["sl"], init_sl=sig["sl"],
        targets=list(sig["targets"]), qty=sig["qty"], qty_open=sig["qty"], stage=0, status="OPEN",
        opened_at=ts, last_bar_ts=ts, sl_from_ts=ts, realized=0.0, pnl=0.0, exit_price=None, closed_at=None,
        close_reason=None, last_price=sig["entry"], risk_ps=sig["risk"], trail_mode=cfg.trail_mode,
        trail_n=cfg.trail_n, atr_k=cfg.atr_k, book_pct=list(cfg.book_pct), eod_exit=cfg.eod_exit,
        log=[f"{sig['ts'].strftime('%H:%M')} {sig['side']} entry {sig['entry']} | SL {sig['sl']} | T {sig['targets']}"],
    )


def manual_trade(symbol: str, side: str, entry: float, sl: float, cfg: Cfg) -> dict:
    p = _size(side, entry, sl, cfg)
    ts = now_ist()
    return new_trade(symbol, dict(ts=ts, side=side, **p), cfg)


def _stamp(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def _raise_sl(t: dict, new_sl: float, ts, ev: List[str], why: str) -> None:
    new_sl = rt(new_sl)
    better = new_sl > t["sl"] if t["side"] == "LONG" else new_sl < t["sl"]
    if better:
        t["sl"] = new_sl
        t["sl_from_ts"] = pd.Timestamp(ts).isoformat()   # takes effect from the NEXT bar
        ev.append(f"🔒 {t['symbol']} SL → {new_sl} ({why})")


def _finalize(t: dict, px: float, reason: str, ts, ev: List[str]) -> None:
    s = 1 if t["side"] == "LONG" else -1
    q = t["qty_open"]
    t["realized"] += (px - t["entry"]) * s * q
    t["qty_open"] = 0
    t.update(status="CLOSED", exit_price=rt(px), closed_at=pd.Timestamp(ts).isoformat(), close_reason=reason,
             pnl=round(t["realized"], 2), last_price=px)
    t["r_multiple"] = round(t["pnl"] / (t["risk_ps"] * t["qty"]), 2) if t["risk_ps"] and t["qty"] else 0.0
    ev.append(f"{'✅' if t['pnl'] >= 0 else '❌'} {t['symbol']} CLOSED ({reason}) @ {rt(px)} | P&L ₹{t['pnl']:,.0f} | {t['r_multiple']}R")


def close_trade(t: dict, px: float, reason: str = "MANUAL") -> Tuple[dict, List[str]]:
    ev: List[str] = []
    if t["status"] == "OPEN":
        _finalize(t, px, reason, now_ist(), ev)
    return t, ev


def _check_targets(t: dict, extreme: float, ts, ev: List[str]) -> None:
    s = 1 if t["side"] == "LONG" else -1
    while t["status"] == "OPEN" and t["stage"] < len(t["targets"]):
        i = t["stage"]
        tgt = t["targets"][i]
        if not (extreme >= tgt if s == 1 else extreme <= tgt):
            break
        last = i == len(t["targets"]) - 1
        if last:
            ev.append(f"🎯 {t['symbol']} T{i + 1} hit @ {tgt}")
            t["stage"] = i + 1
            _finalize(t, tgt, f"T{i + 1}", ts, ev)
            return
        q = min(t["qty_open"], int(round(t["qty"] * t["book_pct"][i] / 100)))
        if q > 0:
            t["realized"] += (tgt - t["entry"]) * s * q
            t["qty_open"] -= q
        t["stage"] = i + 1
        ev.append(f"🎯 {t['symbol']} T{i + 1} hit @ {tgt} - booked {q} sh")
        _raise_sl(t, t["entry"] if i == 0 else t["targets"][i - 1], ts, ev, "breakeven" if i == 0 else f"locked T{i}")


def _trail(t: dict, ind: pd.DataFrame, ts, ev: List[str]) -> None:
    if t["stage"] < 1 or t["trail_mode"] == "None":
        return
    s = 1 if t["side"] == "LONG" else -1
    a = float(ind["atr"].iloc[-1]) if np.isfinite(ind["atr"].iloc[-1]) else 0.0
    close = float(ind["Close"].iloc[-1])
    mode = t["trail_mode"]
    if mode == "Structure":
        w = ind.iloc[-int(t["trail_n"]):]
        cand = float(w["Low"].min()) - 0.1 * a if s == 1 else float(w["High"].max()) + 0.1 * a
    elif mode == "EMA9":
        e = float(ind["ema9"].iloc[-1])
        cand = e - 0.1 * a if s == 1 else e + 0.1 * a
    else:  # ATR chandelier
        since = ind[ind.index >= pd.Timestamp(t["opened_at"])]
        if since.empty:
            return
        cand = float(since["Close"].max()) - t["atr_k"] * a if s == 1 else float(since["Close"].min()) + t["atr_k"] * a
    if (s == 1 and cand < close) or (s == -1 and cand > close):
        _raise_sl(t, cand, ts, ev, f"trail {mode}")


def manage_trade(t: dict, bars: Optional[pd.DataFrame], now: Optional[pd.Timestamp] = None) -> Tuple[dict, List[str]]:
    """Advance one open trade using fresh 1-minute bars. Conservative rules:
    - SL is checked before targets inside a bar; SL changes apply from the next bar.
    - Targets book partial qty, ratchet SL (T1 -> breakeven, T2 -> T1) then trail on completed bars.
    - Live price is also checked between bar closes so stops/targets are not late."""
    ev: List[str] = []
    if t["status"] != "OPEN" or bars is None or len(bars) == 0:
        return t, ev
    now = now if now is not None else now_ist()
    s = 1 if t["side"] == "LONG" else -1
    d = session_df(bars)
    if d is None or d.empty:
        return t, ev
    ind = add_intraday_indicators(d)
    if pd.Timestamp(t["opened_at"]).date() != ind.index[-1].date():
        _finalize(t, t["last_price"], "STALE (carried over)", now, ev)
        return t, ev
    eod = tmin(t["eod_exit"])
    forming = (ind.index[-1] + pd.Timedelta(minutes=1)) > now
    done = ind.iloc[:-1] if forming else ind
    for ts, b in done[done.index > pd.Timestamp(t["last_bar_ts"])].iterrows():
        if t["status"] != "OPEN":
            break
        if ts > pd.Timestamp(t["sl_from_ts"]) and ((s == 1 and b["Low"] <= t["sl"]) or (s == -1 and b["High"] >= t["sl"])):
            px = t["sl"]
            if (s == 1 and b["Open"] < px) or (s == -1 and b["Open"] > px):
                px = float(b["Open"])      # gapped through the stop
            _finalize(t, px, "SL" if t["stage"] == 0 else "TRAILING SL", ts, ev)
            break
        _check_targets(t, float(b["High"] if s == 1 else b["Low"]), ts, ev)
        if t["status"] == "OPEN" and ts.hour * 60 + ts.minute >= eod:
            _finalize(t, float(b["Close"]), "EOD", ts, ev)
            break
        if t["status"] == "OPEN":
            _trail(t, ind.loc[:ts], ts, ev)
            t["last_bar_ts"] = ts.isoformat()
    if t["status"] == "OPEN":
        px = float(ind["Close"].iloc[-1])
        t["last_price"] = px
        if (s == 1 and px <= t["sl"]) or (s == -1 and px >= t["sl"]):
            _finalize(t, px, "SL" if t["stage"] == 0 else "TRAILING SL", ind.index[-1], ev)
        else:
            _check_targets(t, px, ind.index[-1], ev)
            if t["status"] == "OPEN" and now.hour * 60 + now.minute >= eod and market_open(now):
                _finalize(t, px, "EOD", now, ev)
    t["log"] = (t.get("log") or []) + ev
    return t, ev


def trade_pnl(t: dict) -> Tuple[float, float]:
    """(unrealised, net) in rupees."""
    s = 1 if t["side"] == "LONG" else -1
    un = (t["last_price"] - t["entry"]) * s * t["qty_open"] if t["status"] == "OPEN" else 0.0
    return round(un, 2), round(un + t["realized"], 2)


# --------------------------------------------------------------------------- alerts
def notify(text: str) -> bool:
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False
    try:
        return requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                             json={"chat_id": chat, "text": text}, timeout=10).ok
    except Exception:
        return False


def strong_buy_message(r) -> str:
    return (f"🚀 STRONG BUY {r['symbol']} @ {r['entry']} (score {r['score']}, RVOL {r['rvol']})\n"
            f"SL {r['sl']} | T1 {r['t1']} | T2 {r['t2']} | T3 {r['t3']} | Qty {r['qty']} | Risk ₹{r['risk_amt']:,.0f}\n"
            f"Signal {r['sig_time']} IST | ORB {r['orb_low']}-{r['orb_high']}")
