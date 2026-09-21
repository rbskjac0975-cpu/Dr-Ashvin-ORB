"""
fno.py - F&O universe, lot sizes, expiry and option strike / premium suggestion.

IMPORTANT - what is real and what is estimated:
  * Universe + lot sizes come from NSE's fo_mktlots.csv when reachable (cached in ./cache), or from a file
    you drop into ./cache/fno_lots.csv (columns: symbol,lot[,strike_step]).  If neither is available a
    built-in approximate symbol list is used and lot sizes are UNKNOWN (None) - lots change, so none are hard-coded.
  * Option premiums are Black-Scholes ESTIMATES (volatility proxy from the stock's recent realised volatility).
    yfinance has no option chain for NSE. Always read the live premium from your broker before trading.
  * Strike interval is an approximate price-band table (override per symbol via the strike_step column).
"""
from __future__ import annotations

import math
import os
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import engine as E

LOT_URLS = [
    "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://archives.nseindia.com/content/fo/fo_mktlots.csv",
]
LOT_FILE = os.path.join(E.CACHE_DIR, "fo_mktlots.csv")
USER_FILE = os.path.join(E.CACHE_DIR, "fno_lots.csv")
INDEX_NAMES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

# Offline fallback only: an approximate list of liquid F&O stocks. Membership changes - prefer the NSE file.
FNO_FALLBACK = (
    "ABB ABCAPITAL ACC ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ALKEM AMBUJACEM APOLLOHOSP APOLLOTYRE ASHOKLEY "
    "ASIANPAINT ASTRAL AUBANK AUROPHARMA AXISBANK BAJAJ-AUTO BAJAJFINSV BAJFINANCE BALKRISIND BANDHANBNK BANKBARODA "
    "BEL BERGEPAINT BHARATFORG BHARTIARTL BHEL BIOCON BOSCHLTD BPCL BRITANNIA CANBK CHOLAFIN CIPLA COALINDIA "
    "COFORGE COLPAL CONCOR CUMMINSIND DABUR DALBHARAT DIVISLAB DIXON DLF DRREDDY EICHERMOT ESCORTS ETERNAL EXIDEIND "
    "FEDERALBNK GAIL GLENMARK GODREJCP GODREJPROP GRASIM HAL HAVELLS HCLTECH HDFCAMC HDFCBANK HDFCLIFE HEROMOTOCO "
    "HINDALCO HINDPETRO HINDUNILVR ICICIBANK ICICIGI ICICIPRULI IDEA IDFCFIRSTB IEX IGL INDHOTEL INDIGO INDUSINDBK "
    "INDUSTOWER INFY IOC IRCTC IRFC ITC JINDALSTEL JIOFIN JSWSTEEL JUBLFOOD KOTAKBANK LICHSGFIN LT LUPIN M&M "
    "M&MFIN MANAPPURAM MARICO MARUTI MAXHEALTH MCX MFSL MOTHERSON MPHASIS MUTHOOTFIN NATIONALUM NAUKRI NESTLEIND "
    "NMDC NTPC OBEROIRLTY OFSS ONGC PAGEIND PERSISTENT PETRONET PFC PIDILITIND PIIND PNB POLYCAB POWERGRID "
    "PRESTIGE RAMCOCEM RBLBANK RECLTD RELIANCE SAIL SBICARD SBILIFE SBIN SHREECEM SHRIRAMFIN SIEMENS SRF SUNPHARMA "
    "SUNTV SYNGENE TATACHEM TATACOMM TATACONSUMER TATAELXSI TATAPOWER TATASTEEL TCS TECHM TITAN TORNTPHARM TRENT "
    "TVSMOTOR UBL ULTRACEMCO UNIONBANK UPL VEDL VOLTAS WIPRO ZYDUSLIFE"
).split()


# --------------------------------------------------------------------------- universe + lots
def _parse_nse_lots(path: str) -> Dict[str, int]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    sym_col = next((c for c in df.columns if c.upper() == "SYMBOL"), None)
    if sym_col is None:
        return {}
    month_cols = [c for c in df.columns if c != sym_col and c.upper() != "UNDERLYING"]
    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        s = str(r[sym_col]).strip()
        if not s or s.upper() in INDEX_NAMES or s.upper() == "SYMBOL" or s.lower() == "nan":
            continue
        for c in month_cols:                     # first month column with a valid lot = current series
            try:
                v = int(float(str(r[c]).strip()))
            except Exception:
                continue
            if v > 0:
                out[s] = v
                break
    return out


def _parse_user_file(path: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    lots, steps = {}, {}
    if "symbol" not in df.columns:
        return lots, steps
    for _, r in df.iterrows():
        s = str(r["symbol"]).strip()
        if not s:
            continue
        if "lot" in df.columns and pd.notna(r["lot"]):
            lots[s] = int(r["lot"])
        if "strike_step" in df.columns and pd.notna(r["strike_step"]):
            steps[s] = float(r["strike_step"])
    return lots, steps


def load_fno(max_age_days: int = 3) -> Tuple[List[str], Dict[str, Optional[int]], Dict[str, float], str]:
    """Returns (symbols, {symbol: lot or None}, {symbol: strike_step override}, note)."""
    os.makedirs(E.CACHE_DIR, exist_ok=True)
    fresh = os.path.exists(LOT_FILE) and (time.time() - os.path.getmtime(LOT_FILE) < max_age_days * 86400)
    if not fresh:
        for url in LOT_URLS:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if r.status_code == 200 and "SYMBOL" in r.text[:600].upper():
                    with open(LOT_FILE, "w", encoding="utf-8") as f:
                        f.write(r.text)
                    break
            except Exception:
                continue
    lots: Dict[str, Optional[int]] = {}
    steps: Dict[str, float] = {}
    note = ""
    if os.path.exists(LOT_FILE):
        try:
            lots = dict(_parse_nse_lots(LOT_FILE))
            note = f"F&O list + lot sizes from NSE ({len(lots)} stocks)"
        except Exception:
            lots = {}
    if os.path.exists(USER_FILE):
        try:
            ul, us = _parse_user_file(USER_FILE)
            lots.update(ul)
            steps.update(us)
            note = (note + " + " if note else "") + f"overrides from cache/fno_lots.csv ({len(ul)} lots)"
        except Exception:
            pass
    if lots:
        return sorted(lots), lots, steps, note
    syms = list(FNO_FALLBACK)
    return syms, {s: None for s in syms}, steps, (
        f"NSE F&O list unreachable -> using a built-in approximate list of {len(syms)} liquid F&O stocks. "
        "Lot sizes are unknown (not hard-coded because they change): put fo_mktlots.csv from NSE "
        "(or your own fno_lots.csv with columns symbol,lot) into ./cache/.")


# --------------------------------------------------------------------------- strikes, expiry, pricing
_STEP_TABLE = [(50, 2.5), (250, 5.0), (1000, 10.0), (2500, 20.0), (5000, 50.0), (10000, 100.0)]


def strike_step(price: float, override: Optional[float] = None) -> float:
    """Approximate NSE strike interval by price band. Override per symbol in cache/fno_lots.csv."""
    if override:
        return float(override)
    for limit, step in _STEP_TABLE:
        if price < limit:
            return step
    return 250.0


def next_expiry(today: date, weekday: int = 1) -> date:
    """Nearest monthly expiry: last <weekday> of the month, rolling to next month once it has passed."""
    def last_wd(y: int, m: int) -> date:
        d = date(y, m, 28) + timedelta(days=4)
        d = d - timedelta(days=d.day)                 # last day of month m
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d
    e = last_wd(today.year, today.month)
    if e < today:
        y, m = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        e = last_wd(y, m)
    return e


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sig: float, cp: str) -> Tuple[float, float]:
    """Black-Scholes (price, delta) for a European option. cp = 'CE' or 'PE'."""
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0.0) if cp == "CE" else max(K - S, 0.0)
        return intrinsic, (1.0 if S > K else 0.0) if cp == "CE" else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if cp == "CE":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2), _ncdf(d1)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1), _ncdf(d1) - 1.0


def iv_proxy(daily: Optional[pd.DataFrame]) -> float:
    """Annualised 20-day realised volatility, floored/capped. A stand-in for implied vol - it is not IV."""
    try:
        c = daily["Close"].astype(float)
        r = np.log(c / c.shift(1)).dropna().iloc[-20:]
        v = float(r.std() * math.sqrt(252))
        return float(min(max(v, 0.18), 1.0)) if np.isfinite(v) else 0.30
    except Exception:
        return 0.30


def option_suggestion(side: str, spot: float, sl: float, t1: float, iv: float, expiry: date, today: date,
                      step: float, lot: Optional[int], pref: str = "ATM", r: float = 0.065) -> dict:
    """Pick a strike and estimate the premium zone plus the premium at the underlying's SL and T1."""
    cp = "CE" if side == "LONG" else "PE"
    dte = (expiry - today).days
    T = max(dte, 0.25) / 365.0
    atm = round(spot / step) * step
    off = {"ATM": 0, "ITM-1": -1, "OTM-1": 1}.get(pref, 0)
    k = atm + (off * step if cp == "CE" else -off * step)
    k = round(k, 2)

    def px(S: float, sig: float) -> Tuple[float, float]:
        p, d = bs_price(S, k, T, r, sig, cp)
        return E.rt(max(p, 0.05)), d

    mid, delta = px(spot, iv)
    lo, hi = px(spot, iv * 0.9)[0], px(spot, iv * 1.1)[0]
    lo, hi = min(lo, hi), max(lo, hi)
    return dict(strike=k, type=cp, label=f"{k:g} {cp}", prem=mid, prem_lo=lo, prem_hi=hi, delta=round(abs(delta), 2),
                prem_sl=px(sl, iv)[0], prem_t1=px(t1, iv)[0], lot=lot, lot_value=round(mid * lot) if lot else None,
                expiry=expiry.isoformat(), dte=dte, iv=round(iv * 100, 1), estimated=True)
