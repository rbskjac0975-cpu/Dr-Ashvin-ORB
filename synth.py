"""Synthetic market data used by the tests (no network). yfinance is stubbed so engine.py imports cleanly."""
import sys
import types

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")

if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")
    _st.secrets = {}                                      # plain dict: .get(key) -> None when unset, like real st.secrets
    _st.session_state = {}
    for _fn in ("sidebar", "warning", "error", "markdown", "columns", "form", "form_submit_button", "text_input",
               "stop", "rerun", "button"):
        setattr(_st, _fn, lambda *a, **k: None)
    sys.modules["streamlit"] = _st
    _st_components = types.ModuleType("streamlit.components")
    _st_components_v1 = types.ModuleType("streamlit.components.v1")
    _st_components_v1.html = lambda *a, **k: None
    _st_components.v1 = _st_components_v1
    _st.components = _st_components
    sys.modules["streamlit.components"] = _st_components
    sys.modules["streamlit.components.v1"] = _st_components_v1

import numpy as np
import pandas as pd

DAY = "2026-09-18"           # a Friday
IST = "Asia/Kolkata"


def daily(base, trend, seed, n=130, avg_vol=2_000_000, last_day=None, gap=None, noise=0.006):
    """Daily bars ending the business day before DAY. trend: per-day drift. If last_day given, append that session
    with an open gapped by `gap` % vs previous close."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp(DAY) - pd.tseries.offsets.BDay(1), periods=n)
    rets = trend + rng.normal(0, noise, n)
    c = base * np.cumprod(1 + rets)
    o = np.r_[base, c[:-1]] * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = avg_vol * rng.uniform(0.8, 1.2, n)
    df = pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=v), index=idx)
    if last_day is not None:
        pc = c[-1]
        op = pc * (1 + gap / 100)
        cl = op * (1 + last_day / 100)
        df.loc[pd.Timestamp(DAY)] = [op, max(op, cl) * 1.002, min(op, cl) * 0.998, cl, avg_vol]
    return df


def m1(base, rets, vmult, vb, seed, noise=0.0004, first=None):
    """One session of 1-minute bars. rets/vmult are length-375 arrays. first=(open_at_extreme) option handled by caller."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{DAY} 09:15", periods=375, freq="min", tz=IST)
    r = np.asarray(rets, float) + rng.normal(0, noise * 0.25, 375)
    c = base * np.cumprod(1 + r)
    o = np.r_[base, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, noise * 0.4, 375)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, noise * 0.4, 375)))
    v = vb * np.asarray(vmult, float) * rng.uniform(0.9, 1.1, 375)
    return pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=v), index=idx)


def vol_profile(bmin=None, bmult=1.0, base=1.5):
    v = np.full(375, base)
    v[0] = 2.0            # kept below ld_open_spurt (3.0x) so the plain ORB-breakout fixtures do not also
    v[1:5] = 1.3          # register as an opening-surge leader - see screenshot_universe() for OPEN-path fixtures
    v[5:15] = 1.1
    if bmin is not None:
        v[bmin] = bmult
    return v


def breakout(base, bmin, bmult, up=True, seed=1, after=0.0002, vb=5000.0, brk=0.0010):
    """ORB drifts one way for 15 min, 2 flat bars, then a strong breakout candle at minute bmin."""
    s = 1 if up else -1
    r = np.zeros(375)
    r[:15] = s * 0.0003
    r[bmin] = s * brk
    r[bmin + 1:] = s * after
    return m1(base, r, vol_profile(bmin, bmult), vb, seed)


def ignition(base, up=True, seed=5, steps=60, reverse_at=None, vb=5000.0):
    """Bar 0 opens AT the extreme with a big body + big volume, then a steady one-way trend (optionally reverses)."""
    s = 1 if up else -1
    r = np.zeros(375)
    r[0] = s * 0.006
    r[1:steps] = s * 0.0004
    if reverse_at is not None:
        r[reverse_at:reverse_at + 25] = -s * 0.0025
    df = m1(base, r, vol_profile(), vb, seed, noise=0.0002)
    # force bar-0 open exactly at its extreme (opens at the day's low / high)
    if up:
        df.iloc[0, df.columns.get_loc("Low")] = df.iloc[0]["Open"]
    else:
        df.iloc[0, df.columns.get_loc("High")] = df.iloc[0]["Open"]
    return df


def flat(base, seed=9, vb=5000.0):
    r = np.zeros(375)
    return m1(base, r, np.full(375, 1.0), vb, seed, noise=0.0003)


def universe():
    """symbol -> (daily_df, 1m_df). Daily volumes: vb*375 shares/day so turnover matches."""
    u = {}
    up = dict(base=1000, trend=0.0012)
    dn = dict(base=1000, trend=-0.0012)
    u["EXPL"] = (daily(**up, seed=1, avg_vol=1_875_000), breakout(1000, 17, 16.0, seed=11))
    u["STRG"] = (daily(**up, seed=2, avg_vol=1_875_000), breakout(1000, 17, 4.0, seed=12))
    u["SPRT"] = (daily(**up, seed=3, avg_vol=1_875_000), breakout(1000, 17, 2.0, seed=13))
    u["SHRT"] = (daily(**dn, seed=4, avg_vol=1_875_000), breakout(1000, 17, 16.0, up=False, seed=14))
    u["LATE"] = (daily(**up, seed=5, avg_vol=1_875_000), breakout(1000, 17, 16.0, seed=15, after=0.0014))
    u["JUNK"] = (daily(base=30, trend=0.001, seed=6, avg_vol=1_875_000), breakout(30, 17, 16.0, seed=16))
    u["ILLQ"] = (daily(**up, seed=7, avg_vol=20_000), breakout(1000, 17, 16.0, seed=17, vb=53.0))
    u["FLAT"] = (daily(**up, seed=8, avg_vol=1_875_000), flat(1000))
    u["IGNL"] = (daily(**up, seed=9, avg_vol=1_875_000), ignition(1000, up=True))
    u["IGNX"] = (daily(**up, seed=10, avg_vol=1_875_000), ignition(1000, up=True, seed=6, reverse_at=90))
    u["IGNS"] = (daily(**dn, seed=11, avg_vol=1_875_000), ignition(1000, up=False))
    return u


# ----------------------------------------------------------------------------- screenshot-style universe (opening surges etc.)
def aligned_daily(prev_close, trend, seed, avg_vol=1_875_000, noise=0.005, n=260):
    """Daily history (ending the business day before DAY) rescaled so the last close == prev_close."""
    df = daily(1000, trend, seed, n=n, avg_vol=avg_vol, noise=noise)
    f = prev_close / float(df["Close"].iloc[-1])
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] * f
    return df


def surge(prev, gap_pct, r0, mult, seed, trend=0.001, drift=0.00008, vb=5000.0):
    """Opening-surge stock: gaps `gap_pct` vs prev close, bar 0 moves r0 on `mult`x an average minute of volume."""
    base = prev * (1 + gap_pct / 100)
    r = np.full(375, drift)
    r[0] = r0
    v = np.full(375, 1.0)
    v[0], v[1:5], v[5:15] = mult, max(2.0, mult / 4), 1.5
    return aligned_daily(prev, trend, seed), m1(base, r, v, vb, seed, noise=0.0003)


def ign_move(prev, k, seed, jump=0.011, trend=0.001, vb=5000.0, up=True):
    """Opens at its low (or high), holds it, then a strong volume candle at minute k takes it >1% from the open."""
    s = 1 if up else -1
    r = np.full(375, s * 0.000005)
    r[0] = s * 0.0008
    r[k] = s * jump
    r[k + 1:] = s * 0.0001
    v = np.full(375, 1.0)
    v[0], v[1:5], v[5:15], v[k] = 6.0, 3.0, 1.5, 7.0
    df = m1(prev, r, v, vb, seed, noise=0.0002)
    col = "Low" if up else "High"
    df.iloc[0, df.columns.get_loc(col)] = df.iloc[0]["Open"]
    return aligned_daily(prev, trend if up else -trend, seed), df


def screenshot_universe():
    u = {}
    u["KFINTECH"] = surge(944.99, +1.6, -0.0116, 23.2, 101, trend=-0.0010, drift=-0.00025)          # gap-up then faded -> SELL surge (EXPLOSIVE)
    u["LODHA"] = surge(1100.0, +0.9, +0.012, 41.5, 102)                              # EXPLOSIVE buy
    u["NAMINDIA"] = surge(1214.4, +0.6, +0.008, 6.4, 103)                            # STRONG
    u["PNBHOUSING"] = surge(1174.0, +1.2, +0.0023, 7.8, 104)                         # STRONG (weak candle)
    u["ZYDUSLIFE"] = surge(1120.0, +1.0, +0.005, 5.5, 105)                           # STRONG
    u["ABB"] = surge(5200.0, +0.4, +0.0045, 4.75, 106)                               # SPURT
    u["SUPREMEIND"] = surge(4100.0, +0.7, +0.004, 3.7, 107)                          # SPURT
    u["UNOMINDA"] = surge(1130.0, +0.3, +0.0035, 3.5, 108)                           # SPURT
    u["BAJAJHLDG"] = surge(11000.0, +0.5, +0.0030, 3.42, 109)                        # SPURT
    u["TCS"] = surge(3900.0, -0.3, -0.0040, 3.4, 110, trend=-0.0010, drift=-0.00020)                 # SPURT sell
    u["EXIDEIND"] = (aligned_daily(414.0, 0.001, 111), ignition(414.0 * 1.0, up=True, seed=111))   # ignition 09:15
    u["SONACOMS"] = ign_move(778.0, 35, 112)                                          # ignition ~09:50
    u["BHARATFORG"] = ign_move(1920.0, 134, 113)                                      # ignition ~11:29
    u["ADANIGREEN"] = ign_move(1274.0, 232, 114)                                      # ignition ~13:07
    return u
