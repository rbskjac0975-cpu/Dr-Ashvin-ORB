"""
leaders.py - the seven "ORION-style" features built on top of engine.py:

  1. Breakout Leaders     : 3 tiers (EXPLOSIVE > STRONG > SPURT) through a 3-layer quality filter
  2. Option strike ideas  : attached to every leader card (see fno.py; premiums are estimates)
  3. Market mood + alerts : leader_message() for Telegram, mood comes from engine.compute_breadth
  4. Trend Ignition       : candle that opens AT the day's low (or high) and never looks back
  5. Pre-Open Gaps        : gap table frozen at the 09:15 opening price, with long/short trap flags
  6. Journal              : see journal.py
  7. Day Replay           : build_pack() / replay_cards() - every leader re-appears at the minute it fired

Design rule: every signal is evaluated bar-by-bar using ONLY data up to that bar (causal indicators), so the
live board, the replay and any later review show exactly what could have been known at that minute.
"""
from __future__ import annotations

import bisect
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import engine as E
import fno as F
import store

TIER_RANK = {"SPURT": 1, "STRONG": 2, "EXPLOSIVE": 3}
TIER_ICON = {"EXPLOSIVE": "💥", "STRONG": "🔥", "SPURT": "⚡"}
OPEN_MIN = 555          # 09:15


def _c(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _mins(ind: pd.DataFrame) -> np.ndarray:
    return np.asarray(ind.index.hour * 60 + ind.index.minute)


def _completed(ind: pd.DataFrame, now: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Drop the still-forming last 1-minute bar so signals are only ever printed on closed candles."""
    if now is not None and len(ind) and (ind.index[-1] + pd.Timedelta(minutes=1)) > now:
        return ind.iloc[:-1]
    return ind


# =========================================================================== layer 1: junk / illiquid filter
def quality_ok(d1m: Optional[pd.DataFrame], dm: Optional[dict], cfg: E.Cfg, min_bars: int = 15) -> Optional[str]:
    """Layer 1. Returns the removal reason, or None if the stock is clean."""
    if dm is None:
        return "no daily history"
    if d1m is None or len(d1m) < min_bars:
        return "no intraday data"
    if float(d1m["Close"].iloc[-1]) < cfg.ld_min_price:
        return "low price"
    if dm["turnover_cr"] < cfg.ld_min_turnover_cr:
        return "illiquid"
    if float((d1m["Volume"] > 0).mean()) < 0.8:
        return "thin trading"
    if float(d1m["High"].max()) <= float(d1m["Low"].min()):
        return "no range (circuit / frozen)"
    return None


# =========================================================================== 1. breakout leaders
def leader_events(ind: pd.DataFrame, orb: Optional[dict], cfg: E.Cfg, dm: Optional[dict] = None) -> List[dict]:
    """Walk the session bar by bar. Emits the first confirmed breakout per side and any later UPGRADE to a
    higher tier. Layer 2 (setup quality) is enforced here: close beyond the ORB, body, volume, VWAP side,
    EMA9/21 alignment and a not-over-extended entry."""
    if not orb or not orb["formed"] or not orb["range_ok"] or orb["range"] <= 0 or len(ind) == 0:
        return []
    hi, lo, rng = orb["high"], orb["low"], orb["range"]
    mins = _mins(ind)
    O, H, L, C, V = (ind[c].to_numpy(float) for c in ("Open", "High", "Low", "Close", "Volume"))
    VW, E9, E21, AT, VM = (ind[c].to_numpy(float) for c in ("vwap", "ema9", "ema21", "atr", "vol_ma"))
    avgv = dm.get("avg_vol") if dm else None
    n = np.arange(1, len(ind) + 1)
    if avgv and avgv > 0:
        rv = np.cumsum(V) / (avgv * np.minimum(1.0, (n / 375.0) ** 0.65))
    else:
        rv = np.full(len(ind), np.nan)
    dscore = {"LONG": dm["long_score"] if dm else 50, "SHORT": dm["short_score"] if dm else 50}
    sides = ("LONG", "SHORT") if cfg.ld_shorts else ("LONG",)
    last_entry = E.tmin(cfg.last_entry)
    best: Dict[str, int] = {}
    out: List[dict] = []
    for i in range(cfg.confirm_bars, len(ind)):
        if mins[i] < orb["end_minute"] or mins[i] > last_entry:
            continue
        br = H[i] - L[i]
        if br <= 0 or np.isnan(VM[i]) or np.isnan(VW[i]) or VM[i] <= 0:
            continue
        body, vr = abs(C[i] - O[i]) / br, V[i] / VM[i]
        if body < max(cfg.body_ratio, 0.5) or vr < cfg.ld_spurt_vol:
            continue
        at = AT[i] if np.isfinite(AT[i]) else 0.0
        for side in sides:
            long = side == "LONG"
            if long:
                ok = all(C[i - k] > hi for k in range(cfg.confirm_bars)) and C[i] > O[i] and C[i] > VW[i] and E9[i] > E21[i]
                ext, brk, extreme = C[i] - hi, C[i] - hi, L[i]
            else:
                ok = all(C[i - k] < lo for k in range(cfg.confirm_bars)) and C[i] < O[i] and C[i] < VW[i] and E9[i] < E21[i]
                ext, brk, extreme = lo - C[i], lo - C[i], H[i]
            if not ok or ext > max(0.35 * rng, 1.5 * at):
                continue
            # strength score, 0-100
            pts = 30 * _c((vr - 1) / max(cfg.ld_explosive_vol - 1, 0.1))                 # volume jump
            pts += 10 * _c((body - 0.5) / 0.4)                                            # candle conviction
            pts += 15 * _c(brk / (0.5 * at)) if at > 0 else 7.5                           # decisiveness of the close beyond ORB
            pts += 10 * _c(abs(C[i] - VW[i]) / at) if at > 0 else 5.0                     # distance from VWAP
            pts += 15 * _c((rv[i] - 1) / 1.5) if np.isfinite(rv[i]) else 0.0              # cumulative relative volume
            pts += 15 * dscore[side] / 100                                                # daily setup agrees
            pts += 5 if abs(E9[i] - E21[i]) >= 0.1 * at else 2                            # EMA separation
            score = round(float(pts), 1)
            if vr >= cfg.ld_explosive_vol and score >= cfg.ld_explosive_score and body >= 0.65:
                tier = "EXPLOSIVE"
            elif vr >= cfg.ld_strong_vol and score >= cfg.ld_strong_score:
                tier = "STRONG"
            else:
                tier = "SPURT"
            if TIER_RANK[tier] <= best.get(side, 0):
                continue
            best[side] = TIER_RANK[tier]
            plan = E.trade_plan(side, float(C[i]), float(extreme), orb, float(AT[i]), cfg)
            out.append(dict(ts=ind.index[i], side=side, tier=tier, score=score, vol_ratio=round(float(vr), 2),
                            rvol=round(float(rv[i]), 2) if np.isfinite(rv[i]) else None, body=round(float(body), 2),
                            price=float(C[i]), orb_high=hi, orb_low=lo, **plan))
    return sorted(out, key=lambda e: e["ts"])


def scan_leaders(symbols: List[str], cfg: E.Cfg, daily: Dict[str, pd.DataFrame], data_1m: Dict[str, pd.DataFrame],
                 nifty_ret20: float = 0.0, breadth: Optional[dict] = None, lots: Optional[dict] = None,
                 steps: Optional[dict] = None, industry: Optional[dict] = None,
                 now: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, dict]:
    """Whole F&O universe -> 3-layer filter -> ranked leaders. Returns (board, filter_stats).
       Layer 1: junk / illiquid / frozen names are removed.
       Layer 2: a confirmed ORB breakout must exist AND still be valid (price still beyond ORB, right side of VWAP).
       Layer 3: late entries are removed (price already run too far from the ORB, or detected too long ago)."""
    now = now if now is not None else E.now_ist()
    lots, steps, industry = lots or {}, steps or {}, industry or {}
    stats = dict(universe=len(symbols), no_data=0, junk=0, no_setup=0, late=0, against_mood=0, leaders=0)
    rows: List[dict] = []
    for sym in symbols:
        d = E.session_df(data_1m.get(sym))
        dd = daily.get(sym)
        dm = E.daily_metrics(dd, nifty_ret20) if dd is not None else None
        why = quality_ok(d, dm, cfg)
        if why:
            stats["no_data" if why.startswith("no ") and "range" not in why else "junk"] += 1
            continue
        ind_full = E.add_intraday_indicators(d)
        ind = _completed(ind_full, now)
        orb = E.orb_levels(ind, cfg)
        evs = leader_events(ind, orb, cfg, dm)
        if not evs:
            stats["no_setup"] += 1
            continue
        ev = evs[-1]
        side = ev["side"]
        long = side == "LONG"
        first = next(e for e in evs if e["side"] == side)
        ev = [e for e in evs if e["side"] == side][-1]                # latest tier reached on this side
        last = ind_full.iloc[-1]
        px = float(last["Close"])
        valid = (px > orb["high"] and px > float(last["vwap"])) if long else (px < orb["low"] and px < float(last["vwap"]))
        if not valid:
            stats["no_setup"] += 1
            continue
        ext = (px - orb["high"]) if long else (orb["low"] - px)
        ext_r = ext / orb["range"]
        age = (ind_full.index[-1] - first["ts"]).total_seconds() / 60
        if ext_r > cfg.ld_late_ext or age > cfg.ld_max_age:
            stats["late"] += 1
            continue
        mood_ok = True if not breadth else (breadth["long_gate"] if long else breadth["short_gate"])
        stats["against_mood"] += int(not mood_ok)
        sess_day = ind_full.index[-1].date()
        exp = F.next_expiry(sess_day, cfg.expiry_weekday)
        opt = F.option_suggestion(side, px, ev["sl"], ev["targets"][0], F.iv_proxy(dd), exp, sess_day,
                                  F.strike_step(px, steps.get(sym)), lots.get(sym), cfg.ld_strike_pref)
        zone_lo, zone_hi = (orb["high"], orb["high"] + 0.25 * orb["range"]) if long else (orb["low"] - 0.25 * orb["range"], orb["low"])
        in_zone = ext_r <= 0.25
        rows.append(dict(
            symbol=sym, side=side, tier=ev["tier"], score=ev["score"], vol_jump=ev["vol_ratio"],
            rvol=ev["rvol"], detect=first["ts"].strftime("%H:%M"), tier_time=ev["ts"].strftime("%H:%M"),
            sig_ts=ev["ts"].isoformat(), age_min=round(age, 1), price=round(px, 2),
            day_chg=round((px / float(d["Open"].iloc[0]) - 1) * 100, 2),
            trigger=round(orb["high"] if long else orb["low"], 2), orb_high=round(orb["high"], 2), orb_low=round(orb["low"], 2),
            entry=ev["entry"], sl=ev["sl"], t1=ev["targets"][0], t2=ev["targets"][1], t3=ev["targets"][2],
            qty=ev["qty"], risk=ev["risk"], risk_amt=ev["risk_amt"],
            zone="ENTRY ZONE" if in_zone else f"EXTENDED {ext_r:.2f}x ORB",
            zone_lo=round(zone_lo, 2), zone_hi=round(zone_hi, 2), in_zone=in_zone, ext_r=round(ext_r, 2),
            mood_ok=mood_ok, industry=industry.get(sym, "NA"), lot=lots.get(sym), opt=opt))
    stats["leaders"] = len(rows)
    if not rows:
        return pd.DataFrame(), stats
    board = pd.DataFrame(rows)
    board["_r"] = board.tier.map(TIER_RANK)
    board = board.sort_values(["_r", "score"], ascending=[False, False]).drop(columns="_r").reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board, stats


def sig_from_row(r: dict) -> dict:
    """Build the signal dict engine.new_trade() expects from a leader-board row."""
    return dict(ts=pd.Timestamp(r["sig_ts"]), side=r["side"], entry=r["entry"], sl=r["sl"],
                targets=[r["t1"], r["t2"], r["t3"]], qty=int(r["qty"]), risk=r["risk"])


def leader_message(r: dict) -> str:
    o = r.get("opt") or {}
    arrow = "▲ LONG" if r["side"] == "LONG" else "▼ SHORT"
    lines = [f"{TIER_ICON[r['tier']]} {r['tier']} LEADER  {r['symbol']}  {arrow}  (rank #{r['rank']}, score {r['score']:.0f})",
             f"🕐 Detected {r['detect']} IST | volume jump {r['vol_jump']}x | RVOL {r['rvol']}",
             f"ORB trigger {r['trigger']} | entry {r['entry']} | SL {r['sl']} | T1 {r['t1']} T2 {r['t2']}",
             f"Now {r['price']} ({r['zone']})"]
    if o:
        lines.append(f"🎯 {o['label']} ~₹{o['prem_lo']}-{o['prem_hi']} (est.) exp {o['expiry']}"
                     + (f" | lot {o['lot']}" if o.get("lot") else ""))
    return "\n".join(lines)


# =========================================================================== 4. trend ignition
def _simulate(side: str, O, H, L, C, mins, j: int, entry: float, sl0: float, trail_n: int, eod: int, buf: float) -> dict:
    """Structure-trail simulation from the ignition candle. Stop applies from the NEXT bar; gaps fill at the open."""
    long = side == "LONG"
    sl, n = sl0, len(C)
    for k in range(j + 1, n):
        if (long and L[k] <= sl) or (not long and H[k] >= sl):
            px = float(sl)
            if (long and O[k] < sl) or (not long and O[k] > sl):
                px = float(O[k])
            return dict(status="EXITED (SL)" if sl == sl0 else "EXITED (trail)", exit=px, sl=sl, exit_i=k)
        if mins[k] >= eod:
            return dict(status="EXITED (EOD)", exit=float(C[k]), sl=sl, exit_i=k)
        w0 = max(j, k - trail_n + 1)
        if long:
            cand = float(np.min(L[w0:k + 1])) - buf
            if sl < cand < C[k]:
                sl = cand
        else:
            cand = float(np.max(H[w0:k + 1])) + buf
            if C[k] < cand < sl:
                sl = cand
    return dict(status="RUNNING", exit=None, sl=sl, exit_i=None)


def ignition_events(ind: pd.DataFrame, cfg: E.Cfg, dm: Optional[dict] = None) -> List[dict]:
    """Candle inside the first ig_window minutes that opens AT the session low (long) / high (short) - i.e. the
    open is within ig_open_tol% of the candle's extreme, the extreme is the day's extreme so far, the body is
    strong and volume is well above a normal minute. Entry = that candle's close. Then a structure trail is
    simulated, so the card shows where the trade is exited today."""
    if len(ind) < 1:
        return []
    mins = _mins(ind)
    O, H, L, C, V = (ind[c].to_numpy(float) for c in ("Open", "High", "Low", "Close", "Volume"))
    run_lo, run_hi = np.minimum.accumulate(L), np.maximum.accumulate(H)
    avgv = dm.get("avg_vol") if dm else None
    per_min = avgv / 375.0 if avgv and avgv > 0 else None
    eod = E.tmin(cfg.eod_exit)
    out: List[dict] = []
    sides = ("LONG", "SHORT") if cfg.ld_shorts else ("LONG",)
    for side in sides:
        long = side == "LONG"
        for j in range(len(ind)):
            if mins[j] >= OPEN_MIN + cfg.ig_window:
                break
            br = H[j] - L[j]
            if br <= 0 or abs(C[j] - O[j]) / br < cfg.ig_body or abs(C[j] - O[j]) / O[j] * 100 < cfg.ig_min_move:
                continue
            if long:
                if not (C[j] > O[j]) or (O[j] - L[j]) / O[j] * 100 > cfg.ig_open_tol or L[j] > run_lo[j] * 1.0005:
                    continue
            else:
                if not (C[j] < O[j]) or (H[j] - O[j]) / O[j] * 100 > cfg.ig_open_tol or H[j] < run_hi[j] * 0.9995:
                    continue
            if per_min and V[j] < cfg.ig_vol * per_min:
                continue
            entry = float(C[j])
            buf = max(entry * 0.0005, E.TICK)
            sl0 = float(L[j] - buf) if long else float(H[j] + buf)
            risk = abs(entry - sl0)
            if risk <= 0 or risk / entry * 100 > cfg.ig_max_risk_pct:
                continue
            sim = _simulate(side, O, H, L, C, mins, j, entry, sl0, int(cfg.trail_n), eod, buf)
            ref = sim["exit"] if sim["exit"] is not None else float(C[-1])
            s = 1 if long else -1
            ext = (float(np.max(H[j:])) - entry) if long else (entry - float(np.min(L[j:])))
            out.append(dict(ts=ind.index[j], side=side, entry=E.rt(entry), init_sl=E.rt(sl0), sl=E.rt(sim["sl"]),
                            status=sim["status"], exit=E.rt(sim["exit"]) if sim["exit"] is not None else None,
                            last=E.rt(float(C[-1])), pnl_pct=round((ref - entry) * s / entry * 100, 2),
                            mfe_pct=round(ext / entry * 100, 2), r_mult=round((ref - entry) * s / risk, 2),
                            vol_x=round(float(V[j] / per_min), 1) if per_min else None,
                            exit_time=ind.index[sim["exit_i"]].strftime("%H:%M") if sim["exit_i"] is not None else ""))
            break                                   # first qualifying ignition candle per side
    return out


def ignition_scan(symbols: List[str], cfg: E.Cfg, daily: Dict[str, pd.DataFrame], data_1m: Dict[str, pd.DataFrame],
                  nifty_ret20: float = 0.0, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    now = now if now is not None else E.now_ist()
    rows = []
    for sym in symbols:
        d = E.session_df(data_1m.get(sym))
        dd = daily.get(sym)
        dm = E.daily_metrics(dd, nifty_ret20) if dd is not None else None
        if quality_ok(d, dm, cfg, min_bars=1):
            continue
        ind = _completed(E.add_intraday_indicators(d), now)
        for ev in ignition_events(ind, cfg, dm):
            rows.append(dict(symbol=sym, detect=ev["ts"].strftime("%H:%M"), _ts=ev["ts"], **{k: v for k, v in ev.items() if k != "ts"}))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_ts", ascending=False).drop(columns="_ts").reset_index(drop=True)


def ignition_message(r: dict) -> str:
    arrow = "BUY" if r["side"] == "LONG" else "SELL"
    return (f"⚡ TREND IGNITION {r['symbol']} {arrow} @ {r['entry']} (detected {r['detect']}) | "
            f"SL {r['init_sl']} | {r['status']}")


# =========================================================================== 5. pre-open gaps
def fetch_nse_preopen(timeout: int = 12) -> Dict[str, dict]:
    """BEST-EFFORT, UNTESTED against live NSE: indicative equilibrium price during the 09:00-09:08 order window.
    NSE frequently blocks scripted access; returns {} on any failure (the gap table then uses the 09:15 open)."""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"})
        s.get("https://www.nseindia.com/", timeout=timeout)
        js = s.get("https://www.nseindia.com/api/market-data-pre-open?key=FO", timeout=timeout).json()
        out = {}
        for it in js.get("data", []):
            md = it.get("metadata", {})
            sym = md.get("symbol")
            iep = md.get("iep") or md.get("lastPrice")
            prev = md.get("previousClose")
            if sym and iep and prev:
                out[sym] = dict(iep=float(iep), prev=float(prev), pct=(float(iep) / float(prev) - 1) * 100)
        return out
    except Exception:
        return {}


def gap_table(daily: Dict[str, pd.DataFrame], symbols: List[str], cfg: E.Cfg, nifty_ret20: float = 0.0,
              last_px: Optional[Dict[str, float]] = None, gap_min: float = 0.3) -> Tuple[pd.DataFrame, str]:
    """Gap = the session's opening price (the exchange's pre-open auction result) vs previous close. The open is
    fixed at 09:15, so the gap column never changes afterwards; only LTP / live % move.
    Trap flags compare the gap with the daily context that existed BEFORE the open:
       SHORT TRAP: gap-down into an uptrend or an oversold stock (shorts get squeezed)
       LONG TRAP : gap-up into a downtrend or an overbought stock (longs get faded)
    They are heuristics, not predictions. 'Sprung' means price has since moved >=0.5% against the gap."""
    rows, asof = [], ""
    for sym in symbols:
        df = daily.get(sym)
        if df is None or len(df) < 62:
            continue
        hist = df.iloc[:-1]
        dm = E.daily_metrics(hist, nifty_ret20)
        if not dm:
            continue
        cur = df.iloc[-1]
        asof = df.index[-1].date().isoformat()
        prev = float(hist["Close"].iloc[-1])
        opn = float(cur["Open"])
        gap = (opn / prev - 1) * 100
        ltp = float((last_px or {}).get(sym, cur["Close"]))
        move = (ltp / opn - 1) * 100
        up_ctx = dm["close"] > dm["ema20"] > dm["ema50"]
        dn_ctx = dm["close"] < dm["ema20"] < dm["ema50"]
        flag, why, state = "", "", ""
        if gap <= -gap_min and (up_ctx or dm["rsi"] <= 35):
            flag = "SHORT TRAP"
            why = "gap-down into uptrend" if up_ctx else f"gap-down while oversold (RSI {dm['rsi']:.0f})"
            state = "SPRUNG" if move >= 0.5 else "armed"
        elif gap >= gap_min and (dn_ctx or dm["rsi"] >= 70):
            flag = "LONG TRAP"
            why = "gap-up into downtrend" if dn_ctx else f"gap-up while overbought (RSI {dm['rsi']:.0f})"
            state = "SPRUNG" if move <= -0.5 else "armed"
        rows.append(dict(symbol=sym, prev_close=round(prev, 2), open=round(opn, 2), gap_pct=round(gap, 2), ltp=round(ltp, 2),
                         live_pct=round((ltp / prev - 1) * 100, 2), from_open_pct=round(move, 2), trap=flag,
                         trap_state=state, reason=why, rsi=round(dm["rsi"], 0)))
    return (pd.DataFrame(rows) if rows else pd.DataFrame()), asof


def freeze_gaps(gt: pd.DataFrame, session: str) -> bool:
    """Save the opening gap snapshot once per session. Returns True if this call created it."""
    key = f"gaps:{session}"
    if gt is None or gt.empty or store.get_meta(key) is not None:
        return False
    store.set_meta(key, dict(saved=E.now_ist().isoformat(),
                             rows=gt[["symbol", "prev_close", "open", "gap_pct", "trap", "reason"]].to_dict("records")))
    return True


def gap_summary_message(gt: pd.DataFrame, session: str) -> str:
    up, dn = gt[gt.gap_pct >= 0.3], gt[gt.gap_pct <= -0.3]
    lt, st_ = gt[gt.trap == "LONG TRAP"], gt[gt.trap == "SHORT TRAP"]
    fmt = lambda df, asc: ", ".join(f"{r.symbol} {r.gap_pct:+.1f}%" for r in df.sort_values("gap_pct", ascending=asc).head(5).itertuples())
    return (f"🌅 Opening gaps {session}: {len(up)} gap-up, {len(dn)} gap-down\n"
            f"Top up: {fmt(up, False) or '-'}\nTop down: {fmt(dn, True) or '-'}\n"
            f"Long-trap flags: {', '.join(lt.symbol.head(8)) or '-'}\nShort-trap flags: {', '.join(st_.symbol.head(8)) or '-'}")


# =========================================================================== 7. day replay
def _pack_key(day: str) -> str:
    return f"pack:{day}"


def pack_dates() -> List[str]:
    return [k.split(":", 1)[1] for k in store.list_meta("pack:")]


def load_pack(day: str) -> Optional[dict]:
    return store.get_meta(_pack_key(day))


def build_pack(data_1m: Dict[str, pd.DataFrame], daily: Dict[str, pd.DataFrame], symbols: List[str], cfg: E.Cfg,
               day: date, nifty_ret20: float = 0.0, lots: Optional[dict] = None, steps: Optional[dict] = None,
               save: bool = True) -> dict:
    """Re-run the leader logic over a full past session. Because every rule is causal, each leader carries the
    exact minute it fired, its tier at that minute, later upgrades, and the price series for its mini chart.
    Daily context is cut at the day BEFORE the replayed session so nothing from the future leaks in."""
    lots, steps = lots or {}, steps or {}
    leaders = []
    stamp = pd.Timestamp(day)
    for sym in symbols:
        df = data_1m.get(sym)
        if df is None or df.empty:
            continue
        d = df[df.index.date == day].between_time("09:15", "15:29")
        dd = daily.get(sym)
        dm = E.daily_metrics(dd[dd.index < stamp], nifty_ret20) if dd is not None else None
        if quality_ok(d, dm, cfg):
            continue
        ind = E.add_intraday_indicators(d)
        orb = E.orb_levels(ind, cfg)
        evs = leader_events(ind, orb, cfg, dm)
        if not evs:
            continue
        side = evs[0]["side"]
        evs = [e for e in evs if e["side"] == side]
        first = evs[0]
        opt = F.option_suggestion(side, first["price"], first["sl"], first["targets"][0], F.iv_proxy(dd[dd.index < stamp]),
                                  F.next_expiry(day, cfg.expiry_weekday), day, F.strike_step(first["price"], steps.get(sym)),
                                  lots.get(sym), cfg.ld_strike_pref)
        mins = _mins(ind)
        leaders.append(dict(
            symbol=sym, side=side, orb_high=round(orb["high"], 2), orb_low=round(orb["low"], 2),
            m=[int(x) for x in mins], c=[round(float(x), 2) for x in ind["Close"]],
            events=[dict(m=int(e["ts"].hour * 60 + e["ts"].minute), tier=e["tier"], score=float(e["score"]),
                         vol=float(e["vol_ratio"]), price=round(e["price"], 2), entry=e["entry"], sl=e["sl"],
                         t1=e["targets"][0], t2=e["targets"][1]) for e in evs],
            opt={k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in opt.items()}))
    pack = dict(date=day.isoformat(), built=E.now_ist().isoformat(), universe=len(symbols), leaders=leaders)
    if save:
        store.set_meta(_pack_key(day.isoformat()), pack)
    return pack


def replay_cards(pack: dict, minute: int) -> List[dict]:
    """What the leader board looked like at `minute` (minutes since midnight). Strongest tier first."""
    cards = []
    for L in pack["leaders"]:
        evs = [e for e in L["events"] if e["m"] <= minute]
        if not evs:
            continue
        cur, first = evs[-1], evs[0]
        idx = bisect.bisect_right(L["m"], minute)
        series = L["c"][:idx]
        cards.append(dict(symbol=L["symbol"], side=L["side"], tier=cur["tier"], score=cur["score"], vol=cur["vol"],
                          first_m=first["m"], cur_m=cur["m"], upgraded=len(evs) > 1, entry=first["entry"], sl=first["sl"],
                          t1=first["t1"], orb_high=L["orb_high"], orb_low=L["orb_low"], series=series, m=L["m"][:idx],
                          last=series[-1] if series else None, opt=L["opt"], fresh=(minute - cur["m"]) <= 2))
    cards.sort(key=lambda c: (-TIER_RANK[c["tier"]], -c["score"]))
    return cards


def replay_feed(pack: dict, minute: int, n: int = 10) -> List[Tuple[int, str]]:
    feed = []
    for L in pack["leaders"]:
        for e in L["events"]:
            if e["m"] <= minute:
                arrow = "▲" if L["side"] == "LONG" else "▼"
                feed.append((e["m"], f"{TIER_ICON[e['tier']]} {e['tier']} {L['symbol']} {arrow} score {e['score']:.0f} · {e['vol']}x volume"))
    feed.sort(key=lambda x: -x[0])
    return feed[:n]
