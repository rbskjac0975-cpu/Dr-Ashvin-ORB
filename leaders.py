"""
leaders.py - the seven "ORION-style" features built on top of engine.py:

  1. Breakout Leaders     : 3 tiers (EXPLOSIVE > STRONG > SPURT) through a 3-layer quality filter
  2. Option strike ideas  : attached to every leader card (see fno.py; premiums are estimates)
  3. Market mood + alerts : leader_message() for Telegram, mood comes from engine.compute_breadth
  4. Trend Ignition       : stock that opens AT the day's low (or high) and never looks back
  5. Pre-Open Gaps        : gap table frozen at the 09:15 opening price, with long/short trap flags
  6. Journal              : see journal.py
  7. Day Replay           : build_pack() / replay_cards() - every leader re-appears at the minute it fired

Two ways a leader is born (both causal - only data up to that bar is used):
  OPEN  : opening surge. A candle in the first `ld_open_window` minutes whose volume is many times an average
          minute of that stock (fires from the 09:15 candle, i.e. shown at 09:16, before the ORB exists).
  ORB   : a 1-minute close beyond the opening range with volume, VWAP and EMA agreement (from 09:30 by default).
A leader can be upgraded later to a higher tier.
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
CONV_NAMES = ["Day RVOL ≥ 1.5×", "Daily trend agrees", "Holding beyond trigger (3 closes)", "Market mood agrees",
              "20-day relative strength agrees", "52-week zone", "Opening gap agrees"]


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


def prev_close_of(dd: Optional[pd.DataFrame], day) -> Optional[float]:
    """Last daily close strictly before the session day (never today's forming bar)."""
    if dd is None:
        return None
    h = dd[dd.index < pd.Timestamp(day)]
    return float(h["Close"].iloc[-1]) if len(h) else None


def range52(dd: Optional[pd.DataFrame], day) -> Tuple[Optional[float], Optional[float]]:
    if dd is None:
        return None, None
    h = dd[dd.index < pd.Timestamp(day)].iloc[-252:]
    if len(h) < 20:
        return None, None
    return float(h["High"].max()), float(h["Low"].min())


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
def _tier_from_volume(vx: float, cfg: E.Cfg) -> Optional[str]:
    if vx >= cfg.ld_open_explosive:
        return "EXPLOSIVE"
    if vx >= cfg.ld_open_strong:
        return "STRONG"
    if vx >= cfg.ld_open_spurt:
        return "SPURT"
    return None


def leader_events(ind: pd.DataFrame, orb: Optional[dict], cfg: E.Cfg, dm: Optional[dict] = None,
                  prev_close: Optional[float] = None) -> List[dict]:
    """Walk the session bar by bar. Emits the first leader signal per side and any later UPGRADE to a higher tier.
    OPEN path : opening-surge candle (volume vs an average minute of the stock), from the 09:15 candle.
    ORB path  : layer-2 setup quality - close beyond the ORB, body, volume, VWAP side, EMA9/21, not over-extended."""
    if len(ind) == 0:
        return []
    mins = _mins(ind)
    O, H, L, C, V = (ind[c].to_numpy(float) for c in ("Open", "High", "Low", "Close", "Volume"))
    VW, E9, E21, AT, VM = (ind[c].to_numpy(float) for c in ("vwap", "ema9", "ema21", "atr", "vol_ma"))
    avgv = dm.get("avg_vol") if dm else None
    per_min = avgv / 375.0 if avgv and avgv > 0 else None
    n = np.arange(1, len(ind) + 1)
    rv = np.cumsum(V) / (avgv * np.minimum(1.0, (n / 375.0) ** 0.65)) if per_min else np.full(len(ind), np.nan)
    dscore = {"LONG": dm["long_score"] if dm else 50, "SHORT": dm["short_score"] if dm else 50}
    pc = prev_close if prev_close else (dm["close"] if dm else float(O[0]))
    gap = (float(O[0]) / pc - 1) * 100
    sides = ("LONG", "SHORT") if cfg.ld_shorts else ("LONG",)
    last_entry = E.tmin(cfg.last_entry)
    best: Dict[str, int] = {}
    out: List[dict] = []

    # ---- OPEN path: opening surge, fires from the very first completed candle
    open_end = OPEN_MIN + int(cfg.ld_open_window)
    if per_min:
        for i in range(len(ind)):
            if mins[i] >= open_end:
                break
            br = H[i] - L[i]
            if br <= 0:
                continue
            body = abs(C[i] - O[i]) / br
            body_pct = abs(C[i] - O[i]) / O[i] * 100
            vx = V[i] / per_min
            if body < 0.3 or body_pct < cfg.ld_open_min_body or vx < cfg.ld_open_spurt:
                continue
            side = "LONG" if C[i] > O[i] else "SHORT"
            if side not in sides:
                continue
            tier = _tier_from_volume(vx, cfg)
            if tier is None or TIER_RANK[tier] <= best.get(side, 0):
                continue
            long = side == "LONG"
            gap_ok = gap >= 0.3 if long else gap <= -0.3
            vw_ok = bool(np.isfinite(VW[i]) and ((C[i] > VW[i]) if long else (C[i] < VW[i])))
            pts = 40 * _c((vx - 1) / max(cfg.ld_open_explosive - 1, 0.1))
            pts += 15 * _c((body - 0.3) / 0.5) + 15 * _c(body_pct / 0.8) + 15 * dscore[side] / 100
            pts += 10 * gap_ok + 5 * vw_ok
            po = dict(high=float(H[i]), low=float(L[i]), range=max(float(br), 0.004 * float(C[i])))
            best[side] = TIER_RANK[tier]
            plan = E.trade_plan(side, float(C[i]), float(L[i] if long else H[i]), po, float(AT[i]), cfg)
            out.append(dict(ts=ind.index[i], side=side, tier=tier, score=round(float(pts), 1), vol_ratio=round(float(vx), 2),
                            rvol=None, body=round(float(body), 2), price=float(C[i]), kind="OPEN",
                            trigger=float(H[i] if long else L[i]), hold=float((H[i] + L[i]) / 2), range_ref=po["range"],
                            orb_high=po["high"], orb_low=po["low"], **plan))

    # ---- ORB path
    if orb and orb["formed"] and orb["range_ok"] and orb["range"] > 0:
        hi, lo, rng = orb["high"], orb["low"], orb["range"]
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
                pts = 30 * _c((vr - 1) / max(cfg.ld_explosive_vol - 1, 0.1))                 # volume jump
                pts += 10 * _c((body - 0.5) / 0.4)                                            # candle conviction
                pts += 15 * _c(brk / (0.5 * at)) if at > 0 else 7.5                           # decisiveness of the break
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
                                price=float(C[i]), kind="ORB", trigger=float(hi if long else lo), hold=float(hi if long else lo),
                                range_ref=float(rng), orb_high=hi, orb_low=lo, **plan))
    return sorted(out, key=lambda e: e["ts"])


def watch_rating(tier: str, score: float, conviction: int) -> int:
    """Stars: EXPLOSIVE / STRONG = 3. SPURT = 2 when conviction or score is decent, else 1."""
    if tier in ("EXPLOSIVE", "STRONG"):
        return 3
    return 2 if (conviction >= 3 or score >= 55) else 1


def watch_label(stars: int, side: str) -> str:
    w = "BUY" if side == "LONG" else "SELL"
    return {3: f"STRONG {w} WATCH", 2: f"Moderate {w} Watch", 1: f"{w} Watch"}[stars]


def card_fields(ind: pd.DataFrame, side: str, ev: dict, dm: Optional[dict], dd: Optional[pd.DataFrame], day,
                breadth: Optional[dict] = None, combo: bool = False) -> dict:
    """Everything shown on a leader card besides the tier / score: gap, body, candle quality, 'low se / high se',
    7-point conviction, 52-week zone, momentum tag, chip (COMBO / ORB / SURGE) and the watch rating."""
    long = side == "LONG"
    px, o = float(ind["Close"].iloc[-1]), float(ind["Open"].iloc[0])
    hi, lo = float(ind["High"].max()), float(ind["Low"].min())
    pc = prev_close_of(dd, day) or (dm["close"] if dm else o)
    gap, chg = (o / pc - 1) * 100, (px / pc - 1) * 100
    body_pct = abs(px - o) / o * 100
    ratio = abs(px - o) / (hi - lo) if hi > lo else 0.0
    candle = "Strong" if ratio >= 0.6 else "Moderate" if ratio >= 0.4 else "Weak"
    low_se, high_se = (px / lo - 1) * 100, (hi / px - 1) * 100         # "low se" = up from the day's low, "high se" = down from the high
    hi52, lo52 = range52(dd, day)
    z52 = bool(hi52 is not None and ((px >= 0.95 * hi52) if long else (px <= 1.05 * lo52)))
    n = len(ind)
    avgv = dm.get("avg_vol") if dm else None
    rvol = float(ind["Volume"].sum() / (avgv * min(1.0, (n / 375.0) ** 0.65))) if avgv and avgv > 0 else 0.0
    closes = ind["Close"].iloc[-3:].to_numpy(float)
    hold = ev["hold"]
    flags = [
        rvol >= 1.5,
        bool(dm and ((dm["close"] > dm["ema20"] > dm["ema50"]) if long else (dm["close"] < dm["ema20"] < dm["ema50"]))),
        bool(len(closes) and all((c > hold) if long else (c < hold) for c in closes)),
        bool(breadth and breadth.get("verdict") == ("Bullish" if long else "Bearish")),
        bool(dm and ((dm["rs20"] > 0) if long else (dm["rs20"] < 0))),
        z52,
        gap >= 0.3 if long else gap <= -0.3,
    ]
    conviction = int(sum(flags))
    edge = high_se if long else low_se                                   # how close the price sits to the extreme it is trending toward
    momentum = "STRONG" if edge <= 0.25 else "MODERATE" if edge <= 0.8 else "FADING"
    stars = watch_rating(ev["tier"], ev["score"], conviction)
    return dict(prev_close=round(pc, 2), gap_pct=round(gap, 2), day_chg=round(chg, 2), body_pct=round(body_pct, 2), candle=candle,
                low_se=round(low_se, 2), high_se=round(high_se, 2), conviction=conviction, conv_flags=[bool(f) for f in flags],
                z52=z52, momentum=momentum, stars=stars, watch=watch_label(stars, side),
                chip="COMBO" if combo else ("SURGE" if ev.get("kind") == "OPEN" else "ORB"))


def scan_leaders(symbols: List[str], cfg: E.Cfg, daily: Dict[str, pd.DataFrame], data_1m: Dict[str, pd.DataFrame],
                 nifty_ret20: float = 0.0, breadth: Optional[dict] = None, lots: Optional[dict] = None,
                 steps: Optional[dict] = None, industry: Optional[dict] = None,
                 now: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, dict]:
    """Whole F&O universe -> 3-layer filter -> ranked leaders. Returns (board, filter_stats).
       Layer 1: junk / illiquid / frozen names are removed.
       Layer 2: a leader signal must exist AND still be valid (price still beyond its trigger / hold level, right side of VWAP).
       Layer 3: late entries are removed (price already run too far from the trigger, or detected too long ago)."""
    now = now if now is not None else E.now_ist()
    lots, steps, industry = lots or {}, steps or {}, industry or {}
    stats = dict(universe=len(symbols), no_data=0, junk=0, no_setup=0, late=0, against_mood=0, leaders=0)
    rows: List[dict] = []
    for sym in symbols:
        d = E.session_df(data_1m.get(sym))
        dd = daily.get(sym)
        dm = E.daily_metrics(dd, nifty_ret20) if dd is not None else None
        why = quality_ok(d, dm, cfg, min_bars=1)
        if why:
            stats["no_data" if why.startswith("no ") and "range" not in why else "junk"] += 1
            continue
        ind_full = E.add_intraday_indicators(d)
        ind = _completed(ind_full, now)
        if len(ind) == 0:
            stats["no_setup"] += 1
            continue
        day = ind_full.index[-1].date()
        pc = prev_close_of(dd, day)
        orb = E.orb_levels(ind, cfg)
        evs = leader_events(ind, orb, cfg, dm, pc)
        if not evs:
            stats["no_setup"] += 1
            continue
        side = evs[-1]["side"]
        long = side == "LONG"
        mine = [e for e in evs if e["side"] == side]
        first, ev = mine[0], mine[-1]                                    # first detection, latest tier reached
        last = ind_full.iloc[-1]
        px = float(last["Close"])
        vw_ok = (not pd.notna(last["vwap"])) or ((px > float(last["vwap"])) if long else (px < float(last["vwap"])))
        if not (vw_ok and ((px > ev["hold"]) if long else (px < ev["hold"]))):
            stats["no_setup"] += 1
            continue
        ext = max((px - ev["trigger"]) if long else (ev["trigger"] - px), 0.0)
        ext_r = ext / ev["range_ref"]
        age = (ind_full.index[-1] - first["ts"]).total_seconds() / 60
        if ext_r > cfg.ld_late_ext or age > cfg.ld_max_age:
            stats["late"] += 1
            continue
        mood_ok = True if not breadth else (breadth["long_gate"] if long else breadth["short_gate"])
        stats["against_mood"] += int(not mood_ok)
        combo = any(e["side"] == side for e in ignition_events(ind, cfg, dm))
        cf = card_fields(ind_full, side, ev, dm, dd, day, breadth, combo)
        exp = F.next_expiry(day, cfg.expiry_weekday)
        opt = F.option_suggestion(side, px, ev["sl"], ev["targets"][0], F.iv_proxy(dd), exp, day,
                                  F.strike_step(px, steps.get(sym)), lots.get(sym), cfg.ld_strike_pref)
        zone_lo, zone_hi = (ev["trigger"], ev["trigger"] + 0.25 * ev["range_ref"]) if long else (ev["trigger"] - 0.25 * ev["range_ref"], ev["trigger"])
        in_zone = ext_r <= 0.25
        rows.append(dict(
            symbol=sym, side=side, tier=ev["tier"], score=ev["score"], vol_jump=ev["vol_ratio"], kind=ev["kind"],
            rvol=ev["rvol"], detect=first["ts"].strftime("%H:%M"), tier_time=ev["ts"].strftime("%H:%M"),
            sig_ts=ev["ts"].isoformat(), age_min=round(age, 1), price=round(px, 2),
            trigger=round(ev["trigger"], 2), orb_high=round(ev["orb_high"], 2), orb_low=round(ev["orb_low"], 2),
            entry=ev["entry"], sl=ev["sl"], t1=ev["targets"][0], t2=ev["targets"][1], t3=ev["targets"][2],
            qty=ev["qty"], risk=ev["risk"], risk_amt=ev["risk_amt"],
            zone="ENTRY ZONE" if in_zone else f"EXTENDED {ext_r:.2f}x", zone_lo=round(zone_lo, 2), zone_hi=round(zone_hi, 2),
            in_zone=in_zone, ext_r=round(ext_r, 2), mood_ok=mood_ok, industry=industry.get(sym, "NA"),
            lot=lots.get(sym), opt=opt, **cf))
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
    rv = r.get("rvol")
    lines = [f"{TIER_ICON[r['tier']]} {r['tier']} LEADER  {r['symbol']}  {arrow}  (rank #{r['rank']}, score {r['score']:.0f})",
             f"🕐 Detected {r['detect']} IST | volume jump {r['vol_jump']}x" + (f" | RVOL {rv}" if rv else "")
             + f" | conviction {r.get('conviction', 0)}/7",
             f"Trigger {r['trigger']} | entry {r['entry']} | SL {r['sl']} | T1 {r['t1']} T2 {r['t2']}",
             f"Now {r['price']} ({r['zone']}) | {r.get('watch', '')}"]
    if o:
        lot = f" = ₹{o['lot_value']:,}/lot" if o.get("lot_value") else ""
        lines.append(f"🎯 {o['label']} @ ~₹{o['prem']} (est.){lot} exp {o['expiry']}")
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
    """A stock that OPENS AT the session's low (long) / high (short) and never breaks it.
    Path 'candle': one of the first `ig_window` minutes is a strong, high-volume candle that opens at its extreme.
    Path 'move'  : the stock opened at its low, has held it all session, and now (any time before the last-entry
                   time) closes >= `ig_move_pct`% above the open on a volume-backed candle.
    Entry = close of the detection candle; stop just beyond the held extreme; a structure trail is simulated so the
    card can show where the trade exits today. Earliest detection per side wins."""
    if len(ind) < 1:
        return []
    mins = _mins(ind)
    O, H, L, C, V, VM = (ind[c].to_numpy(float) for c in ("Open", "High", "Low", "Close", "Volume", "vol_ma"))
    run_lo, run_hi = np.minimum.accumulate(L), np.maximum.accumulate(H)
    avgv = dm.get("avg_vol") if dm else None
    per_min = avgv / 375.0 if avgv and avgv > 0 else None
    eod, last_entry = E.tmin(cfg.eod_exit), E.tmin(cfg.last_entry)
    out: List[dict] = []
    for side in (("LONG", "SHORT") if cfg.ld_shorts else ("LONG",)):
        long = side == "LONG"
        det, path = None, ""
        o0 = O[0]
        at_extreme = ((o0 - L[0]) / o0 * 100 <= cfg.ig_open_tol) if long else ((H[0] - o0) / o0 * 100 <= cfg.ig_open_tol)
        for j in range(len(ind)):
            if mins[j] > last_entry:
                break
            if long and j > 0 and L[j] < L[0] * (1 - 0.0005):            # 'never looks back': the opening extreme must stay intact
                break
            if not long and j > 0 and H[j] > H[0] * (1 + 0.0005):
                break
            br = H[j] - L[j]
            if br <= 0:
                continue
            body_r, body_p = abs(C[j] - O[j]) / br, abs(C[j] - O[j]) / O[j] * 100
            if not ((C[j] > O[j]) if long else (C[j] < O[j])):
                continue
            cand_open = ((O[j] - L[j]) / O[j] * 100 <= cfg.ig_open_tol) if long else ((H[j] - O[j]) / O[j] * 100 <= cfg.ig_open_tol)
            is_extreme = (L[j] <= run_lo[j] * 1.0005) if long else (H[j] >= run_hi[j] * 0.9995)
            if (mins[j] < OPEN_MIN + cfg.ig_window and cand_open and is_extreme and body_r >= cfg.ig_body
                    and body_p >= cfg.ig_min_move and (per_min is None or V[j] >= cfg.ig_vol * per_min)):
                det, path = j, "candle"
                break
            if at_extreme and body_r >= 0.4:
                move = (C[j] / o0 - 1) * 100 if long else (1 - C[j] / o0) * 100
                vol_ok = (VM[j] > 0 and V[j] >= 1.5 * VM[j]) if np.isfinite(VM[j]) else (per_min is None or V[j] >= per_min)
                if move >= cfg.ig_move_pct and vol_ok:
                    det, path = j, "move"
                    break
        if det is None:
            continue
        j = det
        entry = float(C[j])
        buf = max(entry * 0.0005, E.TICK)
        if path == "move":
            sl0 = (float(run_lo[j]) - buf) if long else (float(run_hi[j]) + buf)
        else:
            sl0 = (float(L[j]) - buf) if long else (float(H[j]) + buf)
        risk = abs(entry - sl0)
        if risk <= 0 or risk / entry * 100 > cfg.ig_max_risk_pct:
            continue
        sim = _simulate(side, O, H, L, C, mins, j, entry, sl0, int(cfg.trail_n), eod, buf)
        ref = sim["exit"] if sim["exit"] is not None else float(C[-1])
        s = 1 if long else -1
        ext = (float(np.max(H[j:])) - entry) if long else (entry - float(np.min(L[j:])))
        out.append(dict(ts=ind.index[j], side=side, path=path, entry=E.rt(entry), init_sl=E.rt(sl0), sl=E.rt(sim["sl"]),
                        status=sim["status"], exit=E.rt(sim["exit"]) if sim["exit"] is not None else None,
                        last=E.rt(float(C[-1])), pnl_pct=round((ref - entry) * s / entry * 100, 2),
                        mfe_pct=round(ext / entry * 100, 2), r_mult=round((ref - entry) * s / risk, 2),
                        vol_x=round(float(V[j] / per_min), 1) if per_min else None,
                        exit_time=ind.index[sim["exit_i"]].strftime("%H:%M") if sim["exit_i"] is not None else ""))
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
        ind_full = E.add_intraday_indicators(d)
        ind = _completed(ind_full, now)
        pc = prev_close_of(dd, ind_full.index[-1].date())
        for ev in ignition_events(ind, cfg, dm):
            ltp = float(ind_full["Close"].iloc[-1])
            rows.append(dict(symbol=sym, detect=ev["ts"].strftime("%H:%M"), _ts=ev["ts"], ltp=round(ltp, 2),
                             day_chg=round((ltp / pc - 1) * 100, 2) if pc else None,
                             **{k: v for k, v in ev.items() if k != "ts"}))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_ts", ascending=False).drop(columns="_ts").reset_index(drop=True)


def ignition_message(r: dict) -> str:
    arrow = "BUY" if r["side"] == "LONG" else "SELL"
    return (f"⚡ TREND IGNITION {r['symbol']} {arrow} @ {r['entry']} (detected {r['detect']}) | "
            f"SL {r['init_sl']} | {r['status']}")


# =========================================================================== 5. pre-open gaps
def fetch_nse_preopen(timeout: int = 12) -> Dict[str, dict]:
    """BEST-EFFORT, UNTESTED against live NSE: indicative equilibrium price and buy/sell order split during the
    09:00-09:08 order window. NSE frequently blocks scripted access; returns {} on any failure (the gap table then
    uses the 09:15 open and shows no B/S chip)."""
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
            if not (sym and iep and prev):
                continue
            row = dict(iep=float(iep), prev=float(prev), pct=(float(iep) / float(prev) - 1) * 100)
            pm = (it.get("detail") or {}).get("preOpenMarket") or {}
            b, sl = pm.get("totalBuyQuantity"), pm.get("totalSellQuantity")
            if b and sl and (float(b) + float(sl)) > 0:
                row["buy_pct"] = round(float(b) / (float(b) + float(sl)) * 100)
            out[sym] = row
        return out
    except Exception:
        return {}


def gap_table(daily: Dict[str, pd.DataFrame], symbols: List[str], cfg: E.Cfg, nifty_ret20: float = 0.0,
              last_px: Optional[Dict[str, float]] = None, gap_min: float = 0.3,
              preopen: Optional[Dict[str, dict]] = None) -> Tuple[pd.DataFrame, str]:
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
                         trap_state=state, reason=why, rsi=round(dm["rsi"], 0),
                         buy_pct=(preopen or {}).get(sym, {}).get("buy_pct")))
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
               save: bool = True, industry: Optional[dict] = None) -> dict:
    """Re-run the leader logic over a full past session. Because every rule is causal, each leader carries the
    exact minute it fired, its tier at that minute, later upgrades, the card fields as they were at that minute and
    the price series for its mini chart. Daily context is cut at the day BEFORE the replayed session."""
    lots, steps, industry = lots or {}, steps or {}, industry or {}
    leaders = []
    stamp = pd.Timestamp(day)
    for sym in symbols:
        df = data_1m.get(sym)
        if df is None or df.empty:
            continue
        d = df[df.index.date == day].between_time("09:15", "15:29")
        dd = daily.get(sym)
        ddh = dd[dd.index < stamp] if dd is not None else None
        dm = E.daily_metrics(ddh, nifty_ret20) if ddh is not None else None
        if quality_ok(d, dm, cfg, min_bars=1):
            continue
        ind = E.add_intraday_indicators(d)
        orb = E.orb_levels(ind, cfg)
        pc = prev_close_of(ddh, day)
        evs = leader_events(ind, orb, cfg, dm, pc)
        if not evs:
            continue
        side = evs[0]["side"]
        evs = [e for e in evs if e["side"] == side]
        first = evs[0]
        upto = ind[ind.index <= first["ts"]]                                   # card fields as they were at that minute
        combo = any(e["side"] == side for e in ignition_events(ind, cfg, dm))
        cf = card_fields(upto, side, first, dm, ddh, day, None, combo)
        opt = F.option_suggestion(side, first["price"], first["sl"], first["targets"][0], F.iv_proxy(ddh),
                                  F.next_expiry(day, cfg.expiry_weekday), day, F.strike_step(first["price"], steps.get(sym)),
                                  lots.get(sym), cfg.ld_strike_pref)
        mins = _mins(ind)
        leaders.append(dict(
            symbol=sym, side=side, orb_high=round(orb["high"], 2) if orb else round(first["orb_high"], 2),
            orb_low=round(orb["low"], 2) if orb else round(first["orb_low"], 2),
            prev_close=cf["prev_close"], chip=cf["chip"], conviction=cf["conviction"], industry=industry.get(sym, "NA"),
            m=[int(x) for x in mins], c=[round(float(x), 2) for x in ind["Close"]],
            events=[dict(m=int(e["ts"].hour * 60 + e["ts"].minute), tier=e["tier"], score=float(e["score"]), kind=e["kind"],
                         vol=float(e["vol_ratio"]), price=round(e["price"], 2), entry=e["entry"], sl=e["sl"],
                         t1=e["targets"][0], t2=e["targets"][1], trigger=round(e["trigger"], 2),
                         stars=watch_rating(e["tier"], e["score"], cf["conviction"])) for e in evs],
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
        last = series[-1] if series else None
        pcl = L.get("prev_close")
        cards.append(dict(symbol=L["symbol"], side=L["side"], tier=cur["tier"], score=cur["score"], vol=cur["vol"],
                          first_m=first["m"], cur_m=cur["m"], upgraded=len(evs) > 1, entry=first["entry"], sl=first["sl"],
                          t1=first["t1"], trigger=first.get("trigger"), orb_high=L["orb_high"], orb_low=L["orb_low"],
                          series=series, m=L["m"][:idx], last=last, opt=L["opt"], fresh=(minute - cur["m"]) <= 2,
                          chip=L.get("chip", "ORB"), conviction=L.get("conviction", 0), stars=cur.get("stars", 1),
                          watch=watch_label(cur.get("stars", 1), L["side"]), industry=L.get("industry", "NA"),
                          day_chg=round((last / pcl - 1) * 100, 2) if (last and pcl) else None))
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
