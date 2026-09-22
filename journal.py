"""
journal.py - trading journal maths (storage lives in store.py, private per access key).

Every entry: date, symbol, side, instrument (Equity / Option), setup, entry, SL, target, exit, qty, option strike,
entry/exit time, mistake tag, note, charges.  P&L, R-multiple and hold time are computed here.
"""
from __future__ import annotations

import calendar
import hashlib
from typing import List, Optional

import numpy as np
import pandas as pd

SETUPS = ["Breakout Leader (EXPLOSIVE)", "Breakout Leader (STRONG)", "Breakout Leader (SPURT)", "Trend Ignition",
          "Gap trap reversal", "ORB scanner (strong buy)", "VWAP pullback", "News / other"]
MISTAKES = ["None", "Chased / late entry", "Moved or removed SL", "No stop-loss", "Oversized", "Exited too early",
            "Held too long", "Traded against market mood", "Overtrading", "Revenge trade", "Ignored my plan"]


SHORT_SETUP = {"Breakout Leader (EXPLOSIVE)": "Leader 💥", "Breakout Leader (STRONG)": "Leader 🔥", "Breakout Leader (SPURT)": "Leader ⚡",
               "Trend Ignition": "Trend Ign.", "Gap trap reversal": "Gap Trap", "ORB scanner (strong buy)": "ORB Scan",
               "VWAP pullback": "VWAP", "News / other": "Own"}
SHORT_MISTAKE = {"None": "OK", "Chased / late entry": "Chased", "Moved or removed SL": "SL moved", "No stop-loss": "No SL",
                 "Oversized": "Oversized", "Exited too early": "Early exit", "Held too long": "Held long",
                 "Traded against market mood": "Vs mood", "Overtrading": "Overtrade", "Revenge trade": "Revenge",
                 "Ignored my plan": "Off-plan"}


def owner_hash(key: str) -> str:
    """The journal is filed under a hash of the access key; without the same key the rows are never queried."""
    return hashlib.sha256(("orb-journal|" + key.strip()).encode()).hexdigest()[:24]


def _minutes(hhmm: Optional[str]) -> Optional[int]:
    try:
        h, m = str(hhmm).split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return None


def enrich(e: dict) -> dict:
    """Compute net P&L, R multiple and hold time from the raw fields."""
    e = dict(e)
    s = 1 if e.get("side", "LONG") == "LONG" else -1
    qty = float(e.get("qty") or 0)
    entry, exit_, sl = float(e.get("entry") or 0), float(e.get("exit") or 0), float(e.get("sl") or 0)
    gross = (exit_ - entry) * s * qty
    if e.get("pnl_override") is not None:            # imported from the Trade manager: partial bookings already netted
        e["pnl"] = round(float(e["pnl_override"]), 2)
        gross = float(e["pnl_override"])
    else:
        e["pnl"] = round(gross - float(e.get("charges") or 0), 2)
    risk = abs(entry - sl) * qty if sl > 0 else 0.0
    e["r"] = round(gross / risk, 2) if risk > 0 else None
    e["pts"] = round((exit_ - entry) * s, 2)
    a, b = _minutes(e.get("entry_time")), _minutes(e.get("exit_time"))
    e["hold_min"] = (b - a) if a is not None and b is not None and b >= a else None
    return e


def to_frame(entries: List[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame([enrich(e) for e in entries])
    for c in ("entry_time", "exit_time", "mistake", "setup", "symbol", "side"):
        if c not in df.columns:
            df[c] = None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "entry_time", "id"], na_position="last").reset_index(drop=True)
    df["equity"] = df["pnl"].cumsum()
    return df


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    w, l = df[df.pnl > 0], df[df.pnl < 0]
    gp, gl = float(w.pnl.sum()), float(-l.pnl.sum())
    dd = float((df.equity - df.equity.cummax()).min())
    daily = df.groupby("date").pnl.sum()
    streak = best = 0
    for v in df.pnl:                                              # best run of consecutive winning trades
        streak = streak + 1 if v > 0 else 0
        best = max(best, streak)
    disc = float((df.mistake.fillna("None") == "None").mean() * 100)
    return dict(trades=len(df), wins=int(len(w)), losses=int(len(l)), win_streak=best, discipline=disc,
                net=float(df.pnl.sum()), win_rate=float((df.pnl > 0).mean() * 100),
                avg_win=float(w.pnl.mean()) if len(w) else 0.0, avg_loss=float(l.pnl.mean()) if len(l) else 0.0,
                profit_factor=(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
                expectancy=float(df.pnl.mean()), avg_r=float(df.r.dropna().mean()) if df.r.notna().any() else None,
                avg_hold=float(df.hold_min.dropna().mean()) if df.hold_min.notna().any() else None,
                max_dd=dd, best_day=(daily.idxmax().date().isoformat(), float(daily.max())),
                worst_day=(daily.idxmin().date().isoformat(), float(daily.min())))


def insights(df: pd.DataFrame) -> List[str]:
    """Plain-English findings: best setup, average hold time, and the mistake that costs you most."""
    if df.empty:
        return []
    out = []
    g = df.groupby("setup").agg(n=("pnl", "size"), net=("pnl", "sum"), win=("pnl", lambda x: (x > 0).mean() * 100))
    if len(g):
        b = g.sort_values("net", ascending=False).iloc[0]
        out.append(f"Best setup: **{g.sort_values('net', ascending=False).index[0]}** - net ₹{b.net:,.0f} over {int(b.n)} trade(s), {b.win:.0f}% wins.")
        if len(g) > 1:
            wname = g.sort_values("net").index[0]
            wr = g.sort_values("net").iloc[0]
            if wr.net < 0:
                out.append(f"Weakest setup: **{wname}** - net ₹{wr.net:,.0f} over {int(wr.n)} trade(s).")
    if df.hold_min.notna().any():
        h = df.hold_min.dropna()
        wins, loss = df[(df.pnl > 0) & df.hold_min.notna()].hold_min, df[(df.pnl < 0) & df.hold_min.notna()].hold_min
        msg = f"Average hold time: **{h.mean():.0f} min**"
        if len(wins) and len(loss):
            msg += f" (winners {wins.mean():.0f} min, losers {loss.mean():.0f} min)"
            if loss.mean() > wins.mean() * 1.3:
                msg += " - you hold losers longer than winners."
        out.append(msg)
    m = df[(df.mistake.fillna("None") != "None") & (df.pnl < 0)]
    if len(m):
        mm = m.groupby("mistake").pnl.agg(["sum", "size"]).sort_values("sum")
        out.append(f"Mistake costing you most: **{mm.index[0]}** - ₹{-mm.iloc[0]['sum']:,.0f} lost across {int(mm.iloc[0]['size'])} trade(s).")
    else:
        out.append("No losing trades carry a mistake tag yet - tag them to see what your errors cost.")
    return out


def calendar_html(df: pd.DataFrame, year: int, month: int) -> str:
    """Month grid: green day / red day with the day's net P&L."""
    daily = {}
    if not df.empty:
        d = df[(df.date.dt.year == year) & (df.date.dt.month == month)].groupby(df.date.dt.day).pnl.sum()
        daily = {int(k): float(v) for k, v in d.items()}
    mx = max([abs(v) for v in daily.values()] or [1.0])
    css = ("<style>.jc{border-collapse:collapse;width:100%;table-layout:fixed}.jc th{font-size:12px;opacity:.7;padding:4px}"
           ".jc td{height:64px;border:1px solid rgba(128,128,128,.25);vertical-align:top;padding:4px;font-size:12px}"
           ".jc .n{opacity:.6}.jc .p{font-weight:600;font-size:13px}</style>")
    rows = ["<tr>" + "".join(f"<th>{d}</th>" for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")) + "</tr>"]
    for week in calendar.Calendar(firstweekday=0).monthdayscalendar(year, month):
        cells = []
        for day in week:
            if day == 0:
                cells.append("<td></td>")
                continue
            if day in daily:
                v = daily[day]
                a = 0.15 + 0.5 * abs(v) / mx
                bg = f"rgba(46,196,166,{a:.2f})" if v >= 0 else f"rgba(239,91,91,{a:.2f})"
                cells.append(f"<td style='background:{bg}'><span class='n'>{day}</span><br><span class='p'>{v:+,.0f}</span></td>")
            else:
                cells.append(f"<td><span class='n'>{day}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return css + "<table class='jc'>" + "".join(rows) + "</table>"


def csv_bytes(df: pd.DataFrame) -> bytes:
    cols = [c for c in ["date", "symbol", "side", "instrument", "setup", "strike", "lots", "lot_size", "qty", "entry", "sl",
                        "target", "exit", "entry_time", "exit_time", "hold_min", "charges", "pnl", "r", "mistake", "note"]
            if c in df.columns]
    out = df[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")


def insight_cards(df: pd.DataFrame) -> dict:
    """The three cards under the calendar: BEST SETUP, AVG HOLD, WATCH OUT (plain text, no markdown)."""
    if df.empty:
        return {}
    out = {}
    g = df.groupby("setup").agg(n=("pnl", "size"), net=("pnl", "sum"), win=("pnl", lambda x: (x > 0).mean() * 100)).sort_values("net", ascending=False)
    b = g.iloc[0]
    out["best"] = (f"{g.index[0]} - {b.win:.0f}% wins ({int(b.n)} trade{'s' if b.n != 1 else ''}), "
                   f"{'+' if b.net >= 0 else '-'}₹{abs(b.net):,.0f} net. " + ("Lean on what works." if b.net > 0 else "Not paying yet - review it."))
    h = df.hold_min.dropna()
    if len(h):
        txt = f"{h.mean():.0f} min"
        et = df.entry_time.map(_minutes)
        am, pm = df[(et < 720) & et.notna()], df[(et >= 720) & et.notna()]
        if len(am) and len(pm):
            a_wr, p_wr = (am.pnl > 0).mean() * 100, (pm.pnl > 0).mean() * 100
            if a_wr >= p_wr + 15:
                txt += " - your winners are morning trades; afternoon trades are mostly losses."
            elif p_wr >= a_wr + 15:
                txt += " - your afternoon trades win more than your morning ones."
        w, l = df[(df.pnl > 0) & df.hold_min.notna()].hold_min, df[(df.pnl < 0) & df.hold_min.notna()].hold_min
        if txt.endswith("min") and len(w) and len(l):
            txt += f" - winners {w.mean():.0f} min, losers {l.mean():.0f} min" + (". You hold losers longer." if l.mean() > w.mean() * 1.3 else ".")
        out["hold"] = txt
    else:
        out["hold"] = "Add entry and exit times to see how long you hold."
    m = df[(df.mistake.fillna("None") != "None") & (df.pnl < 0)]
    if len(m):
        latest = m.date.max()
        mm = m[(m.date.dt.year == latest.year) & (m.date.dt.month == latest.month)]
        gm = mm.groupby("mistake").pnl.agg(["sum", "size"]).sort_values("sum")
        out["watch"] = (f"{int(gm.iloc[0]['size'])} loss{'es' if gm.iloc[0]['size'] != 1 else ''} tagged \"{gm.index[0]}\" - cost "
                        f"₹{-gm.iloc[0]['sum']:,.0f} in {latest.strftime('%b %Y')}. Stop that one habit.")
    else:
        out["watch"] = "No losing trade carries a mistake tag - tag your losses to see what your errors cost."
    return out
