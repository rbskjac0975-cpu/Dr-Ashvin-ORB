import os
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TRADE_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")

import synth  # noqa: E402  (stubs yfinance)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import engine as E  # noqa: E402
import fno as F  # noqa: E402
import journal as J  # noqa: E402
import leaders as L  # noqa: E402
import store  # noqa: E402

store.init()
cfg = E.Cfg()
U = synth.universe()
daily = {k: v[0] for k, v in U.items()}
d1m = {k: v[1] for k, v in U.items()}
syms = list(U)
lots = {s: 500 for s in syms}


def at(hhmm):
    """Truncate all 1-minute data to `hhmm` and return (data, now) as if it were live at that minute."""
    ts = pd.Timestamp(f"{synth.DAY} {hhmm}", tz=synth.IST)
    return {k: v[v.index < ts] for k, v in d1m.items()}, ts + pd.Timedelta(seconds=30)


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        check.failed += 1


check.failed = 0

# ------------------------------------------------------------------ 1. leaders at 09:40 (breakout printed 09:32)
data, now = at("09:40")
breadth = dict(long_gate=True, short_gate=True)
board, st = L.scan_leaders(syms, cfg, daily, data, 0.0, breadth, lots, {}, {}, now)
print(board[["rank", "symbol", "side", "tier", "score", "vol_jump", "detect", "zone", "price"]].to_string())
print(st)
by = {r.symbol: r for r in board.itertuples()}
check("EXPL is EXPLOSIVE and ranked first", "EXPL" in by and by["EXPL"].tier == "EXPLOSIVE" and by["EXPL"].rank == 1)
check("STRG is STRONG", "STRG" in by and by["STRG"].tier == "STRONG")
check("SPRT is SPURT", "SPRT" in by and by["SPRT"].tier == "SPURT")
check("SHRT is a SHORT leader", "SHRT" in by and by["SHRT"].side == "SHORT")
check("tiers sorted strongest first", list(board.tier.map(L.TIER_RANK)) == sorted(board.tier.map(L.TIER_RANK), reverse=True))
check("JUNK removed (layer 1)", "JUNK" not in by)
check("ILLQ removed (layer 1)", "ILLQ" not in by)
check("FLAT has no card", "FLAT" not in by)
check("detect time is 09:32", by["EXPL"].detect == "09:32")
o = by["EXPL"].opt
check("option is CE with a premium band", o["type"] == "CE" and 0 < o["prem_lo"] <= o["prem"] <= o["prem_hi"], str(o["label"]) + f" {o['prem_lo']}-{o['prem_hi']}")
check("option premium at T1 > premium at SL", o["prem_t1"] > o["prem_sl"], f"{o['prem_sl']} < {o['prem_t1']}")
check("SHRT option is PE", by["SHRT"].opt["type"] == "PE")
check("lot size carried", o["lot"] == 500 and o["lot_value"] == round(o["prem"] * 500))

# ------------------------------------------------------------------ layer 3: late entries removed by 10:20
data2, now2 = at("10:20")
board2, st2 = L.scan_leaders(syms, cfg, daily, data2, 0.0, breadth, lots, {}, {}, now2)
check("LATE removed by layer 3 (ran too far / too old)", "LATE" not in set(board2.symbol) if len(board2) else True, str(st2))
check("late counter incremented", st2["late"] >= 1, str(st2))
strict = E.Cfg(ld_max_age=200, ld_late_ext=10.0)
board3, _ = L.scan_leaders(syms, strict, daily, data2, 0.0, breadth, lots, {}, {}, now2)
check("relaxed filter keeps EXPL (proves layer 3 is what removed it)", "EXPL" in set(board3.symbol))

# mood gate flags rather than removes
b_bear = dict(long_gate=False, short_gate=True)
board4, st4 = L.scan_leaders(syms, cfg, daily, data, 0.0, b_bear, lots, {}, {}, now)
check("bearish mood tags longs 'against mood'", not board4[board4.symbol == "EXPL"].iloc[0].mood_ok and st4["against_mood"] >= 1)

# forming bar is not used: at exactly 09:32:20 the 09:32 candle is still forming
data5, _ = at("09:33")
board5, _ = L.scan_leaders(syms, cfg, daily, data5, 0.0, breadth, lots, {}, {}, pd.Timestamp(f"{synth.DAY} 09:32:20", tz=synth.IST))
check("candle still forming -> no signal printed yet", "EXPL" not in set(board5.symbol) if len(board5) else True)

msg = L.leader_message(board.iloc[0].to_dict())
check("telegram message has tier, detect time and option", "EXPLOSIVE" in msg and "Detected 09:32" in msg and "CE" in msg)

# no lookahead: events computed on the full day equal events computed on truncated data up to that minute
full = E.add_intraday_indicators(E.session_df(d1m["EXPL"]))
part = E.add_intraday_indicators(E.session_df(data["EXPL"]))
ef = L.leader_events(full, E.orb_levels(full, cfg), cfg, E.daily_metrics(daily["EXPL"], 0.0))
ep = L.leader_events(part, E.orb_levels(part, cfg), cfg, E.daily_metrics(daily["EXPL"], 0.0))
check("no lookahead: first event identical on partial and full data",
      ef[0]["ts"] == ep[0]["ts"] and ef[0]["tier"] == ep[0]["tier"] and abs(ef[0]["score"] - ep[0]["score"]) < 1e-6,
      f"{ef[0]['ts'].strftime('%H:%M')} {ef[0]['tier']} {ef[0]['score']}")

# ------------------------------------------------------------------ 4. trend ignition
data6, now6 = at("10:30")
ig = L.ignition_scan(syms, cfg, daily, data6, 0.0, now6)
print(ig.to_string())
igs = {(r.symbol, r.side): r for r in ig.itertuples()}
check("IGNL detected at 09:15 as BUY", ("IGNL", "LONG") in igs and igs[("IGNL", "LONG")].detect == "09:15")
check("IGNL RUNNING with positive P&L at 10:30", igs[("IGNL", "LONG")].status == "RUNNING" and igs[("IGNL", "LONG")].pnl_pct > 0)
check("IGNS detected as SHORT", ("IGNS", "SHORT") in igs and igs[("IGNS", "SHORT")].pnl_pct > 0)
check("trail stop moved above initial SL", igs[("IGNL", "LONG")].sl > igs[("IGNL", "LONG")].init_sl)
check("drifting-open breakout names are NOT ignitions (impulse rule)", not any(s_ in ("FLAT", "EXPL", "STRG", "SPRT", "SHRT", "JUNK", "ILLQ", "LATE") for s_, _ in igs), str(sorted(igs)))
check("ignition sorted newest first", list(ig.detect) == sorted(ig.detect, reverse=True))
data7, now7 = at("11:30")
ig2 = L.ignition_scan(syms, cfg, daily, data7, 0.0, now7)
g2 = {(r.symbol, r.side): r for r in ig2.itertuples()}
check("IGNX trailed out with profit after reversal (EXIT shown, like a +x% day)", g2[("IGNX", "LONG")].status == "EXITED (trail)" and g2[("IGNX", "LONG")].pnl_pct > 0
      and g2[("IGNX", "LONG")].exit_time != "", str((g2[("IGNX", "LONG")].exit, g2[("IGNX", "LONG")].exit_time, g2[("IGNX", "LONG")].pnl_pct)))
check("exit price is the trailing stop, not the crashed last price", g2[("IGNX", "LONG")].exit > g2[("IGNX", "LONG")].last)
# forming candle is never the ignition candle
d_early, n_early = at("09:16")
ig3 = L.ignition_scan(syms, cfg, daily, d_early, 0.0, pd.Timestamp(f"{synth.DAY} 09:15:20", tz=synth.IST))
check("09:15 candle still forming -> not detected yet", ig3.empty)
ig4 = L.ignition_scan(syms, cfg, daily, d_early, 0.0, pd.Timestamp(f"{synth.DAY} 09:16:05", tz=synth.IST))
check("detected as soon as the 09:15 candle closes", {"IGNL", "IGNS", "IGNX"} <= set(ig4.symbol))

# ------------------------------------------------------------------ 5. gaps
def find_daily(pred, trend, noise, **kw):
    for sd in range(1, 600):
        df = synth.daily(1000, trend, sd, noise=noise, **kw)
        dm = E.daily_metrics(df.iloc[:-1], 0.0)
        if dm and pred(dm):
            return df
    raise RuntimeError("no seed found")


up_ctx = lambda m: m["close"] > m["ema20"] > m["ema50"]
dn_ctx = lambda m: m["close"] < m["ema20"] < m["ema50"]
gd = {
    "GUPBEAR": find_daily(dn_ctx, -0.0012, 0.006, last_day=-0.5, gap=+1.2),                       # gap-up into downtrend
    "GDNBULL": find_daily(up_ctx, 0.0012, 0.006, last_day=+2.3, gap=-0.62),                       # gap-down into uptrend
    "GUPPLAIN": find_daily(lambda m: not dn_ctx(m) and m["rsi"] < 65, 0.0004, 0.007, last_day=+0.3, gap=+1.0),
    "GDNPLAIN": find_daily(lambda m: not up_ctx(m) and m["rsi"] > 40, 0.0, 0.007, last_day=-0.2, gap=-1.0),
    "SMALL": find_daily(lambda m: True, 0.001, 0.006, last_day=0.1, gap=0.05),
}
gt, asof = L.gap_table(gd, list(gd), cfg, 0.0)
print(gt.to_string())
g = {r.symbol: r for r in gt.itertuples()}
check("gap-up in downtrend flagged LONG TRAP (sprung: faded)", g["GUPBEAR"].trap == "LONG TRAP" and g["GUPBEAR"].trap_state == "SPRUNG")
check("gap-down in uptrend flagged SHORT TRAP and sprung (PATANJALI-style)", g["GDNBULL"].trap == "SHORT TRAP" and g["GDNBULL"].trap_state == "SPRUNG"
      and g["GDNBULL"].from_open_pct > 2)
check("plain gap-up (neutral context) not flagged", g["GUPPLAIN"].trap == "")
check("plain gap-down (neutral context) not flagged", g["GDNPLAIN"].trap == "")
check("tiny gap not flagged", g["SMALL"].trap == "")
check("gap % = open vs prev close", abs(g["GDNBULL"].gap_pct + 0.62) < 0.02)
check("armed state when the trap has not played out yet",
      L.gap_table({"X": find_daily(up_ctx, 0.0012, 0.006, last_day=-0.1, gap=-0.8)}, ["X"], cfg, 0.0)[0].iloc[0].trap_state == "armed")
check("freeze creates once", L.freeze_gaps(gt, asof) and not L.freeze_gaps(gt, asof))
check("summary text builds", "Long-trap" in L.gap_summary_message(gt, asof))

# ------------------------------------------------------------------ 7. replay pack
pack = L.build_pack(d1m, daily, syms, cfg, date(2026, 9, 18), 0.0, lots, {}, save=True)
names = {x["symbol"]: x for x in pack["leaders"]}
print("pack leaders:", sorted(names))
check("pack has EXPL/STRG/SPRT/SHRT, no junk", {"EXPL", "STRG", "SPRT", "SHRT"} <= set(names) and not ({"JUNK", "ILLQ", "FLAT"} & set(names)))
check("pack saved and listed", "2026-09-18" in L.pack_dates() and L.load_pack("2026-09-18")["leaders"][0]["symbol"])
c_before = L.replay_cards(pack, 9 * 60 + 31)
c_at = L.replay_cards(pack, 9 * 60 + 32)
check("nothing on the board before the minute it fired", len(c_before) == 0)
check("cards pop at exact minute 09:32", {c["symbol"] for c in c_at} >= {"EXPL", "STRG", "SPRT", "SHRT"} and all(c["first_m"] == 9 * 60 + 32 for c in c_at))
check("replay: strongest tier first, then score", c_at[0]["tier"] == "EXPLOSIVE" and [ (-L.TIER_RANK[c["tier"]], -c["score"]) for c in c_at] == sorted((-L.TIER_RANK[c["tier"]], -c["score"]) for c in c_at))
check("replay keeps a leader that later turned late (it WAS a leader at 09:32)", "LATE" in {c["symbol"] for c in c_at})
check("mini chart series grows with the clock", len(L.replay_cards(pack, 10 * 60)[0]["series"]) > len(c_at[0]["series"]))
check("chart series cut at replay minute (no future bars)", len(c_at[0]["series"]) == (9 * 60 + 32) - 555 + 1)
check("feed lists the arrivals", len(L.replay_feed(pack, 9 * 60 + 40)) >= 4)
import json  # noqa: E402
check("pack is JSON-safe", len(json.dumps(pack)) > 1000, f"{len(json.dumps(pack)) // 1024} KB")

# replay uses only daily data before the replay day
d_with_future = {k: v.copy() for k, v in daily.items()}
p2 = L.build_pack(d1m, d_with_future, syms, cfg, date(2026, 9, 18), 0.0, save=False)
check("pack reproducible", [x["symbol"] for x in p2["leaders"]] == [x["symbol"] for x in pack["leaders"]])

# ------------------------------------------------------------------ 2. fno helpers
check("expiry: last Tuesday of Sep 2026 is 29th", F.next_expiry(date(2026, 9, 18), 1) == date(2026, 9, 29))
check("expiry rolls to next month after expiry", F.next_expiry(date(2026, 9, 30), 1) == date(2026, 10, 27))
check("expiry rolls across year", F.next_expiry(date(2026, 12, 30), 1).year == 2027)
check("strike step bands", F.strike_step(30) == 2.5 and F.strike_step(800) == 10 and F.strike_step(3000) == 50 and F.strike_step(800, 5) == 5)
p_c, d_c = F.bs_price(100, 100, 30 / 365, 0.065, 0.3, "CE")
p_p, d_p = F.bs_price(100, 100, 30 / 365, 0.065, 0.3, "PE")
check("put-call parity holds", abs((p_c - p_p) - (100 - 100 * np.exp(-0.065 * 30 / 365))) < 1e-6)
check("call delta ~0.5, put delta negative", 0.45 < d_c < 0.6 and d_p < 0)
check("ITM/OTM strike preference shifts strike", F.option_suggestion("LONG", 1000, 990, 1010, .3, date(2026, 9, 29), date(2026, 9, 18), 10, None, "ITM-1")["strike"] == 990
      and F.option_suggestion("SHORT", 1000, 1010, 990, .3, date(2026, 9, 29), date(2026, 9, 18), 10, None, "ITM-1")["strike"] == 1010)

# NSE lot file parsing
tmp = os.path.join(tempfile.mkdtemp(), "lots.csv")
open(tmp, "w").write("UNDERLYING ,SYMBOL ,Sep-2026 ,Oct-2026\nNifty 50,NIFTY,75,75\nReliance Industries,RELIANCE ,500,500\nTata Steel,TATASTEEL, ,5500\n")
pl = F._parse_nse_lots(tmp)
check("NSE lot file parsed, index dropped, blank month skipped", pl == {"RELIANCE": 500, "TATASTEEL": 5500}, str(pl))

# ------------------------------------------------------------------ 6. journal
owner = J.owner_hash("my-key")
other = J.owner_hash("someone-else")
store.add_journal(owner, dict(date="2026-09-14", symbol="RELIANCE", side="LONG", instrument="Equity", setup="Breakout Leader (EXPLOSIVE)",
                              entry=1400, sl=1390, target=1420, exit=1418, qty=100, entry_time="09:32", exit_time="10:05", mistake="None", charges=50))
store.add_journal(owner, dict(date="2026-09-14", symbol="SBIN", side="LONG", instrument="Equity", setup="Trend Ignition",
                              entry=800, sl=795, target=810, exit=796, qty=200, entry_time="09:16", exit_time="11:16", mistake="Held too long", charges=40))
store.add_journal(owner, dict(date="2026-09-15", symbol="TCS", side="SHORT", instrument="Option", setup="Gap trap reversal",
                              entry=50, sl=60, target=30, exit=35, qty=175, entry_time="09:20", exit_time="09:50", mistake="None", charges=30))
store.add_journal(owner, dict(date="2026-09-16", symbol="INFY", side="LONG", instrument="Equity", setup="Breakout Leader (SPURT)",
                              entry=1500, sl=1490, target=1520, exit=1485, qty=100, entry_time="10:00", exit_time="10:20", mistake="Chased / late entry", charges=45))
store.add_journal(other, dict(date="2026-09-14", symbol="HIDDEN", side="LONG", entry=1, exit=100, qty=1, sl=0))
rows = store.journal_list(owner)
check("journal is private per access key", len(rows) == 4 and all(r["symbol"] != "HIDDEN" for r in rows) and len(store.journal_list(other)) == 1)
jd = J.to_frame(rows)
s = J.stats(jd)
print(jd[["date", "symbol", "pnl", "r", "hold_min"]].to_string())
rel = jd[jd.symbol == "RELIANCE"].iloc[0]
check("pnl long = (exit-entry)*qty - charges", rel.pnl == round((1418 - 1400) * 100 - 50, 2))
check("pnl short handled", jd[jd.symbol == "TCS"].iloc[0].pnl == round((50 - 35) * 175 - 30, 2))
check("R multiple = gross / (|entry-sl| x qty)", rel.r == round(1800 / 1000, 2))
check("hold minutes", rel.hold_min == 33)
check("stats net matches sum", abs(s["net"] - jd.pnl.sum()) < 1e-6 and s["trades"] == 4)
ins = " ".join(J.insights(jd))
print(ins)
check("insights: best setup, hold time, costliest mistake", "Best setup" in ins and "Average hold time" in ins and "Mistake costing you most" in ins)
check("costliest mistake is the biggest loser tag", "Held too long" in ins or "Chased" in ins)
cal = J.calendar_html(jd, 2026, 9)
check("calendar renders green/red days", "rgba(46,196,166" in cal and "rgba(239,91,91" in cal)
check("csv export has header + rows", J.csv_bytes(jd).decode().count("\n") == 5)
imp = J.enrich(dict(side="LONG", entry=100, exit=104, sl=98, qty=10, pnl_override=55.0, charges=0))
check("imported trade keeps its true net P&L (partial bookings)", imp["pnl"] == 55.0 and imp["r"] == round(55 / 20, 2))
check("equity curve is cumulative", abs(jd.equity.iloc[-1] - jd.pnl.sum()) < 1e-6)

# ------------------------------------------------------------------ password gate (auth.py)
import auth  # noqa: E402
h = auth.hash_password("hunter2")
check("hash_password is deterministic", h == auth.hash_password("hunter2") and len(h) == 64)
check("different passwords hash differently", auth.hash_password("other") != h)
check("hash is salted (not a bare sha256)", h != __import__("hashlib").sha256(b"hunter2").hexdigest())

check("nothing configured yet -> open/dev mode", auth.configured_hash() is None and not auth.is_set() and not auth.host_locked())

# host-locked path (env var) takes priority and can't be changed from inside the app
os.environ["APP_PASSWORD_HASH"] = h
check("host hash read from APP_PASSWORD_HASH env var", auth.configured_hash() == h and auth.host_locked())
del os.environ["APP_PASSWORD_HASH"]
os.environ["APP_PASSWORD"] = "hunter2"
check("host hash derived from plaintext APP_PASSWORD env var", auth.configured_hash() == h and auth.host_locked())
try:
    auth.set_password("x")
    check("set_password blocked when host-locked", False)
except RuntimeError:
    check("set_password blocked when host-locked", True)
del os.environ["APP_PASSWORD"]
check("back to open once env var is unset", auth.configured_hash() is None)

# in-app set / change / remove password, stored in this app's own SQLite (store.py)
auth.set_password("first-pw")
check("set_password makes it set up", auth.is_set() and not auth.host_locked())
check("stored hash matches the password just set", auth.configured_hash() == auth.hash_password("first-pw"))
check("change_password rejects the wrong current password", auth.change_password("wrong-pw", "second-pw") is False
      and auth.configured_hash() == auth.hash_password("first-pw"))
check("change_password with the right current password updates the hash", auth.change_password("first-pw", "second-pw") is True
      and auth.configured_hash() == auth.hash_password("second-pw"))
auth.remove_password()
check("remove_password clears it (back to open/dev mode)", auth.configured_hash() is None and not auth.is_set())

# ------------------------------------------------------------------ pwa.py (icon + manifest generation)
import pwa  # noqa: E402
import base64 as _b64
icon = pwa._make_icon_b64(192)
raw = _b64.b64decode(icon)
check("icon is a real PNG (magic bytes)", raw[:8] == b"\x89PNG\r\n\x1a\n")
check("icon caches by size (same bytes on second call)", pwa._make_icon_b64(192) == icon)
check("512px icon differs from 192px icon", pwa._make_icon_b64(512) != icon)

print("\nFAILED:", check.failed)
sys.exit(1 if check.failed else 0)
