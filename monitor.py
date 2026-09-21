"""
monitor.py - the backend that stays with you until the trade is managed.

Runs without the browser: every N seconds it pulls 1-minute data for each open trade, applies
partial booking + breakeven + trailing SL + EOD exit, saves state, and pushes Telegram alerts.
Optionally re-runs the strong-buy scan on a universe and alerts on fresh signals.

    python monitor.py --interval 20
    python monitor.py --interval 20 --scan "Nifty 200" --scan-every 180
    python monitor.py --always          # ignore market hours (testing / replay of last session)
    python monitor.py --leaders         # + Breakout Leaders auto-scan from 09:16, Telegram on new EXPLOSIVE leaders,
                                        #   opening-gap snapshot at the bell, and the Day Replay pack after the close

Telegram (optional):  set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import engine as E
import fno as F
import leaders as L
import store


def alert(text: str) -> None:
    print(f"[{E.now_ist().strftime('%H:%M:%S')}] {text}", flush=True)
    store.add_event(text)
    E.notify(text)


def manage_open(cfg: E.Cfg) -> None:
    trades = store.open_trades()
    if not trades:
        return
    data = E.download_batch(sorted({t["symbol"] for t in trades}), "1d", "1m", chunk=40)
    for t in trades:
        t, events = E.manage_trade(t, data.get(t["symbol"]))
        store.update_trade(t)
        for e in events:
            alert(e)


def scan_and_alert(name: str, cfg: E.Cfg, state: dict, always: bool) -> None:
    if time.time() - state.get("daily_ts", 0) > 1800 or "daily" not in state:      # refresh daily data every 30 min
        syms, ind, _ = E.load_universe(name)
        state.update(syms=syms, ind=ind, daily=E.download_batch(syms, "6mo", "1d"),
                     nret=E.index_ret20(), daily_ts=time.time())
    breadth = E.compute_breadth(state["daily"], state["ind"])
    res = E.run_scan(state["syms"], state["ind"], cfg, daily=state["daily"], nifty_ret20=state["nret"], breadth=breadth)
    print(f"[{E.now_ist().strftime('%H:%M:%S')}] scan: breadth {breadth['verdict']} | "
          f"{0 if res.empty else int((res.status == 'STRONG BUY').sum())} strong buy", flush=True)
    if res.empty:
        return
    done = set(store.get_meta("alerted", []) or [])
    for _, r in res[res.status == "STRONG BUY"].iterrows():
        key = f"{date.today()}|{r['symbol']}|{r['sig_time']}"
        if key not in done:
            done.add(key)
            alert(E.strong_buy_message(r))
    store.set_meta("alerted", sorted(done)[-500:])


def leaders_tick(cfg: E.Cfg, st: dict, always: bool) -> None:
    """Breakout Leaders auto-scan. First scan at 09:16 IST; alerts on NEW, fresh, mood-compatible EXPLOSIVE leaders."""
    now = E.now_ist()
    if not always and now.hour * 60 + now.minute < 556:
        return
    if "fno" not in st or time.time() - st.get("daily_ts", 0) > 1800:            # refresh daily data every 30 min
        syms, lots, steps, note = F.load_fno()
        st.update(fno=syms, lots=lots, steps=steps, daily=E.download_batch(syms, "6mo", "1d"),
                  nret=E.index_ret20(), daily_ts=time.time())
        print(f"[{now.strftime('%H:%M:%S')}] F&O universe: {note}", flush=True)
    data = E.download_batch(st["fno"], "1d", "1m", chunk=40)
    if "breadth" not in st or time.time() - st.get("breadth_ts", 0) > 120:
        st.update(breadth=E.compute_breadth(st["daily"], {s: "NA" for s in st["fno"]}), breadth_ts=time.time())
    board, stats = L.scan_leaders(st["fno"], cfg, st["daily"], data, st["nret"], st["breadth"], st["lots"], st["steps"])
    print(f"[{E.now_ist().strftime('%H:%M:%S')}] leaders: {stats['leaders']} "
          f"(junk {stats['junk']}, no setup {stats['no_setup']}, late {stats['late']}) | breadth {st['breadth']['verdict']}", flush=True)
    if board.empty:
        return
    done = set(store.get_meta("ld_alerted", []) or [])                            # shared with the dashboard
    day = E.now_ist().date().isoformat()
    for r in board[(board.tier == "EXPLOSIVE") & board.mood_ok & (board.age_min <= cfg.fresh_min)].to_dict("records"):
        key = f"{day}|{r['symbol']}|{r['side']}|{r['tier']}"
        if key not in done:
            done.add(key)
            alert(L.leader_message(r))
    store.set_meta("ld_alerted", sorted(done)[-500:])


def freeze_gaps_once(st: dict) -> None:
    """At/after 09:16: re-download daily bars (they now contain today's open), save the gap snapshot, send a summary."""
    today = E.now_ist().date().isoformat()
    if st.get("gap_day") == today:
        return
    syms = st.get("fno") or F.load_fno()[0]
    daily = E.download_batch(syms, "6mo", "1d")
    gt, asof = L.gap_table(daily, syms, E.load_cfg(), E.index_ret20())
    if gt.empty or asof != today:
        return
    st["gap_day"] = today
    if L.freeze_gaps(gt, asof):
        alert(L.gap_summary_message(gt, asof))


def build_day_pack(cfg: E.Cfg, st: dict) -> None:
    """After the close: replay the whole session bar by bar and save the pack for the Day Replay tab."""
    day = E.now_ist().date()
    if st.get("pack_day") == day.isoformat() or "fno" not in st:
        return
    data = E.download_batch(st["fno"], "1d", "1m", chunk=40)
    pack = L.build_pack(data, st["daily"], st["fno"], cfg, day, st["nret"], st["lots"], st["steps"])
    st["pack_day"] = day.isoformat()
    alert(f"📦 Day Replay pack saved for {day}: {len(pack['leaders'])} leaders")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=20, help="seconds between trade checks")
    ap.add_argument("--scan", default="", help='universe to scan for strong buys, e.g. "Nifty 200"')
    ap.add_argument("--scan-every", type=int, default=180, help="seconds between scans")
    ap.add_argument("--always", action="store_true", help="run outside market hours too")
    ap.add_argument("--leaders", action="store_true", help="run the Breakout Leaders auto-scan (from 09:16), gap snapshot and replay pack")
    ap.add_argument("--leaders-every", type=int, default=60, help="seconds between leader scans")
    a = ap.parse_args()
    store.init()
    print("monitor started - Ctrl+C to stop", flush=True)
    state: dict = {}
    lstate: dict = {}
    last_scan = last_leaders = 0.0
    while True:
        try:
            store.heartbeat()
            cfg = E.load_cfg()                                     # hot-reloads settings saved from the dashboard
            if a.always or E.market_open():
                manage_open(cfg)
                if a.scan and time.time() - last_scan >= a.scan_every:
                    scan_and_alert(a.scan, cfg, state, a.always)
                    last_scan = time.time()
                if a.leaders and time.time() - last_leaders >= a.leaders_every:
                    leaders_tick(cfg, lstate, a.always)
                    last_leaders = time.time()
                    if E.now_ist().hour * 60 + E.now_ist().minute >= 556:
                        freeze_gaps_once(lstate)
            elif a.leaders and not a.always:
                n = E.now_ist()
                if n.weekday() < 5 and n.hour * 60 + n.minute >= 931:             # market closed -> save today's replay pack once
                    build_day_pack(cfg, lstate)
        except KeyboardInterrupt:
            break
        except Exception as e:                                      # never die on a bad tick
            print("monitor error:", repr(e), flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
