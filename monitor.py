"""
monitor.py - the backend that stays with you until the trade is managed.

Runs without the browser: every N seconds it pulls 1-minute data for each open trade, applies
partial booking + breakeven + trailing SL + EOD exit, saves state, and pushes Telegram alerts.
Optionally re-runs the strong-buy scan on a universe and alerts on fresh signals.

    python monitor.py --interval 20
    python monitor.py --interval 20 --scan "Nifty 200" --scan-every 180
    python monitor.py --always          # ignore market hours (testing / replay of last session)

Telegram (optional):  set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import engine as E
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=20, help="seconds between trade checks")
    ap.add_argument("--scan", default="", help='universe to scan for strong buys, e.g. "Nifty 200"')
    ap.add_argument("--scan-every", type=int, default=180, help="seconds between scans")
    ap.add_argument("--always", action="store_true", help="run outside market hours too")
    a = ap.parse_args()
    store.init()
    print("monitor started - Ctrl+C to stop", flush=True)
    state: dict = {}
    last_scan = 0.0
    while True:
        try:
            store.heartbeat()
            cfg = E.load_cfg()                                     # hot-reloads settings saved from the dashboard
            if a.always or E.market_open():
                manage_open(cfg)
                if a.scan and time.time() - last_scan >= a.scan_every:
                    scan_and_alert(a.scan, cfg, state, a.always)
                    last_scan = time.time()
        except KeyboardInterrupt:
            break
        except Exception as e:                                      # never die on a bad tick
            print("monitor error:", repr(e), flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
