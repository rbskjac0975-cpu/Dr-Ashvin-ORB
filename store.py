"""store.py - SQLite persistence shared by the dashboard and monitor.py (trades, events, watchlist, heartbeat)."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd

DB = os.environ.get("TRADE_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db"))


def _j(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _run(sql: str, args: tuple = (), fetch: bool = False):
    with closing(sqlite3.connect(DB, timeout=15)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, args)
        rows = cur.fetchall() if fetch else None
        c.commit()
        return rows if fetch else cur.lastrowid


def init() -> None:
    _run("CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, symbol TEXT, data TEXT)")
    _run("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, text TEXT)")
    _run("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    _run("CREATE TABLE IF NOT EXISTS journal(id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, data TEXT)")


# ---- trades
def add_trade(t: dict) -> int:
    tid = _run("INSERT INTO trades(status, symbol, data) VALUES(?,?,?)", (t["status"], t["symbol"], json.dumps(t, default=_j)))
    t["id"] = tid
    update_trade(t)
    return tid


def update_trade(t: dict) -> None:
    _run("UPDATE trades SET status=?, symbol=?, data=? WHERE id=?",
         (t["status"], t["symbol"], json.dumps(t, default=_j), t["id"]))


def _load(rows) -> list:
    out = []
    for r in rows:
        d = json.loads(r["data"])
        d["id"] = r["id"]
        out.append(d)
    return out


def open_trades() -> list:
    return _load(_run("SELECT * FROM trades WHERE status='OPEN' ORDER BY id", fetch=True))


def closed_trades(limit: int = 200) -> list:
    return _load(_run("SELECT * FROM trades WHERE status!='OPEN' ORDER BY id DESC LIMIT ?", (limit,), fetch=True))


def open_for(symbol: str):
    rows = _load(_run("SELECT * FROM trades WHERE status='OPEN' AND symbol=?", (symbol,), fetch=True))
    return rows[0] if rows else None


# ---- events
def add_event(text: str) -> None:
    _run("INSERT INTO events(ts, text) VALUES(?,?)", (datetime.now().strftime("%d %b %H:%M:%S"), text))


def recent_events(n: int = 40) -> list:
    return [(r["ts"], r["text"]) for r in _run("SELECT ts, text FROM events ORDER BY id DESC LIMIT ?", (n,), fetch=True)]


# ---- meta / heartbeat / watchlist
def set_meta(k: str, v) -> None:
    _run("INSERT INTO meta(k, v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v, default=_j)))


def get_meta(k: str, default=None):
    rows = _run("SELECT v FROM meta WHERE k=?", (k,), fetch=True)
    return json.loads(rows[0]["v"]) if rows else default


def heartbeat() -> None:
    set_meta("monitor_hb", time.time())


def monitor_alive(max_age: float = 90.0) -> bool:
    hb = get_meta("monitor_hb")
    return bool(hb and time.time() - hb < max_age)


def save_watchlist(symbols: list) -> None:
    set_meta("watchlist", list(symbols))


def load_watchlist() -> list:
    return get_meta("watchlist", []) or []


def list_meta(prefix: str) -> list:
    """Keys in the meta table that start with prefix (used for saved replay packs / gap snapshots)."""
    rows = _run("SELECT k FROM meta WHERE k LIKE ? ORDER BY k", (prefix + "%",), fetch=True)
    return [r["k"] for r in rows]


# ---- trading journal (private per access key: rows are only ever read back with the same owner hash)
def add_journal(owner: str, e: dict) -> int:
    return _run("INSERT INTO journal(owner, data) VALUES(?,?)", (owner, json.dumps(e, default=_j)))


def journal_list(owner: str) -> list:
    out = []
    for r in _run("SELECT * FROM journal WHERE owner=? ORDER BY id", (owner,), fetch=True):
        d = json.loads(r["data"])
        d["id"] = r["id"]
        out.append(d)
    return out


def journal_delete(owner: str, jid: int) -> None:
    _run("DELETE FROM journal WHERE owner=? AND id=?", (owner, jid))
