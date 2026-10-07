"""Angel One SmartAPI feed via plain REST (no SDK, no extra installs - only `requests`).
Same return shape as data.fetch_live. Secrets: ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN, ANGEL_TOTP_SECRET."""
import base64
import datetime as dt
import hashlib
import hmac
import struct
import time

import pandas as pd
import requests

from data import COLS, IST, split_today

BASE = "https://apiconnect.angelone.in"
SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
INDEX_TOKENS = {"^NSEI": ("NSE", "99926000"), "^NSEBANK": ("NSE", "99926009"), "^BSESN": ("BSE", "99919000")}
_state = {"jwt": None, "day": None, "tokens": {}}


def totp(secret, now=None, digits=6, step=30):
    """RFC 6238 TOTP (stdlib only)."""
    key = base64.b32decode(secret.replace(" ", "").upper() + "=" * (-len(secret.replace(" ", "")) % 8))
    counter = int((time.time() if now is None else now) // step)
    h = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = h[-1] & 15
    return str((struct.unpack(">I", h[o:o + 4])[0] & 0x7FFFFFFF) % 10 ** digits).zfill(digits)


def _headers(sec, jwt=None):
    h = {"Content-Type": "application/json", "Accept": "application/json", "X-UserType": "USER",
         "X-SourceID": "WEB", "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
         "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": sec["ANGEL_API_KEY"]}
    if jwt:
        h["Authorization"] = f"Bearer {jwt}"
    return h


def _login(sec):
    r = requests.post(f"{BASE}/rest/auth/angelbroking/user/v1/loginByPassword", headers=_headers(sec), timeout=30,
                      json={"clientcode": sec["ANGEL_CLIENT_CODE"], "password": sec["ANGEL_PIN"],
                            "totp": totp(sec["ANGEL_TOTP_SECRET"])})
    j = r.json()
    if not j.get("status"):
        raise RuntimeError(f"Angel One login failed: {j.get('message')} ({j.get('errorcode')})")
    return j["data"]["jwtToken"]


def _jwt(sec, force=False):
    if force or not _state["jwt"] or _state["day"] != dt.date.today():
        _state["jwt"], _state["day"] = _login(sec), dt.date.today()
    return _state["jwt"]


def _tokens(symbols):
    missing = [s for s in symbols if s not in _state["tokens"]]
    if missing:
        master = requests.get(SCRIP_URL, timeout=60).json()
        want = {s + "-EQ": s for s in missing}
        for m in master:
            if m.get("exch_seg") == "NSE" and m.get("symbol") in want:
                _state["tokens"][want[m["symbol"]]] = m["token"]
    return {s: _state["tokens"][s] for s in symbols if s in _state["tokens"]}


def _candles(sec, token, frm, to, exch="NSE"):
    body = {"exchange": exch, "symboltoken": token, "interval": "ONE_MINUTE", "fromdate": frm, "todate": to}
    last = None
    for attempt in range(3):
        r = requests.post(f"{BASE}/rest/secure/angelbroking/historical/v1/getCandleData",
                          headers=_headers(sec, _jwt(sec, force=attempt == 1)), json=body, timeout=30)
        try:
            res = r.json()
        except ValueError:
            res = {"message": r.text[:200]}
        if res.get("status") and res.get("data"):
            d = pd.DataFrame(res["data"], columns=["time"] + COLS)
            d["time"] = pd.to_datetime(d["time"])
            d = d.set_index("time")
            d.index = d.index.tz_convert(IST) if d.index.tz is not None else d.index.tz_localize(IST)
            return d.astype(float)
        last = res
        time.sleep(1.0)  # rate limit or expired token: retry (2nd try re-logins)
    if last and last.get("message"):
        raise RuntimeError(f"Angel One candle error: {last.get('message')} ({last.get('errorcode')})")
    return None


def fetch_live(symbols, sec):
    now = pd.Timestamp.now(tz=IST)
    frm = (now - pd.Timedelta(days=7)).strftime("%Y-%m-%d 09:15")
    to = now.strftime("%Y-%m-%d %H:%M")
    toks = {s: ("NSE", t) for s, t in _tokens(symbols).items()}
    toks.update(INDEX_TOKENS)
    out = {}
    for sym, (exch, tok) in toks.items():
        d = _candles(sec, tok, frm, to, exch)
        if d is not None and not d.empty:
            out[sym] = split_today(d)
        time.sleep(0.4)  # stay under the historical-data rate limit
    return out
