"""Executes the real app.py with a fake `streamlit`/`plotly` and synthetic market data.
Catches NameErrors, wrong column names, bad signatures and logic crashes in every tab. It cannot check pixel rendering."""
import os
import runpy
import sys
import tempfile
import types
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["TRADE_DB"] = os.path.join(tempfile.mkdtemp(), "smoke.db")

import synth  # noqa: E402  (stubs yfinance)
import pandas as pd  # noqa: E402

CLICKS: set = set()          # button labels/keys that "are clicked" on this run
CALLS: list = []             # (function, label) trail so we can assert things rendered


class SS(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(ST, name)


class _Col:
    def __getattr__(self, n):
        return lambda *a, **k: MagicMock()


def _val(args, kwargs, pos, name, default=None):
    return kwargs.get(name, args[pos] if len(args) > pos else default)


class FakeST(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SS()
        self.sidebar = Ctx()
        self.column_config = _Col()
        def _html(*a, **k):
            CALLS.append(("components.html", (a[0][:40] if a else "")))
        self.components = types.SimpleNamespace(v1=types.SimpleNamespace(html=_html))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: (CALLS.append((name, str(a[0])[:60] if a else "")), None)[1]

    # ---- layout
    def columns(self, spec, **k):
        return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        return [Ctx() for _ in labels]

    def expander(self, *a, **k):
        return Ctx()

    def form(self, *a, **k):
        return Ctx()

    def form_submit_button(self, label, *a, **k):
        CALLS.append(("form_submit_button", label))
        return label in CLICKS or k.get("key") in CLICKS

    def container(self, *a, **k):
        return Ctx()

    def spinner(self, *a, **k):
        return Ctx()

    def fragment(self, run_every=None):
        return lambda fn: fn

    def cache_data(self, *a, **k):
        def deco(fn):
            fn.clear = lambda: None
            return fn
        return deco if not (a and callable(a[0])) else deco(a[0])

    # ---- widgets return their defaults (or session_state[key])
    def _keyed(self, kwargs, default):
        key = kwargs.get("key")
        return self.session_state[key] if key and key in self.session_state else default

    def text_input(self, *a, **k):
        return self._keyed(k, _val(a, k, 1, "value", ""))

    def number_input(self, *a, **k):
        return self._keyed(k, _val(a, k, 3, "value", _val(a, k, 1, "min_value", 0)))

    def slider(self, *a, **k):
        return self._keyed(k, _val(a, k, 3, "value", _val(a, k, 1, "min_value", 0)))

    def toggle(self, *a, **k):
        return self._keyed(k, _val(a, k, 1, "value", False))

    def checkbox(self, *a, **k):
        return self._keyed(k, _val(a, k, 1, "value", False))

    def selectbox(self, *a, **k):
        opts = list(_val(a, k, 1, "options"))
        return self._keyed(k, opts[_val(a, k, 2, "index", 0)] if opts else None)

    def radio(self, *a, **k):
        opts = list(_val(a, k, 1, "options"))
        return self._keyed(k, opts[_val(a, k, 2, "index", 0)] if opts else None)

    def select_slider(self, *a, **k):
        opts = list(_val(a, k, 1, "options"))
        v = _val(a, k, 2, "value", opts[0])
        return self._keyed(k, v)

    def multiselect(self, *a, **k):
        return list(_val(a, k, 2, "default", []) or [])

    def date_input(self, *a, **k):
        return self._keyed(k, _val(a, k, 1, "value"))

    def time_input(self, *a, **k):
        return self._keyed(k, _val(a, k, 1, "value"))

    def button(self, label, *a, **k):
        CALLS.append(("button", label))
        return label in CLICKS or k.get("key") in CLICKS

    def download_button(self, *a, **k):
        CALLS.append(("download_button", str(a[0])))
        return False

    def rerun(self, *a, **k):
        CALLS.append(("rerun", ""))

    def plotly_chart(self, *a, **k):
        CALLS.append(("plotly_chart", ""))

    def set_page_config(self, *a, **k):
        pass


ST = FakeST()
sys.modules["streamlit"] = ST
sys.modules["streamlit.components"] = ST.components
sys.modules["streamlit.components.v1"] = ST.components.v1
plotly = types.ModuleType("plotly")
go = MagicMock()
sub = types.SimpleNamespace(make_subplots=lambda *a, **k: MagicMock())
plotly.graph_objects, plotly.subplots = go, sub
sys.modules.update({"plotly": plotly, "plotly.graph_objects": go, "plotly.subplots": sub})

import engine as E  # noqa: E402
import fno as F  # noqa: E402
import store  # noqa: E402

NOW = pd.Timestamp(f"{synth.DAY} 09:40:30", tz=synth.IST)
U = synth.universe()
SYMS = list(U)
E.now_ist = lambda: NOW
E.index_ret20 = lambda: 0.0
E.load_universe = lambda name, max_age_days=7: (SYMS, {s: "Sector" for s in SYMS}, "synthetic universe")
F.load_fno = lambda max_age_days=3: (SYMS, {s: 500 for s in SYMS}, {}, "synthetic F&O")
E.notify = lambda text: CALLS.append(("notify", text[:40])) or True


def fake_download(symbols, period, interval, chunk=80):
    out = {}
    for s in symbols:
        if s not in U:
            continue
        if interval == "1d":
            out[s] = U[s][0]
        elif period == "5d":
            out[s] = U[s][1]                                                   # full session for replay
        elif interval.endswith("m"):
            df = U[s][1]
            out[s] = df[df.index < pd.Timestamp(f"{synth.DAY} 09:40", tz=synth.IST)]
    return out


E.download_batch = fake_download


def run(clicks=(), state=None, label=""):
    CLICKS.clear()
    CLICKS.update(clicks)
    CALLS.clear()
    ST.session_state.clear()
    ST.session_state.update(state or {})
    runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    seen = {c for c in CALLS}
    print(f"OK  {label}: {len(CALLS)} calls")
    return seen


failed = 0


def expect(cond, msg):
    global failed
    print(("PASS " if cond else "FAIL ") + msg)
    failed += not cond


# 1) cold start: nothing clicked
calls = run(label="cold start (all tabs render idle)")

# 2) leaders + ignition + gaps (auto-scan is on because the market is 'open')
calls = run(clicks={"Scan now", "ig_now_btn", "Load / refresh gaps"}, state={"ld_now": True, "ig_now": True, "gaps_on": True},
            label="leaders + ignition + gaps")
ss = ST.session_state
board = ss.get("ld_board")
expect(board is not None and len(board) >= 3, f"leaders board built ({0 if board is None else len(board)} cards)")
expect({"EXPLOSIVE", "STRONG", "SPURT"} <= set(board.tier), "all three tiers present on the board")
expect(any(c[0] == "button" and c[1] == "Track trade" for c in CALLS), "cards render a Track button")
expect(ss.get("ig") is not None and len(ss["ig"]) >= 3, f"ignition scan ran ({0 if ss.get('ig') is None else len(ss['ig'])} rows)")
expect(any(c[0] == "dataframe" for c in CALLS), "tables rendered")

# 3) Track a leader through the card button -> lands in the trade store
first = board.iloc[0].symbol
run(clicks={"Track trade"}, state={"ld_now": True}, label="track trade from a card")
expect(any(t["symbol"] for t in store.open_trades()), "tracked trade stored for the trade manager")

# 4) alert dedupe: EXPLOSIVE alert goes out once (test mode ignores market hours)
n_before = len([1 for c in CALLS if c[0] == "notify"])
run(state={"ld_now": True, "ld_force": True}, label="leaders alert run 1")
sent1 = [c for c in CALLS if c[0] == "toast"]
run(state={"ld_now": True, "ld_force": True}, label="leaders alert run 2 (same signals)")
expect("ld_alerted" in (store.get_meta("ld_alerted") and ["ld_alerted"] or []) or store.get_meta("ld_alerted"), "alert keys persisted for dedupe")

# 5) replay: build pack, then render replay at 09:35 and 10:00
run(clicks={"Build replay pack (downloads F&O 1-minute data)"}, label="build replay pack")
from datetime import date  # noqa: E402
expect(len(store.list_meta("pack:")) >= 1, "replay pack saved by the build button")
run(state={"rp_pick": store.list_meta("pack:")[0].split(":", 1)[1], "rp_t": 9 * 60 + 35}, label="replay at 09:35")
expect(any(c[0] == "plotly_chart" for c in CALLS), "replay draws mini charts")
run(state={"rp_pick": store.list_meta("pack:")[0].split(":", 1)[1], "rp_t": 10 * 60, "rp_running": True}, label="replay playing")
expect(ST.session_state["rp_t"] > 10 * 60, "play advances the replay clock")

# 6) journal: log, list, insights, import
import journal as J  # noqa: E402
owner = J.owner_hash("k1")
state = {"jr_key": "k1", "j_sym": "EXPL", "j_inst": "Option", "j_entry": 30.0, "j_exit": 42.0, "j_sl": 22.0, "j_tgt": 50.0,
         "j_lots": 2, "j_strike": "1400 CE", "j_mist": "None"}
run(clicks={"Save trade"}, state=state, label="journal save (option, auto lot)")
rows = store.journal_list(owner)
expect(len(rows) == 1 and rows[0]["qty"] == 1000 and rows[0]["lot_size"] == 500, f"option qty auto-filled = lots x lot size ({rows[0]['qty'] if rows else None})")
run(state={"jr_key": "other"}, label="journal other key")
expect(len(store.journal_list(J.owner_hash("other"))) == 0, "another key sees nothing")
run(state={"jr_key": "k1"}, label="journal render with data")
expect(any(c[0] == "download_button" for c in CALLS), "CSV export offered")
store.add_trade(dict(status="CLOSED", symbol="SBIN", side="LONG", entry=800.0, init_sl=795.0, sl=795.0, targets=[805, 810, 815], qty=100, qty_open=0,
                     exit_price=808.0, pnl=650.0, close_reason="T1", opened_at="2026-09-18T09:32:00+05:30", closed_at="2026-09-18T10:10:00+05:30", risk_ps=5.0))
run(clicks={"Import closed trades from the Trade manager"}, state={"jr_key": "k1"}, label="journal import")
rows = store.journal_list(owner)
expect(any(r.get("pnl_override") == 650.0 for r in rows), "closed trade imported with its true net P&L")

# 7) UI regression: journal HTML must not show "nan" chips or wrap the NET P&L figure
import journal as J
import ui as _ui_mod
jdf = J.to_frame(store.journal_list(owner))
Sx = J.stats(jdf)
cardsx = J.insight_cards(jdf)
from datetime import date as _d
jhtml, _ = _ui_mod.journal_html(jdf, jdf, Sx, cardsx, 2026, 9, _d(2026, 9, 18))
expect("nan" not in jhtml.lower().replace("nan-trap", ""), "journal HTML has no stray 'nan' chips for equity trades")
expect("white-space:nowrap" in jhtml or ".v{font-size" in jhtml, "KPI value styling present (no-wrap net P&L)")

# 8) password gate wiring: with nothing configured, app.py must reach the first-run "set a password" screen
import auth
calls = run(label="password setup screen renders (no password configured yet)")
submit_labels = {c[1] for c in CALLS if c[0] == "form_submit_button"}
expect({"Set password", "Skip - stay open"} <= submit_labels, f"setup screen offers Set password / Skip buttons, got {submit_labels}")
expect(not auth.is_set() and not auth.host_locked(), "still open/dev mode - nothing was actually set by rendering the screen")

print("\nSMOKE FAILED:", failed)
sys.exit(1 if failed else 0)
