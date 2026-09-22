"""
ui.py - HTML panels that reproduce the reference design: tier-coloured leader columns with conviction bars and
watch badges, trend-ignition cards, the 2x2 pre-open gaps panel, the "as they appeared" replay board with sparklines
and the journal dashboard (KPIs, equity curve, calendar, insight cards, trades table).

Every builder returns (html, height_px) and is meant for streamlit.components.v1.html(html, height=height, scrolling=True).
Layout uses tables / inline-blocks (no CSS grid / flex-gap) so it renders identically in old and new browsers.
"""
from __future__ import annotations

import calendar
import html
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import journal as J
import leaders as L


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _f(x, nd: int = 2) -> str:
    """Compact number: 1231.5 -> 1,231.5 ; 948.95 -> 948.95"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{nd}f}".rstrip("0").rstrip(".") if abs(x) < 1e7 else f"{x:,.0f}"


def _pct(x: Optional[float], nd: int = 2, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}%" if sign else f"{x:.{nd}f}%"


def _cls(x: Optional[float]) -> str:
    return "up" if (x or 0) >= 0 else "dn"


CSS = """<meta charset="utf-8"><style>
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{padding:4px 2px 8px 2px;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#111827;background:#ffffff;font-size:12px}
table{border-collapse:separate;border-spacing:0}
.w{width:100%}
.up{color:#15803d}.dn{color:#b91c1c}.mut{color:#6b7280}
b{font-weight:700}
.cols{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:10px 0;margin:0 -5px}
.col{vertical-align:top;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff}
.hd{padding:9px 12px;border-radius:12px 12px 0 0;color:#fff;font-weight:800;font-size:13px;letter-spacing:.3px}
.hd .cnt{display:inline-block;min-width:26px;text-align:center;padding:1px 9px;border-radius:12px;background:rgba(255,255,255,.88);font-size:11px}
.sub{padding:6px 12px;font-size:10px;font-weight:700;letter-spacing:.2px}
.body{padding:8px 8px 4px 8px}
.card{border-radius:10px;padding:9px 11px 9px 11px;margin:0 0 9px 0;border:1px solid #e5e7eb}
.card.buy{background:#f0fdf4;border-color:#bbf7d0}
.card.sell{background:#fef2f2;border-color:#fecaca}
.sym{font-size:14px;font-weight:800;color:#0f172a;letter-spacing:.2px}
.chips{text-align:right;white-space:nowrap}
.vol{display:inline-block;padding:2px 8px;border-radius:6px;color:#fff;font-weight:800;font-size:12px;margin-left:5px}
.sc{display:inline-block;padding:1px 7px;border-radius:5px;font-weight:800;font-size:11px;margin-left:5px;border:1px solid transparent}
.sc.hi{background:#dbeafe;color:#1d4ed8;border-color:#bfdbfe}
.sc.mid{background:#e0f2fe;color:#0369a1;border-color:#bae6fd}
.sc.lo{background:#f3f4f6;color:#6b7280;border-color:#e5e7eb}
.clk{display:inline-block;margin-left:7px;font-family:Consolas,Menlo,monospace;font-weight:700;font-size:12px;color:#0369a1}
.clk svg{vertical-align:-2px;margin-right:3px}
.px{margin-top:3px;font-size:15px;font-weight:800;color:#0f172a}
.px .ar{font-size:11px;margin:0 3px 0 4px}
.px .chg{font-size:12px;font-weight:800}
.meta{margin-top:2px;font-size:11px;color:#6b7280}
.conv{margin-top:6px;font-size:10px;font-weight:800;color:#6b7280;letter-spacing:.6px}
.seg{display:inline-block;width:13px;height:6px;border-radius:2px;background:#d1d5db;margin-right:2px;vertical-align:1px}
.seg.on{background:#3b82f6}
.conv b.n{color:#334155;font-size:11px;margin-left:6px;letter-spacing:0}
.stats{margin-top:6px;font-size:10.5px;color:#6b7280}
.stats b{font-weight:800}
.badge{display:inline-block;margin-top:7px;padding:3px 9px;border-radius:5px;color:#fff;font-weight:800;font-size:11px;letter-spacing:.3px}
.opt{margin-top:7px;background:#ffffff;border:1px solid #d1d5db;border-radius:8px;padding:5px 9px;font-size:12px;font-weight:800}
.opt.buy{border-color:#86efac;color:#166534}.opt.sell{border-color:#fca5a5;color:#991b1b}
.opt small{font-weight:600;color:#6b7280}
.tiny{margin-top:5px;font-size:10px;color:#6b7280;line-height:1.5}
.warn{color:#b91c1c;font-weight:700}
.empty{padding:14px 6px;text-align:center;color:#9ca3af;font-size:11px}
.ign{display:inline-block;width:26px;height:20px;line-height:20px;text-align:center;border-radius:6px;background:#7c3aed;color:#fff;font-size:12px;margin-left:5px}
/* gaps */
.gp{border:1px solid #fecaca;border-radius:12px;background:#fff7f7;overflow:hidden}
.gph{padding:8px 14px;background:#dc2626;background:-webkit-linear-gradient(left,#b91c1c,#ef4444);background:linear-gradient(90deg,#b91c1c,#ef4444);color:#fff;font-weight:800;letter-spacing:.5px;font-size:12px}
.gph .hide{display:inline-block;background:#fff;color:#b91c1c;border-radius:10px;padding:0 9px;font-size:10px;cursor:pointer;font-weight:700}
.gps{padding:5px 14px;font-size:9.5px;color:#b91c1c;font-weight:600;border-bottom:1px solid #fecaca}
.g2{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:10px}
.gb{vertical-align:top;border-radius:10px;padding:8px 10px;border:1px solid #e5e7eb;height:170px}
.gb.gu{background:#f0fdf4;border-color:#bbf7d0}.gb.gd{background:#fef2f2;border-color:#fecaca}.gb.gt{background:#fff7ed;border-color:#fed7aa}
.gt2{font-weight:800;font-size:11px;letter-spacing:.4px;padding-bottom:6px}
.gu .gt2{color:#15803d}.gd .gt2{color:#b91c1c}.gt .gt2{color:#c2410c}
.gt2 .n{float:right;display:inline-block;min-width:20px;text-align:center;padding:0 6px;border-radius:10px;background:#fff;font-size:10px;border:1px solid rgba(0,0,0,.08)}
.gl{max-height:150px;overflow:auto}
.gl td{padding:3.5px 0;font-size:11.5px;border-bottom:1px solid rgba(0,0,0,.04)}
.gl .s{font-weight:800;color:#0f172a}.gl .p{text-align:right;font-family:Consolas,Menlo,monospace;color:#6b7280;font-size:11px}
.bs{display:inline-block;padding:0 5px;border-radius:4px;background:#e0e7ff;color:#3730a3;font-size:9px;font-weight:800;margin-left:4px}
.sp{display:inline-block;padding:0 5px;border-radius:4px;background:#dc2626;color:#fff;font-size:9px;font-weight:800;margin-left:4px}
.ar2{font-size:9px;margin-right:3px}
/* replay */
.rh{padding:8px 4px 6px 4px;font-size:10.5px;font-weight:800;letter-spacing:.5px;color:#64748b}
.rc{background:#fff;border:1px solid #e2e8f0;border-radius:8px;margin:0 0 8px 0;padding:7px 10px}
.rc.fresh{border-color:#5eead4;background:#f0fdfa}
.rsym{font-weight:800;font-size:12.5px;color:#0f172a}
.rt{font-family:Consolas,Menlo,monospace;font-size:10.5px;color:#64748b;font-weight:700;text-align:right;white-space:nowrap}
.rv{display:inline-block;padding:1px 7px;border-radius:5px;background:#1e293b;color:#fff;font-weight:800;font-size:10.5px;margin-right:5px}
.rcb{display:inline-block;padding:1px 7px;border-radius:5px;font-weight:800;font-size:10px;margin-right:5px}
.rcb.COMBO{background:#dcfce7;color:#166534;border:1px solid #86efac}.rcb.ORB{background:#dbeafe;color:#1e40af;border:1px solid #93c5fd}
.rcb.SURGE{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}
.rp{display:inline-block;padding:1px 7px;border-radius:5px;font-weight:800;font-size:10.5px}
.rp.up{background:#dcfce7;color:#15803d}.rp.dn{background:#fee2e2;color:#b91c1c}
.rw{margin-top:5px;font-size:10px;font-weight:800;color:#b45309}
/* journal */
.kp{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:8px 0;margin:0 -4px}
.k{border:1px solid #e5e7eb;border-radius:10px;padding:9px 11px;background:#fff;vertical-align:top}
.k .l{font-size:9.5px;font-weight:800;color:#64748b;letter-spacing:.5px}
.k .v{font-size:20px;font-weight:800;margin-top:2px;white-space:nowrap}
.k .s{font-size:9.5px;color:#94a3b8;margin-top:1px}
.pn{border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;background:#fff;vertical-align:top}
.pn .t{font-weight:800;font-size:12px}
.pn .t small{font-weight:500;color:#94a3b8;font-size:10px;margin-left:6px}
.ins{border:1px solid #e5e7eb;border-radius:8px;padding:9px 12px;background:#fffdf8;vertical-align:top;font-size:11.5px;color:#334155}
.ins .l{font-size:9.5px;font-weight:800;letter-spacing:.6px;margin-bottom:3px}
.tt{width:100%;border-collapse:collapse;margin-top:10px}
.tt th{font-size:10px;color:#64748b;font-weight:800;text-align:left;padding:7px 8px;border-bottom:1px solid #e5e7eb;background:#fafafa}
.tt td{padding:8px 8px;border-bottom:1px solid #f1f5f9;font-size:11.5px;vertical-align:middle;white-space:nowrap}
.tt tr.row{cursor:pointer}.tt tr.row:hover td{background:#f8fafc}
.tt tr.note td{background:#fafafa;color:#475569;font-size:11px;display:none}
.sd{display:inline-block;padding:1px 9px;border-radius:5px;font-weight:800;font-size:10px}
.sd.BUY{background:#dcfce7;color:#15803d;border:1px solid #86efac}.sd.SELL{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5}
.oc{display:inline-block;padding:0 6px;border-radius:4px;background:#e0e7ff;color:#3730a3;font-size:9.5px;font-weight:800;margin-left:5px}
.st{display:inline-block;padding:1px 8px;border-radius:5px;font-weight:800;font-size:10px;border:1px solid transparent}
.dc{display:inline-block;padding:1px 8px;border-radius:5px;font-weight:800;font-size:10px}
.dc.ok{background:#dcfce7;color:#15803d}.dc.bad{background:#fef3c7;color:#b45309}
.cal td{width:14.28%;text-align:center;padding:2px}
.cal .d{display:block;height:24px;line-height:24px;border-radius:6px;font-size:10.5px;background:#eef2ff;color:#475569;font-weight:600}
.cal .d.p{background:#bbf7d0;color:#14532d}.cal .d.l{background:#fecaca;color:#7f1d1d}.cal .d.we{background:#f8fafc;color:#94a3b8}
.cal .d.today{border:2px solid #d4a017;line-height:20px}
.cal th{font-size:9.5px;color:#94a3b8;font-weight:700;padding:2px}
.pill{display:inline-block;padding:3px 12px;border:1px solid #d4a017;border-radius:14px;color:#b45309;font-weight:800;font-size:10px;letter-spacing:.4px;background:#fffbeb}
</style>"""

CLOCK = ('<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="5" fill="none" stroke="#475569" stroke-width="1.2"/>'
         '<path d="M6 3v3.2l2 1.1" stroke="#475569" fill="none" stroke-width="1.2"/></svg>')

TIERS = {
    "EXPLOSIVE": dict(icon="💥", hd="#dc2626", hd2="#ef4444", sub="BIG PLAYER ENTRY 🏛 — institutional dhamaka", sub_bg="#fff1f2", sub_fg="#b91c1c", chip="#be123c"),
    "STRONG": dict(icon="🔥", hd="#6d28d9", hd2="#8b5cf6", sub="SMART MONEY MOVE 🧠 — bade log active", sub_bg="#f5f3ff", sub_fg="#6d28d9", chip="#7c3aed"),
    "SPURT": dict(icon="⚡", hd="#0369a1", hd2="#0ea5e9", sub="EARLY SPURT ⚡ — volume waking up", sub_bg="#f0f9ff", sub_fg="#0369a1", chip="#0284c7"),
}
IGN = dict(icon="⚡", title="TREND IGNITION", hd="#c2410c", hd2="#f59e0b", sub_bg="#fffbeb", sub_fg="#b45309")


def _grad(c1: str, c2: str) -> str:
    return f"background:{c1};background:-webkit-linear-gradient(left,{c1},{c2});background:linear-gradient(90deg,{c1},{c2});"


def _wrap(body: str, script: str = "") -> str:
    return f"<html><head>{CSS}</head><body>{body}{('<script>' + script + '</script>') if script else ''}</body></html>"


# =========================================================================== 3. market breadth + toolbar
def breadth_html(b: Optional[dict]) -> Tuple[str, int]:
    if not b:
        return _wrap("<div class='empty'>Market mood appears after the first scan.</div>"), 60
    adv, dec, unch = int(b.get("adv", 0)), int(b.get("dec", 0)), int(b.get("unch", 0))
    tot = max(adv + dec + unch, 1)
    verdict = b.get("verdict", "Neutral")
    vcol = {"Bullish": "#16a34a", "Bearish": "#dc2626", "Neutral": "#d97706"}.get(verdict, "#6b7280")
    vbg = {"Bullish": "#dcfce7", "Bearish": "#fee2e2", "Neutral": "#fef3c7"}.get(verdict, "#f3f4f6")
    ratio = adv / max(dec, 1)
    green_pct = (adv + unch * 0.5) / tot * 100
    kp = [("ADVANCES", f"{adv}", "#16a34a", "▲"), ("UNCHANGED", f"{unch}", "#0f172a", "·"), ("DECLINES", f"{dec}", "#dc2626", "▼"),
          ("A / D RATIO", f"{ratio:.2f}" if dec else "∞", "#1d4ed8", "")]
    kpis = "".join(f"<td class='k' style='text-align:center'><div class='l'>{a}</div><div class='v' style='color:{c}'>{d} {b_}</div></td>"
                   for a, b_, c, d in kp)
    body = (f"<table class='w'><tr><td><span style='font-size:14px;font-weight:800'>📶 Market Breadth — Advance / Decline</span></td>"
            f"<td style='text-align:right'><span class='pill' style='border-color:{vcol};color:{vcol};background:{vbg}'>🚀 {verdict.upper()} breadth</span></td></tr></table>"
            f"<table class='kp' style='margin-top:8px'><tr>{kpis}</tr></table>"
            f"<div style='height:9px;border-radius:4px;overflow:hidden;background:#fee2e2;margin-top:10px'>"
            f"<div style='height:9px;background:#16a34a;width:{green_pct:.1f}%;display:inline-block'></div></div>"
            f"<div style='font-size:10px;color:#94a3b8;margin-top:5px'>{tot} stocks me <b class='up'>{adv} upar</b> · "
            f"<b class='dn'>{dec} neeche</b> · {unch} flat · {green_pct:.0f}% universe green</div>")
    return _wrap(body), 150


def leaders_toolbar_html(stats: dict, breadth: Optional[dict], interval_sec: int, auto_on: bool, last_scan: str,
                         counts: dict, spurt_x: float = 1.5, signal_bars: int = 11) -> Tuple[str, int]:
    verdict = (breadth or {}).get("verdict", "Neutral")
    dono = "DONO" if verdict == "Neutral" else ("BULL" if verdict == "Bullish" else "BEAR")
    dcol = {"DONO": "#0f172a", "BULL": "#16a34a", "BEAR": "#dc2626"}[dono]
    bull, bear = int((breadth or {}).get("adv", 0)), int((breadth or {}).get("dec", 0))
    tcounts = "".join(f"<span style='margin-left:6px'>{L.TIER_ICON[t]}<b style='margin-left:2px'>{counts.get(t, 0)}</b></span>"
                      for t in ("SPURT", "STRONG", "EXPLOSIVE"))
    live = "🟢 live" if auto_on else "⏸ paused"
    right = (f"<span class='pill' style='background:#fff;border-color:#cbd5e1;color:#334155'>⚡ {interval_sec}s</span>"
            f"<span class='pill' style='background:#0f172a;border-color:#0f172a;color:#fff;margin-left:6px'>{dono}</span>"
            f"<span class='pill' style='background:#fff7ed;border-color:#fdba74;color:#c2410c;margin-left:6px'>🐂 BULL {bull}</span>"
            f"<span class='pill' style='background:#eff6ff;border-color:#93c5fd;color:#1d4ed8;margin-left:6px'>🐻 BEAR {bear}</span>"
            f"<span class='pill' style='background:#fff;border-color:#cbd5e1;color:#334155;margin-left:6px'>{tcounts} <b style='margin-left:4px'>· total {sum(counts.values())}</b></span>"
            f"<span class='pill' style='background:{'#dcfce7' if auto_on else '#f3f4f6'};border-color:{'#86efac' if auto_on else '#e2e8f0'};color:{'#15803d' if auto_on else '#64748b'};margin-left:6px'>{live}</span>")
    body = (f"<div style='background:#0f172a;border-radius:12px;padding:9px 14px'>"
            f"<table class='w'><tr><td><span style='color:#fff;font-weight:800;font-size:13px'>⚡ Breakout Leaders</span></td>"
            f"<td style='text-align:right'>{right}</td></tr></table>"
            f"<div style='font-size:10px;color:#94a3b8;margin-top:6px'>🔍 UNIVERSE {stats.get('universe', 0)} → ⚡SNAP {stats.get('universe', 0) - stats.get('no_data', 0)} · "
            f"scan OK {stats.get('universe', 0) - stats.get('no_data', 0)} → ⚡ SIGNAL {stats.get('no_setup', 0) + stats.get('leaders', 0) + stats.get('late', 0)} "
            f"(≥{spurt_x:g}x {signal_bars}) → ✅ CLEAN {stats.get('leaders', 0)} &nbsp;‖ shaant stocks SNAP me hi skip · baaki illiquid/late · card pe chip dekho"
            f"</div></div>")
    return _wrap(body), 92


# =========================================================================== 1-2. leaders board
def leader_card_html(r: dict, tier_cfg: dict) -> str:
    long = r["side"] == "LONG"
    cls = "buy" if long else "sell"
    sc = r["score"]
    scc = "hi" if sc >= 70 else "mid" if sc >= 50 else "lo"
    ind = r.get("industry") or "NA"
    ind = "Others" if ind in ("NA", "", None) else ind
    mom = {"STRONG": "🔥 STRONG", "MODERATE": "〰 MODERATE", "FADING": "⚠ FADING"}[r["momentum"]]
    meta = f"{esc(ind)} · {mom}" + (" · 🔺 52W zone" if r.get("z52") and long else " · 🔻 52W zone" if r.get("z52") else "")
    segs = "".join(f'<span class="seg{" on" if i < r["conviction"] else ""}"></span>' for i in range(7))
    flags = r.get("conv_flags") or []
    tip = " | ".join(f"{'✔' if f else '✘'} {n}" for n, f in zip(L.CONV_NAMES, flags))
    candle_col = {"Strong": "#15803d", "Moderate": "#b45309", "Weak": "#6b7280"}[r["candle"]]
    stats = (f"gap <b class='{_cls(r['gap_pct'])}'>{_pct(r['gap_pct'], 1)}</b> · body <b>{r['body_pct']:.1f}%</b> · "
             f"<b style='color:{candle_col}'>{r['candle']}</b> candle · low se <b class='up'>▲{r['low_se']:.2f}%</b> · "
             f"high se <b class='dn'>▼{r['high_se']:.2f}%</b>")
    stars = r["stars"]
    bcol = {3: "#16a34a" if long else "#dc2626", 2: "#d97706", 1: "#6b7280"}[stars]
    badge = f"<span class='badge' style='background:{bcol}'>{'★' * stars} {esc(r['watch'])}</span>"
    o = r["opt"]
    lot = f" = ₹{o['lot_value']:,}/lot" if o.get("lot_value") else " <small>(lot size unknown)</small>"
    opt = f"<div class='opt {cls}'>🎯 <b>{esc(o['label'])}</b> @ ≈₹{_f(o['prem'])}{lot}</div>"
    trig = "ORB" if r.get("kind") == "ORB" else "open-candle"
    up = f" · upgraded to {r['tier']} at {r['tier_time']}" if r["tier_time"] != r["detect"] else ""
    tiny = (f"{trig} trigger <b>{'above' if long else 'below'} {_f(r['trigger'])}</b> · SL <b>{_f(r['sl'])}</b> · T1 {_f(r['t1'])} · T2 {_f(r['t2'])}<br>"
            f"{esc(r['zone'])} ({_f(r['zone_lo'])}–{_f(r['zone_hi'])}) · premium ≈₹{_f(o['prem_lo'])}–{_f(o['prem_hi'])}, "
            f"₹{_f(o['prem_sl'])} at SL, ₹{_f(o['prem_t1'])} at T1 (est.)<br>"
            f"🕐 detected {r['detect']} ({r['age_min']:.0f} min ago){up}" + (" · <span class='warn'>⛔ against market mood</span>" if not r.get("mood_ok", True) else ""))
    arrow = f"<span class='ar {'up' if long else 'dn'}'>{'▲' if long else '▼'}</span>"
    return (f"<div class='card {cls}'><table class='w'><tr><td class='sym'>{esc(r['symbol'])}</td><td class='chips'>"
            f"<span class='vol' style='background:{tier_cfg['chip']}'>{r['vol_jump']:.2f}x</span><span class='sc {scc}'>{sc:.0f}</span>"
            f"<span class='clk'>{CLOCK}{esc(r['detect'])}</span></td></tr></table>"
            f"<div class='px'>{_f(r['price'])}{arrow}<span class='chg {_cls(r['day_chg'])}'>{_pct(r['day_chg'])}</span></div>"
            f"<div class='meta'>{meta}</div>"
            f"<div class='conv' title='{esc(tip)}'>CONVICTION &nbsp;{segs}<b class='n'>{r['conviction']}/7</b></div>"
            f"<div class='stats'>{stats}</div>{badge}{opt}<div class='tiny'>{tiny}</div></div>")


def leaders_board_html(board: pd.DataFrame, per_tier: int = 8) -> Tuple[str, int]:
    cols, mx = [], 1
    for tier in ("EXPLOSIVE", "STRONG", "SPURT"):
        t = TIERS[tier]
        sub = board[board.tier == tier] if board is not None and not board.empty else pd.DataFrame()
        cards = "".join(leader_card_html(r, t) for r in sub.head(per_tier).to_dict("records")) if len(sub) else "<div class='empty'>none right now</div>"
        more = f"<div class='empty'>+{len(sub) - per_tier} more - raise 'Cards per tier'</div>" if len(sub) > per_tier else ""
        mx = max(mx, min(len(sub), per_tier))
        cols.append(f"<td class='col'><div class='hd' style='{_grad(t['hd'], t['hd2'])}'><table class='w'><tr><td>{t['icon']} {tier}</td>"
                    f"<td style='text-align:right'><span class='cnt' style='color:{t['hd']}'>{len(sub)}</span></td></tr></table></div>"
                    f"<div class='sub' style='background:{t['sub_bg']};color:{t['sub_fg']}'>{t['sub']}</div><div class='body'>{cards}{more}</div></td>")
    return _wrap(f"<table class='cols'><tr>{''.join(cols)}</tr></table>"), min(140 + mx * 270, 2400)


# =========================================================================== 4. trend ignition
def ignition_card_html(r: dict) -> str:
    long = r["side"] == "LONG"
    cls = "buy" if long else "sell"
    dc = r.get("day_chg")
    arrow = f"<span class='ar {'up' if long else 'dn'}'>{'▲' if long else '▼'}</span>"
    pnl = r["pnl_pct"]
    if r["status"] == "RUNNING":
        state = f"RUNNING · trail SL ₹{_f(r['sl'])} · <b class='{_cls(pnl)}'>{_pct(pnl)}</b> ({r['r_mult']:+.1f}R)"
    else:
        state = f"{esc(r['status'])} at ₹{_f(r['exit'])} ({esc(r['exit_time'])}) · <b class='{_cls(pnl)}'>{_pct(pnl)}</b> ({r['r_mult']:+.1f}R)"
    kind = "opened at day's low" if long else "opened at day's high"
    how = "strong opening candle" if r.get("path") == "candle" else "held the open, then moved"
    return (f"<div class='card {cls}'><table class='w'><tr><td class='sym'>{esc(r['symbol'])}</td><td class='chips'>"
            f"<span class='ign'>⚡</span><span class='clk'>{CLOCK}{esc(r['detect'])}</span></td></tr></table>"
            f"<div class='px'>₹{_f(r['ltp'])}{arrow}<span class='chg {_cls(dc)}'>{_pct(dc)}</span></div>"
            f"<div class='meta'>⚡ TREND IGNITION · {kind}</div>"
            f"<div class='tiny'>{'BUY' if long else 'SELL'} ₹{_f(r['entry'])} · initial SL ₹{_f(r['init_sl'])} · {how}<br>{state}</div></div>")


def ignition_html(df: pd.DataFrame) -> Tuple[str, int]:
    rows = df.to_dict("records") if df is not None and not df.empty else []
    left, right = "".join(ignition_card_html(r) for r in rows[0::2]), "".join(ignition_card_html(r) for r in rows[1::2])
    body = (f"<div style='border:1px solid #e5e7eb;border-radius:12px;overflow:hidden'>"
            f"<div class='hd' style='{_grad(IGN['hd'], IGN['hd2'])}'><table class='w'><tr><td>⚡ {IGN['title']}</td>"
            f"<td style='text-align:right'><span class='cnt' style='color:{IGN['hd']}'>{len(rows)}</span></td></tr></table></div>"
            f"<div class='sub' style='background:{IGN['sub_bg']};color:{IGN['sub_fg']}'>OPENED AT THE DAY'S LOW (OR HIGH) AND NEVER LOOKED BACK · newest first · fresh detections from 09:15</div>"
            f"<table class='cols' style='margin:6px 0 0 0'><tr><td style='vertical-align:top;padding:0 4px 0 8px'>{left}</td>"
            f"<td style='vertical-align:top;padding:0 8px 0 4px'>{right}</td></tr></table></div>")
    return _wrap(body), min(150 + ((len(rows) + 1) // 2) * 135, 2400)


# =========================================================================== 5. pre-open gaps
def _grow(r: dict, trap: bool, show_bs: bool) -> str:
    g = r["gap_pct"]
    arrow = f"<span class='ar2 {_cls(g)}'>{'▲' if g >= 0 else '▼'}</span>" if trap else ""
    bs = f"<span class='bs'>B {int(r['buy_pct'])}%</span>" if show_bs and r.get("buy_pct") == r.get("buy_pct") and r.get("buy_pct") is not None else ""
    sp = "<span class='sp'>SPRUNG</span>" if trap and r.get("trap_state") == "SPRUNG" else ""
    return (f"<tr><td class='s'>{arrow}{esc(r['symbol'])}{bs}{sp}</td><td class='{_cls(g)}' style='font-weight:800'>{_pct(g, 2)}</td>"
            f"<td class='p'>₹{_f(r['ltp'] if r.get('ltp') else r['open'])}</td></tr>")


def gaps_html(gt: pd.DataFrame, gap_min: float, updated: str, session: str, has_bs: bool = False) -> Tuple[str, int]:
    sel = gt[gt.gap_pct.abs() >= gap_min] if gt is not None and not gt.empty else pd.DataFrame(columns=["gap_pct", "trap"])
    up = sel[(sel.gap_pct > 0) & (sel.trap != "LONG TRAP")].sort_values("gap_pct", ascending=False)
    dn = sel[(sel.gap_pct < 0) & (sel.trap != "SHORT TRAP")].sort_values("gap_pct")
    lt = sel[sel.trap == "LONG TRAP"].sort_values("gap_pct", ascending=False)
    st_ = sel[sel.trap == "SHORT TRAP"].sort_values("gap_pct")

    def box(cls, title, d, trap):
        rows = "".join(_grow(r, trap, has_bs) for r in d.head(12).to_dict("records")) or "<tr><td class='mut' style='padding:8px 0'>none</td></tr>"
        return f"<td class='gb {cls}'><div class='gt2'>{title}<span class='n'>{len(d)}</span></div><div class='gl'><table class='w'>{rows}</table></div></td>"

    bs_note = " · B/S chip = share of buy orders in the pre-open book" if has_bs else ""
    body = (f"<div class='gp'><div class='gph'><table class='w'><tr><td>PRE-OPEN GAPS</td><td style='text-align:right'>"
            f"<span class='hide' onclick=\"var b=document.getElementById('gb');var h=b.style.display=='none';b.style.display=h?'block':'none';this.innerHTML=h?'- hide':'+ show'\">- hide</span></td></tr></table></div>"
            f"<div class='gps'>gap = opening price (exchange auction result) vs previous close · frozen at the 09:15 bell{bs_note} · session {esc(session)} · "
            f"updated {esc(updated)} IST · information only, not a recommendation</div>"
            f"<div id='gb'><table class='g2'><tr>{box('gu', '▲ GAP UP', up, False)}{box('gt', '◻ LONG-TRAP', lt, True)}</tr>"
            f"<tr>{box('gd', '▼ GAP DOWN', dn, False)}{box('gt', '◻ SHORT-TRAP', st_, True)}</tr></table></div></div>")
    return _wrap(body), 470


# =========================================================================== 7. replay board
def sparkline(series: List[float], step: int = 1, w: int = 120, h: int = 34) -> str:
    if not series or len(series) < 2:
        return f"<svg width='{w}' height='{h}'></svg>"
    pts = series[::step]
    if pts[-1] != series[-1]:
        pts = pts + [series[-1]]
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    n = len(pts)
    xy = " ".join(f"{(i / max(n - 1, 1)) * (w - 4) + 2:.1f},{h - 3 - ((v - lo) / rng) * (h - 6):.1f}" for i, v in enumerate(pts))
    col = "#16a34a" if pts[-1] >= pts[0] else "#dc2626"
    return f"<svg width='{w}' height='{h}'><polyline points='{xy}' fill='none' stroke='{col}' stroke-width='1.6' stroke-linejoin='round'/></svg>"


def replay_card_html(c: dict, step: int) -> str:
    long = c["side"] == "LONG"
    dc = c.get("day_chg")
    pct = f"<span class='rp {_cls(dc)}'>{_pct(dc)}</span>" if dc is not None else ""
    fresh = " fresh" if c.get("fresh") else ""
    up = f" · up {L.hhmm(c['cur_m'])}" if c.get("upgraded") else ""
    return (f"<table class='rc w{fresh}' style='border-collapse:separate'><tr>"
            f"<td style='vertical-align:top'><div class='rsym'>{esc(c['symbol'])}</div>"
            f"<div style='margin-top:4px'><span class='rv'>{c['vol']:.2f}x</span><span class='rcb {esc(c['chip'])}'>◎ {esc(c['chip'])}</span>{pct}</div>"
            f"<div class='rw'>{'★' * c['stars']} {esc(c['watch'])}</div></td>"
            f"<td style='vertical-align:middle;text-align:center;width:130px'>{sparkline(c['series'], step)}</td>"
            f"<td style='vertical-align:top;width:58px'><div class='rt'>{L.hhmm(c['first_m'])} <span class='ar {'up' if long else 'dn'}' style='font-size:11px'>{'▲' if long else '▼'}</span></div>"
            f"<div class='rt' style='font-size:9px;font-weight:600'>{up.strip(' ·')}</div></td></tr></table>")


def replay_html(cards: List[dict], per_col: int = 8, step: int = 1) -> Tuple[str, int]:
    cols, mx = [], 1
    for tier in ("EXPLOSIVE", "STRONG", "SPURT"):
        t = TIERS[tier]
        sub = [c for c in cards if c["tier"] == tier]
        sub.sort(key=lambda c: (c["first_m"], -c["score"]))                      # in the order they appeared
        mx = max(mx, min(len(sub), per_col))
        inner = "".join(replay_card_html(c, step) for c in sub[:per_col]) or "<div class='empty'>nothing yet</div>"
        more = f"<div class='empty'>+{len(sub) - per_col} more</div>" if len(sub) > per_col else ""
        cols.append(f"<td class='col' style='border:none'><div class='rh'>{t['icon']} {tier} — AS THEY APPEARED</div>{inner}{more}</td>")
    return _wrap(f"<table class='cols'><tr>{''.join(cols)}</tr></table>"), min(90 + mx * 100, 1800)


# =========================================================================== 6. journal dashboard
def _money(v: float, sign: bool = True) -> str:
    s = "+" if v >= 0 and sign else "-" if v < 0 else ""
    return f"{s}₹{abs(v):,.0f}"


def _equity_svg(df: pd.DataFrame, w: int = 520, h: int = 170) -> str:
    daily = df.groupby("date").pnl.sum().cumsum()
    ys = [0.0] + [float(v) for v in daily.values]
    n = len(ys)
    lo, hi = min(0.0, min(ys)), max(ys) * 1.08 if max(ys) > 0 else 1.0
    span = (hi - lo) or 1.0
    L_, R_, T_, B_ = 40, 10, 8, 18
    X = lambda i: L_ + (i / max(n - 1, 1)) * (w - L_ - R_)
    Y = lambda v: T_ + (1 - (v - lo) / span) * (h - T_ - B_)
    step = 5000 if span > 12000 else 2000 if span > 5000 else 1000 if span > 2000 else 500
    grid = "".join(f"<line x1='{L_}' y1='{Y(t):.1f}' x2='{w - R_}' y2='{Y(t):.1f}' stroke='#eef2f7'/><text x='{L_ - 5}' y='{Y(t) + 3:.1f}' font-size='9' fill='#94a3b8' text-anchor='end'>{t / 1000:.1f}k</text>"
                   for t in np.arange(0, hi, step)) + f"<line x1='{L_}' y1='{Y(0):.1f}' x2='{w - R_}' y2='{Y(0):.1f}' stroke='#cbd5e1'/>"
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
    area = f"{X(0):.1f},{Y(0):.1f} {pts} {X(n - 1):.1f},{Y(0):.1f}"
    dots = "".join(f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='2.6' fill='#16a34a'/>" for i, v in enumerate(ys) if i > 0 and (i % max(n // 6, 1) == 0 or i == n - 1))
    return (f"<svg width='{w}' height='{h}'>{grid}<polygon points='{area}' fill='#16a34a' fill-opacity='.14'/>"
            f"<polyline points='{pts}' fill='none' stroke='#16a34a' stroke-width='2'/>{dots}</svg>")


def _calendar(df: pd.DataFrame, year: int, month: int, today: date) -> str:
    d = df[(df.date.dt.year == year) & (df.date.dt.month == month)].groupby(df.date.dt.day).pnl.sum() if not df.empty else pd.Series(dtype=float)
    daily = {int(k): float(v) for k, v in d.items()}
    rows = "<tr>" + "".join(f"<th>{x}</th>" for x in "MTWTFSS") + "</tr>"
    for wk in calendar.Calendar(0).monthdayscalendar(year, month):
        cells = []
        for i, dd in enumerate(wk):
            if dd == 0:
                cells.append("<td></td>")
                continue
            c = "p" if daily.get(dd, 0) > 0 else "l" if daily.get(dd, 0) < 0 else ("we" if i >= 5 else "")
            if date(year, month, dd) == today:
                c += " today"
            tip = f" title='{_money(daily[dd])}'" if dd in daily else ""
            cells.append(f"<td><span class='d {c}'{tip}>{dd}</span></td>")
        rows += "<tr>" + "".join(cells) + "</tr>"
    return (f"<div class='t' style='text-align:center'>{calendar.month_name[month]} {year}</div>"
            f"<div style='text-align:center;font-size:9px;color:#94a3b8;margin-bottom:4px'>green = profit day · red = loss day</div>"
            f"<table class='cal w'>{rows}</table><div style='font-size:9px;color:#94a3b8;margin-top:4px'>gold ring = today</div>")


SETUP_COL = {"Leader": ("#fff7ed", "#c2410c", "#fdba74"), "Trend": ("#eef2ff", "#4338ca", "#c7d2fe"), "Gap": ("#fdf4ff", "#a21caf", "#f0abfc"),
             "ORB": ("#ecfeff", "#0e7490", "#a5f3fc"), "VWAP": ("#f0fdf4", "#166534", "#bbf7d0"), "Own": ("#f3f4f6", "#4b5563", "#e5e7eb")}


def _setup_chip(setup: str) -> str:
    lab = J.SHORT_SETUP.get(setup, setup)
    key = next((k for k in SETUP_COL if lab.startswith(k)), "Own")
    bg, fg, bd = SETUP_COL[key]
    return f"<span class='st' style='background:{bg};color:{fg};border-color:{bd}'>{esc(lab)}</span>"


def journal_html(full: pd.DataFrame, rows: pd.DataFrame, S: dict, cards: dict, year: int, month: int, today: date,
                 limit: int = 60) -> Tuple[str, int]:
    span = f"{full.date.min():%m-%d} → {full.date.max():%m-%d}"
    kp = [("NET P&L", _money(S["net"]), "#15803d" if S["net"] >= 0 else "#b91c1c", f"{S['trades']} trades"),
          ("WIN RATE", f"{S['win_rate']:.0f}%", "#0f172a", f"{S['wins']}W · {S['losses']}L"),
          ("PROFIT FACTOR", "∞" if S["profit_factor"] == float("inf") else f"{S['profit_factor']:.1f}", "#0f172a", "gross W ÷ L"),
          ("AVG WIN", _money(S["avg_win"], False), "#15803d", "per winning trade"),
          ("AVG LOSS", _money(S["avg_loss"]), "#b91c1c", "per losing trade"),
          ("WIN STREAK", str(S["win_streak"]), "#0f172a", "best run"),
          ("DISCIPLINE", f"{S['discipline']:.0f}%", "#0f766e", "rules followed")]
    kpis = "<table class='kp'><tr>" + "".join(f"<td class='k'><div class='l'>{a}</div><div class='v' style='color:{c}'>{b}</div><div class='s'>{s}</div></td>" for a, b, c, s in kp) + "</tr></table>"
    mid = (f"<table class='w' style='border-collapse:separate;border-spacing:0 0;margin-top:10px'><tr>"
           f"<td class='pn' style='width:64%'><div class='t'>Equity Curve<small>{span} · net {_money(S['net'])}</small></div>{_equity_svg(full)}"
           f"<div style='font-size:9.5px;color:#94a3b8'>cumulative P&amp;L per trading day — the line your discipline draws</div></td>"
           f"<td style='width:2%'></td><td class='pn' style='width:34%'>{_calendar(full, year, month, today)}</td></tr></table>")
    ins = ("<table class='w' style='border-collapse:separate;border-spacing:8px 0;margin:10px -4px 0 -4px;table-layout:fixed'><tr>"
           f"<td class='ins' style='border-left:4px solid #ea580c'><div class='l' style='color:#ea580c'>BEST SETUP</div>{esc(cards.get('best', ''))}</td>"
           f"<td class='ins' style='border-left:4px solid #16a34a'><div class='l' style='color:#16a34a'>AVG HOLD</div>{esc(cards.get('hold', ''))}</td>"
           f"<td class='ins' style='border-left:4px solid #dc2626'><div class='l' style='color:#dc2626'>WATCH OUT</div>{esc(cards.get('watch', ''))}</td></tr></table>")
    trs = ""
    for i, r in enumerate(rows.sort_values(["date", "entry_time"], ascending=False).head(limit).to_dict("records")):
        long = r["side"] == "LONG"
        _sk = r.get("strike")
        strike = "" if (_sk is None or (isinstance(_sk, float) and np.isnan(_sk))) else str(_sk).replace(" ", "")
        oc = f"<span class='oc'>{esc(strike)}</span>" if strike else ""
        mis = r.get("mistake") or "None"
        d_ok = mis == "None"
        note = esc(r.get("note") or "") or "no note"
        trs += (f"<tr class='row' onclick=\"var n=document.getElementById('n{i}');n.style.display=(n.style.display=='table-row')?'none':'table-row'\">"
                f"<td>{pd.Timestamp(r['date']):%d-%b}</td><td><b>{esc(r['symbol'])}</b>{oc}</td>"
                f"<td><span class='sd {'BUY' if long else 'SELL'}'>{'BUY' if long else 'SELL'}</span></td>"
                f"<td class='mut'>{_f(r['entry'], 1)} / {_f(r.get('sl') or 0, 1)} / {_f(r.get('target') or 0, 1)} / {_f(r['exit'], 1)}</td>"
                f"<td>{int(r['qty']):,}</td><td class='{_cls(r['pts'])}' style='font-weight:700'>{r['pts']:+.1f}</td>"
                f"<td class='{_cls(r['pnl'])}' style='font-weight:800'>{_money(r['pnl'])}</td>"
                f"<td class='{_cls(r['r'] or 0)}' style='font-weight:700'>{('%+.1fR' % r['r']) if r.get('r') is not None else '-'}</td>"
                f"<td>{_setup_chip(r['setup'])}</td><td><span class='dc {'ok' if d_ok else 'bad'}'>{esc(J.SHORT_MISTAKE.get(mis, mis))}</span></td></tr>"
                f"<tr class='note' id='n{i}'><td colspan='10'>📝 {note}</td></tr>")
    table = (f"<table class='tt'><tr><th>Date</th><th>Symbol</th><th>Side</th><th>E / SL / T1 / Exit</th><th>Qty</th><th>Pts</th>"
             f"<th>P&amp;L ₹</th><th>R</th><th>Setup</th><th>Disc.</th></tr>{trs}</table>"
             f"<div style='font-size:9.5px;color:#94a3b8;margin-top:4px'>{len(rows)} trades · click any row for its note &amp; lesson</div>")
    head = (f"<table class='w'><tr><td><span style='font-size:18px;font-weight:800'>Trading Journal</span> "
            f"<span class='pill' style='margin-left:8px'>PRIVATE — ONLY YOUR KEY</span></td></tr></table><div style='height:8px'></div>")
    foot = "<div style='text-align:center;font-size:9.5px;color:#94a3b8;margin-top:10px'>Tied only to your access key — private per key · export anytime</div>"
    return _wrap(head + kpis + mid + ins + table + foot), 640 + min(len(rows), limit) * 37
