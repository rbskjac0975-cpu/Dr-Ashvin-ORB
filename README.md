# ORB Command Center

Watchlist → breadth → ORB + 1-minute confirmation → levels/risk → live trade management, **plus** an F&O leaders board,
option-strike ideas, trend ignition, opening gaps, a day-replay and a private trading journal.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py                                            # dashboard
python monitor.py --scan "Nifty 200" --scan-every 180 --leaders  # backend (separate terminal)
```

## Password-protect the dashboard
The first time you open the app with no password configured, it shows a screen to **set a password right there**
(or **skip and stay open** - fine for your own machine; anyone with the link can open it while skipped, and the
same screen appears again next time). Once set, unlocking just needs the password - nothing to configure on the
command line. From the sidebar you can **change** or **remove** the password at any time, or hit **Lock** to
re-lock the session immediately.
- Wrong attempts lock out for 30s after 5 tries.
- The password is stored as a salted hash in this app's own SQLite (`trades.db`), never in plain text.
- If you're deploying this for others and want to fix the password outside the app (so it can't be changed from
  the UI), set **one** of these as an environment variable (or the same key in `.streamlit/secrets.toml`) before
  running `streamlit run app.py`; this always overrides whatever is set in-app:
  ```bash
  export APP_PASSWORD_HASH=$(python -c "import auth; print(auth.hash_password(input('Password: ')))")
  # or, simpler but less safe on a shared machine:
  export APP_PASSWORD=your-password
  ```
This is basic protection against casual/drive-by access, not enterprise auth - one shared password, no per-user
accounts, no password-reset flow, and whoever has shell access to the host can still read the stored hash or an
env-var override. The Journal's per-access-key privacy (see below) is a separate, unrelated mechanism that keeps
journals apart from each other and works regardless of whether this password gate is on.

## Use it like an app on your phone (Add to Home Screen)
The dashboard adds a manifest, icons and theme-color tags to the page automatically (`pwa.py`) so that once it's
hosted somewhere with HTTPS (Streamlit Community Cloud, a VPS, Render, etc.), opening it in Chrome (Android) or
Safari (iOS) and choosing **Add to Home Screen** gives it a real icon and a full-screen, address-bar-free window -
no app-store build needed. This does **not** produce an installable `.apk`; it needs a live server somewhere (a
phone can't run the Python/Streamlit backend itself), and it hasn't been verified against a live install prompt on
a real device in this environment - confirm the "Add to Home Screen" look after you deploy.
Optional Telegram alerts: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (or paste in the sidebar).
Click **Save settings for backend monitor** in the sidebar so `monitor.py` uses the same settings.
For real F&O data drop NSE's `fo_mktlots.csv` into `./cache/` (it is also downloaded automatically when NSE is reachable).

## The seven tabs

| # | Tab | What it does |
|---|-----|--------------|
| 1 | 💥 **Breakout Leaders** | Whole F&O universe → 3-layer filter → cards in three tiers: 💥 EXPLOSIVE → 🔥 STRONG → ⚡ SPURT, strongest on top. Each card: strength score, rank, volume jump, exact detect time, live zone. |
| 2 | 🎯 **Option strike** (on every card) | Strike (ATM / ITM-1 / OTM-1), premium zone, premium at the stop-loss and at T1, delta, expiry, lot value, ORB trigger level, SL area. |
| 3 | 📊 **Market mood + alerts** | Advance/decline strip on the leaders tab. Telegram + toast + beep when a **new, fresh EXPLOSIVE** leader appears. Auto-scan starts 09:16 (dashboard toggle or `monitor.py --leaders`). |
| 4 | ⚡ **Trend Ignition** | Candle that opens *at* the day's low (or high) on a strong, high-volume body inside the first N minutes. Entry printed from that candle; a structure trail is simulated so you see the exit. Newest on top. |
| 5 | 🌅 **Pre-Open Gaps** | Gap-up / gap-down lists (gap frozen at the 09:15 open, live % moving), plus LONG TRAP / SHORT TRAP flags and whether the trap has since "sprung". Snapshot saved once per day. |
| 6 | 📓 **Journal** | One-tap logging (entry, SL, target, exit, setup, note, mistake tag). Option strike field + **auto lot size** (lots × lot size). Equity curve, month calendar, auto insights (best setup, average hold time, costliest mistake), CSV export. Private per access key. |
| 7 | ▶ **Day Replay** | Every leader re-appears at the minute it fired, with the badge and mini chart it had *then*. Play / pause / scrub / speed. Build a pack for any of the last ~5 sessions, or let `monitor.py --leaders` save today's automatically after the close. |

The original five tabs (Watchlist, Market breadth, Strong-buy scanner, Live desk, Trade manager) are unchanged.

## Look and feel
The Breakout Leaders board, Trend Ignition cards, Pre-Open Gaps panel, Day Replay board and Journal dashboard are
rendered as self-contained HTML/CSS (`ui.py`) via `streamlit.components.v1.html(...)`, not native Streamlit widgets -
this is what gives the gradient tier headers, conviction bars, ★ watch badges, sparkline replay cards and the 2×2
gaps panel their look. `app.py` still uses native Streamlit controls (buttons, toggles, sliders) around each panel
for the interactive parts (scan now, live auto-scan, speed, etc).

## How the rules work

**Leaders - 3 layers**
1. *Junk / illiquid*: price ≥ min, avg turnover ≥ min, trades in ≥80% of minutes, not frozen (circuit).
2. *Setup quality*: ORB formed and tradable, a 1-minute **close** beyond the ORB, strong body, on the right side of VWAP, EMA9/21 aligned, not over-extended, and price still beyond the ORB now.
3. *Late-entry*: dropped if price is already `ld_late_ext` ORB-widths past the breakout, or the leader was detected more than `ld_max_age` minutes ago.

**Tiers** use the signal bar's volume jump (vs its 20-bar average) and a 0-100 strength score (volume jump 30, candle body 10, decisiveness of the break 15, VWAP distance 10, cumulative RVOL 15, daily setup agreement 15, EMA separation 5). Defaults: EXPLOSIVE ≥4× and ≥78; STRONG ≥2.5× and ≥60; SPURT ≥1.5×. A leader can be upgraded later in the day. Breadth "against mood" leaders are tagged, not hidden, and never alerted.

**Trend ignition**: within the first `ig_window` minutes, a candle whose open is within `ig_open_tol`% of its low (long) / high (short), that low is the session low so far, body ≥ `ig_body` of range **and** ≥ `ig_min_move`% of price, volume ≥ `ig_vol`× an average minute. Entry = that candle's close; stop just beyond it; trail = last N bars' extreme.

**Gap traps** (heuristic, not a prediction): SHORT TRAP = gap-down into an uptrend (close > EMA20 > EMA50) or an oversold stock (RSI ≤ 35); LONG TRAP = gap-up into a downtrend or an overbought stock (RSI ≥ 70). SPRUNG = price is ≥0.5% from the open against the gap.

**No hindsight**: every signal is computed bar by bar from data available at that minute, forming candles are ignored, and the replay cuts daily context at the day before. A test proves a signal computed on partial data equals the one computed on the full day.

## Read this before trading (limitations)
- **Option premiums are estimates** (Black-Scholes on 20-day realised volatility as an IV stand-in). yfinance has no NSE option chain. Strike interval is an approximate price-band table and the monthly expiry weekday is a setting (default Tuesday) - **verify both with NSE / your broker**. Override strike steps and lot sizes per symbol in `cache/fno_lots.csv` (`symbol,lot,strike_step`).
- **Lot sizes are never hard-coded** (they change). They come from NSE's `fo_mktlots.csv` or your CSV. If neither exists the built-in F&O list is approximate, lots are "unknown", and the journal asks you to type the lot size.
- **Pre-open**: yfinance has no 09:00-09:08 indicative prices. The gap is the exchange's opening price at 09:15 (the auction result), frozen at the bell. An optional NSE pre-open fetch is included but **untested against live NSE** and often blocked.
- **"Synced on phone and PC"**: the journal is stored in this app's SQLite file under a hash of your access key. Entries are private per key, but syncing across devices only works if both devices open the *same running server* (e.g. the app hosted on a VPS). There is no account system or password recovery.
- Data is yfinance (unofficial, can lag or drop symbols). The first F&O scan downloads ~150-200 symbols of 1-minute data and takes a while; results are cached 20 s.
- Ignition exits and replay results are simulated on 1-minute bars; real fills differ. Inside one bar the stop is assumed to hit before the target.
- Paper-trade for a few weeks first. This is decision-support, not investment advice.

## Tests
```bash
python tests/test_logic.py   # 70+ checks on synthetic data: tiers, filters, no-lookahead, ignition, gaps, replay, options, journal
python tests/smoke_app.py    # executes app.py end-to-end with a fake Streamlit, incl. journal HTML regression checks
```
These use synthetic data only; nothing here has been run against live yfinance / NSE. The HTML panels in `ui.py`
were additionally checked by rendering them to PNG with `wkhtmltoimage` against synthetic data shaped like the
reference screenshots (gap-up/down/trap stocks, opening-surge and delayed-ignition stocks) and comparing side by
side. One rendering quirk found only in that headless tool: a few emoji (💥 🔥 🎯) fall back to generic glyphs
because its bundled font lacks full color-emoji coverage - this does not affect real browsers (Chrome/Firefox/Edge),
which is what Streamlit serves to users.
