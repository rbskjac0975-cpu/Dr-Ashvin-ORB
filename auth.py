"""
auth.py - a simple password gate for the whole dashboard (separate from the Journal's per-access-key privacy,
which protects individual journals from each other, not the app itself).

Set ONE of these before running `streamlit run app.py`:
  * environment variable APP_PASSWORD_HASH = the sha256 hex digest of your password (recommended - see below)
  * environment variable APP_PASSWORD = the password itself, in plain text (simpler, less safe on shared machines)
  * st.secrets["APP_PASSWORD_HASH"] or st.secrets["APP_PASSWORD"] in .streamlit/secrets.toml, same rules

To get a hash without typing your password in plain text anywhere:
    python -c "import auth; print(auth.hash_password(input('Password: ')))"

If neither is set, the app runs open (unprotected) and shows a small warning in the sidebar - this is meant for
local/private use only. Do not expose an unprotected instance on the open internet.

This is basic protection (keeps casual/unauthenticated access out), not enterprise auth: one shared password, no
per-user accounts, no password reset flow. A wrong-password lockout slows down guessing but the password itself
lives in an environment variable or secrets.toml on the host, so anyone with access to the host can read it there.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import time
from typing import Optional

import streamlit as st

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 30
DB = os.environ.get("TRADE_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db"))
_ITERATIONS = 310_000


def hash_password(pw: str) -> str:
    """Return a salted, deliberately slow password hash."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify(pw: str, saved: str) -> bool:
    if saved.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt, expected = saved.split("$")
            got = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt), int(rounds)).hex()
            return hmac.compare_digest(got, expected)
        except (ValueError, TypeError):
            return False
    # Compatibility with the documented legacy SHA-256 hash setting.
    return hmac.compare_digest(hashlib.sha256(pw.encode("utf-8")).hexdigest(), saved)


def _stored_hash() -> Optional[str]:
    try:
        with sqlite3.connect(DB, timeout=5) as c:
            row = c.execute("SELECT v FROM meta WHERE k='app_password_hash'").fetchone()
            return row[0] if row else None
    except sqlite3.Error:
        return None


def _save_hash(value: str) -> None:
    with sqlite3.connect(DB, timeout=5) as c:
        c.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        c.execute("INSERT INTO meta(k,v) VALUES('app_password_hash',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (value,))
        c.commit()


def _secret(key: str) -> Optional[str]:
    try:
        v = st.secrets.get(key)                              # st.secrets raises if no secrets.toml exists at all
        return str(v) if v else None
    except Exception:
        return None


def _configured_hash() -> Optional[str]:
    h = os.environ.get("APP_PASSWORD_HASH") or _secret("APP_PASSWORD_HASH")
    if h:
        return h.strip().lower()
    pw = os.environ.get("APP_PASSWORD") or _secret("APP_PASSWORD")
    return hashlib.sha256(pw.encode("utf-8")).hexdigest() if pw else _stored_hash()


def logged_in() -> bool:
    return bool(st.session_state.get("_authed"))


def require_login() -> None:
    """Call once, immediately after st.set_page_config, before anything else renders. Halts the script with
    st.stop() until the correct password is entered; does nothing if no password is configured."""
    real_hash = _configured_hash()
    if real_hash is None:
        st.markdown("<div style='max-width:420px;margin:12vh auto 0;text-align:center'><div style='font-size:34px'>🔒</div><h3>Set a dashboard password</h3><p>Choose a password to protect this app. It is saved as a salted hash in the app database.</p></div>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1.3, 1])
        with mid:
            with st.form("orb_first_password", clear_on_submit=True):
                pw = st.text_input("New password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                ok = st.form_submit_button("Set password", type="primary", use_container_width=True)
            if ok:
                if len(pw) < 8:
                    st.error("Use at least 8 characters.")
                elif pw != confirm:
                    st.error("Passwords do not match.")
                else:
                    _save_hash(hash_password(pw))
                    st.session_state._authed = True
                    st.rerun()
        st.stop()
    if logged_in():
        return

    locked_until = st.session_state.get("_pw_locked_until", 0.0)
    remaining = locked_until - time.time()
    st.markdown("<div style='max-width:360px;margin:12vh auto 0 auto;text-align:center'>"
               "<div style='font-size:34px'>🔒</div><h3 style='margin:6px 0 18px 0'>ORB Command Center</h3></div>",
               unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        if remaining > 0:
            st.error(f"Too many wrong attempts. Try again in {int(remaining) + 1}s.")
            st.stop()
        with st.form("orb_login", clear_on_submit=True):
            pw = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            ok = st.form_submit_button("Unlock", type="primary", use_container_width=True)
        if ok:
            if _verify(pw, real_hash):
                st.session_state._authed = True
                st.session_state._pw_attempts = 0
                st.rerun()
            else:
                n = st.session_state.get("_pw_attempts", 0) + 1
                st.session_state._pw_attempts = n
                if n >= MAX_ATTEMPTS:
                    st.session_state._pw_locked_until = time.time() + LOCKOUT_SECONDS
                    st.session_state._pw_attempts = 0
                    st.error(f"Too many wrong attempts. Locked for {LOCKOUT_SECONDS}s.")
                else:
                    st.error(f"Wrong password. {MAX_ATTEMPTS - n} attempt(s) left before a short lockout.")
    st.stop()


def logout_button() -> None:
    """Sidebar control shown only when a password is configured and the session is unlocked."""
    if _configured_hash() is not None and logged_in():
        if st.sidebar.button("🔒 Lock", key="_pw_logout", use_container_width=True):
            st.session_state._authed = False
            st.rerun()
        # Host-provided credentials cannot be changed from the UI; use the local password store otherwise.
        if not (os.environ.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD_HASH") or _secret("APP_PASSWORD") or _secret("APP_PASSWORD_HASH")):
            with st.sidebar.expander("Change password"):
                with st.form("_pw_change", clear_on_submit=True):
                    old = st.text_input("Current password", type="password")
                    new = st.text_input("New password", type="password")
                    confirm = st.text_input("Confirm new password", type="password")
                    save = st.form_submit_button("Save password", use_container_width=True)
                if save:
                    current = _stored_hash() or ""
                    if not _verify(old, current):
                        st.error("Current password is incorrect.")
                    elif len(new) < 8:
                        st.error("Use at least 8 characters.")
                    elif new != confirm:
                        st.error("Passwords do not match.")
                    else:
                        _save_hash(hash_password(new))
                        st.success("Password changed.")
