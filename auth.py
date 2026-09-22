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
import os
import time
from typing import Optional

import streamlit as st

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 30


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


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
    return hash_password(pw) if pw else None


def logged_in() -> bool:
    return bool(st.session_state.get("_authed"))


def require_login() -> None:
    """Call once, immediately after st.set_page_config, before anything else renders. Halts the script with
    st.stop() until the correct password is entered; does nothing if no password is configured."""
    real_hash = _configured_hash()
    if real_hash is None:
        with st.sidebar:
            st.warning("⚠️ No password set - this dashboard is open to anyone with the link. Set APP_PASSWORD "
                      "(or APP_PASSWORD_HASH) before exposing it beyond your own machine.", icon="⚠️")
        return
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
            if hash_password(pw) == real_hash:
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
