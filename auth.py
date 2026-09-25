"""
auth.py - password gate for the whole dashboard, with in-app set/change/remove, on top of an optional
host-level override for whoever is deploying/running the app.

Priority, checked in this order:
  1. Host override: APP_PASSWORD_HASH or APP_PASSWORD, as an environment variable or the same keys in
     .streamlit/secrets.toml. This wins over anything set from inside the app, and reports itself via
     host_locked() so the sidebar hides the change-password controls and set_password()/change_password()
     refuse to run - there's no in-app facility to change an environment variable on the host.
  2. In-app password: set from the sidebar, stored as a salted hash in this app's own SQLite (store.py),
     under its own meta key - unrelated to the Journal's per-access-key privacy in journal.py, which keeps
     journals apart from each other and works regardless of whether this gate is on.
  3. Nothing configured: the app runs open (fine for your own machine), and the first screen offers to set a
     password now or skip and stay open. Skipping only affects the current session; the same choice is offered
     again next time unless a password gets set.

This is basic protection against casual/drive-by access, not enterprise auth: one shared password, no per-user
accounts, no password-reset flow. Anyone with shell access to the host can still read an env-var override, or the
stored hash in trades.db (a hash, not the password itself).
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import streamlit as st

import store

_SALT = "orb-command-center::"          # fixed pepper - keeps a leaked hash from being a bare, rainbow-table-able sha256
_META_KEY = "auth_pw_hash"
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 30


def hash_password(pw: str) -> str:
    return hashlib.sha256((_SALT + pw).encode("utf-8")).hexdigest()


def _secret(key: str) -> Optional[str]:
    try:
        v = st.secrets.get(key)                              # st.secrets raises if no secrets.toml exists at all
        return str(v) if v else None
    except Exception:
        return None


def host_hash() -> Optional[str]:
    """The hash implied by an env var / secrets override, or None if the host hasn't set one."""
    h = os.environ.get("APP_PASSWORD_HASH") or _secret("APP_PASSWORD_HASH")
    if h:
        return h.strip().lower()
    pw = os.environ.get("APP_PASSWORD") or _secret("APP_PASSWORD")
    return hash_password(pw) if pw else None


def host_locked() -> bool:
    return host_hash() is not None


def stored_hash() -> Optional[str]:
    return store.get_meta(_META_KEY)


def configured_hash() -> Optional[str]:
    return host_hash() or stored_hash()


def is_set() -> bool:
    return configured_hash() is not None


def set_password(pw: str) -> None:
    if host_locked():
        raise RuntimeError("Password is set by the host (APP_PASSWORD / APP_PASSWORD_HASH) and can't be changed from inside the app.")
    store.set_meta(_META_KEY, hash_password(pw))


def change_password(old_pw: str, new_pw: str) -> bool:
    if host_locked():
        raise RuntimeError("Password is set by the host (APP_PASSWORD / APP_PASSWORD_HASH) and can't be changed from inside the app.")
    if hash_password(old_pw) != configured_hash():
        return False
    store.set_meta(_META_KEY, hash_password(new_pw))
    return True


def remove_password() -> None:
    """Clears the in-app password. A host override, if any, still applies regardless."""
    store.delete_meta(_META_KEY)


def logged_in() -> bool:
    return bool(st.session_state.get("_authed"))


# --------------------------------------------------------------------------- screens
def _lockout_remaining() -> float:
    return st.session_state.get("_pw_locked_until", 0.0) - time.time()


def _register_attempt(ok: bool) -> None:
    if ok:
        st.session_state._pw_attempts = 0
        return
    n = st.session_state.get("_pw_attempts", 0) + 1
    st.session_state._pw_attempts = n
    if n >= MAX_ATTEMPTS:
        st.session_state._pw_locked_until = time.time() + LOCKOUT_SECONDS
        st.session_state._pw_attempts = 0


def _shell(body) -> None:
    st.markdown("<div style='max-width:360px;margin:10vh auto 0 auto;text-align:center'>"
               "<div style='font-size:34px'>🔒</div><h3 style='margin:6px 0 18px 0'>ORB Command Center</h3></div>",
               unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        body()
    st.stop()


def _login_screen() -> None:
    def body():
        remaining = _lockout_remaining()
        if remaining > 0:
            st.error(f"Too many wrong attempts. Try again in {int(remaining) + 1}s.")
            return
        with st.form("orb_login", clear_on_submit=True):
            pw = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            ok = st.form_submit_button("Unlock", type="primary", use_container_width=True)
        if ok:
            correct = hash_password(pw) == configured_hash()
            _register_attempt(correct)
            if correct:
                st.session_state._authed = True
                st.rerun()
            else:
                left = MAX_ATTEMPTS - st.session_state.get("_pw_attempts", 0)
                st.error("Wrong password." if st.session_state.get("_pw_locked_until", 0) > time.time()
                        else f"Wrong password. {left} attempt(s) left before a short lockout.")
    _shell(body)


def _first_run_screen() -> None:
    def body():
        st.caption("No password set yet. Set one now, or skip and use the dashboard without one - fine for your "
                  "own machine, but anyone with the link can open it while skipped.")
        with st.form("orb_setup", clear_on_submit=False):
            pw1 = st.text_input("New password", type="password")
            pw2 = st.text_input("Confirm password", type="password")
            c1, c2 = st.columns(2)
            set_click = c1.form_submit_button("Set password", type="primary", use_container_width=True)
            skip_click = c2.form_submit_button("Skip - stay open", use_container_width=True)
        if set_click:
            if not pw1:
                st.error("Enter a password.")
            elif pw1 != pw2:
                st.error("Passwords don't match.")
            else:
                set_password(pw1)
                st.session_state._authed = True
                st.rerun()
        if skip_click:
            st.session_state._authed = True
            st.rerun()
    _shell(body)


def require_login() -> None:
    """Call once, right after store.init(), before anything else renders. Halts the script with st.stop() until
    unlocked; does nothing once the session is authed."""
    if logged_in():
        return
    if is_set():
        _login_screen()
    else:
        _first_run_screen()


def sidebar_controls() -> None:
    """Sidebar block: lock button, plus set/change/remove for the in-app password. Hidden (mostly) when the host
    has locked it via an env var, since there's nothing to manage from inside the app in that case."""
    if host_locked():
        st.caption("🔒 Password set by host (env var) - can't be changed here.")
        if logged_in() and st.button("Lock", key="_pw_lock", use_container_width=True):
            st.session_state._authed = False
            st.rerun()
        return

    if is_set():
        if st.button("🔒 Lock now", key="_pw_lock", use_container_width=True):
            st.session_state._authed = False
            st.rerun()
        with st.expander("Change password"):
            with st.form("orb_change_pw", clear_on_submit=True):
                old = st.text_input("Current password", type="password", key="_pw_old")
                new1 = st.text_input("New password", type="password", key="_pw_new1")
                new2 = st.text_input("Confirm new password", type="password", key="_pw_new2")
                go = st.form_submit_button("Update password", use_container_width=True)
            if go:
                if not new1:
                    st.error("Enter a new password.")
                elif new1 != new2:
                    st.error("New passwords don't match.")
                elif change_password(old, new1):
                    st.success("Password updated.")
                else:
                    st.error("Current password is wrong.")
            if st.button("Remove password (go back to open access)", key="_pw_remove", use_container_width=True):
                remove_password()
                st.rerun()
    else:
        with st.expander("🔓 Set a password"):
            with st.form("orb_set_pw", clear_on_submit=True):
                new1 = st.text_input("New password", type="password", key="_pw_set1")
                new2 = st.text_input("Confirm password", type="password", key="_pw_set2")
                go = st.form_submit_button("Set password", use_container_width=True)
            if go:
                if not new1:
                    st.error("Enter a password.")
                elif new1 != new2:
                    st.error("Passwords don't match.")
                else:
                    set_password(new1)
                    st.success("Password set. It'll be required next time the app is opened.")
