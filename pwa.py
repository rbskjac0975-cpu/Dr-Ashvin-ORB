"""
pwa.py - best-effort "Add to Home Screen" support.

Streamlit doesn't let a page edit its own <head> directly, so this injects a manifest link, icons and
theme-color meta tags into the OUTER page's <head> from inside a components.html iframe via
`window.parent.document` (the iframe is same-origin with the Streamlit page, so this is allowed by the browser).
The manifest and icons are embedded as data: URIs so nothing needs to be hosted as a separate static file.

NOT independently verified against a live Chrome-on-Android install prompt in this environment (no browser or
device here to test against) - this is the standard community technique for adding PWA tags to a Streamlit app,
but please confirm the "Add to Home Screen" / install-banner behaviour after deploying. Even if Chrome doesn't
offer a full "install app" banner (it can be picky about criteria such as HTTPS, a service worker, or repeat
visits), the manifest + theme-color + apple-touch-icon here still make a plain "Add to Home Screen" shortcut look
and feel like a real app (proper icon, standalone window, no browser address bar) on both Android and iOS Safari.
"""
from __future__ import annotations

import base64
import io
import json

import streamlit.components.v1 as components

_ICON_CACHE: dict = {}


def _make_icon_b64(size: int, bg: str = "#0f172a", fg: str = "#22c55e") -> str:
    """A small candlestick + breakout-arrow glyph, drawn in-process (no external asset, nothing copyrighted)."""
    if size in _ICON_CACHE:
        return _ICON_CACHE[size]
    from PIL import Image, ImageDraw
    w = size
    im = Image.new("RGB", (w, w), bg)
    d = ImageDraw.Draw(im)
    cx = w * 0.36
    lw = max(2, w // 32)
    d.line([(cx, w * 0.18), (cx, w * 0.40)], fill=fg, width=lw)
    d.rectangle([cx - w * 0.07, w * 0.40, cx + w * 0.07, w * 0.66], fill=fg)
    d.line([(cx, w * 0.66), (cx, w * 0.82)], fill=fg, width=lw)
    ax = w * 0.62
    d.line([(ax - w * 0.16, w * 0.62), (ax + w * 0.16, w * 0.26)], fill=fg, width=max(3, w // 24))
    ah = w * 0.09
    d.polygon([(ax + w * 0.16, w * 0.26), (ax + w * 0.16 - ah, w * 0.26 + ah * 0.4),
              (ax + w * 0.16 - ah * 0.4, w * 0.26 + ah)], fill=fg)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    _ICON_CACHE[size] = b64
    return b64


def inject(app_name: str = "ORB Command Center", short_name: str = "ORB", theme_color: str = "#0f172a",
          bg_color: str = "#ffffff") -> None:
    """Call once near the top of the script, after st.set_page_config. Safe to call on every rerun - the JS
    checks for existing tags first so it doesn't duplicate them."""
    try:
        icon192, icon512 = _make_icon_b64(192, theme_color), _make_icon_b64(512, theme_color)
    except Exception:
        return                                          # e.g. Pillow unavailable - fail quietly, app still works
    manifest = {
        "name": app_name, "short_name": short_name, "start_url": ".", "scope": ".", "display": "standalone",
        "background_color": bg_color, "theme_color": theme_color,
        "icons": [{"src": f"data:image/png;base64,{icon192}", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                 {"src": f"data:image/png;base64,{icon512}", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}],
    }
    manifest_b64 = base64.b64encode(json.dumps(manifest).encode()).decode()
    js = f"""<script>
(function() {{
  try {{
    var d = window.parent.document;
    function ensure(sel, tag, attrs) {{
      if (d.querySelector(sel)) return;
      var el = d.createElement(tag);
      for (var k in attrs) el.setAttribute(k, attrs[k]);
      d.head.appendChild(el);
    }}
    ensure('link[rel="manifest"]', 'link', {{rel: 'manifest', href: 'data:application/manifest+json;base64,{manifest_b64}'}});
    ensure('meta[name="theme-color"]', 'meta', {{name: 'theme-color', content: '{theme_color}'}});
    ensure('meta[name="apple-mobile-web-app-capable"]', 'meta', {{name: 'apple-mobile-web-app-capable', content: 'yes'}});
    ensure('meta[name="apple-mobile-web-app-status-bar-style"]', 'meta', {{name: 'apple-mobile-web-app-status-bar-style', content: 'black-translucent'}});
    ensure('meta[name="apple-mobile-web-app-title"]', 'meta', {{name: 'apple-mobile-web-app-title', content: '{short_name}'}});
    ensure('link[rel="apple-touch-icon"]', 'link', {{rel: 'apple-touch-icon', href: 'data:image/png;base64,{icon192}'}});
    ensure('meta[name="mobile-web-app-capable"]', 'meta', {{name: 'mobile-web-app-capable', content: 'yes'}});
  }} catch (e) {{ /* iframe not same-origin, or browser blocked it - the app still works without this */ }}
}})();
</script>"""
    components.html(js, height=0, width=0)
