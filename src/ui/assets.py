"""HTML səhifələrini tək, özünə-kifayət sənədə çevirir.

- <link rel="stylesheet"> və <script src> faylları inline edilir → HTTP server və
  port açılmır, heç bir xarici resurs yüklənmir.
- data-dev ilə işarələnmiş skriptlər (brauzer önizləməsi üçün mock) atılır.
- Content-Security-Policy hər hansı şəbəkə sorğusunu qadağan edir.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

WEB_DIR = Path(__file__).resolve().parent / "web"

CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; "
    "img-src data:; "
    "connect-src 'none'; "
    "font-src 'none'; "
    "media-src 'none'; "
    "frame-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

_LINK_RE = re.compile(r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*/?>')
_SCRIPT_RE = re.compile(r'<script\s+src="([^"]+)"([^>]*)></script>')


def _read(name: str) -> str:
    path = (WEB_DIR / name).resolve()
    if WEB_DIR not in path.parents:
        raise ValueError(f"web qovluğundan kənar fayl: {name}")
    return path.read_text(encoding="utf-8")


def render_page(name: str, boot: dict[str, Any]) -> str:
    html = _read(f"{name}.html")

    html = _LINK_RE.sub(lambda m: f"<style>\n{_read(m.group(1))}\n</style>", html)

    def script(m: re.Match) -> str:
        if "data-dev" in m.group(2):
            return ""
        return f"<script>\n{_read(m.group(1))}\n</script>"

    html = _SCRIPT_RE.sub(script, html)

    # "</" ardıcıllığı skripti vaxtından əvvəl bağlamasın
    boot_json = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    head = (
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}">\n'
        f"<script>window.__BOOT__ = {boot_json};</script>\n"
    )
    return html.replace("<head>", "<head>\n" + head, 1)
