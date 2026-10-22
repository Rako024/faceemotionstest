"""Windows 11 DWM köməkçiləri: Mica/Acrylic şüşə, yuvarlaq künclər, kapsul forması.

Hamısı "best effort"-dur: dəstəklənməyən sistemdə False qaytarır və UI CSS fonuna
keçir. Windows-dan başqa platformalarda heç nə etmir.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2
BACKDROPS = {"mica": 2, "acrylic": 3}


def _build() -> int:
    return sys.getwindowsversion().build if IS_WINDOWS else 0


def supports_backdrop() -> bool:
    """DWMWA_SYSTEMBACKDROP_TYPE Windows 11 22H2 (build 22621) və sonrası."""
    return _build() >= 22621


def resolve_material(requested: str) -> str:
    if requested == "none" or not supports_backdrop():
        return "none"
    return "mica" if requested == "auto" else requested


def is_dark_mode() -> bool:
    if not IS_WINDOWS:
        return False
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        ) as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def _hwnd(window) -> int | None:
    form = getattr(window, "native", None)
    if form is None:
        return None
    try:
        return int(form.Handle.ToInt64())
    except Exception:  # noqa: BLE001
        return None


def _set_attr(hwnd: int, attr: int, value: int) -> bool:
    import ctypes
    from ctypes import wintypes

    v = ctypes.c_int(value)
    return ctypes.windll.dwmapi.DwmSetWindowAttribute(wintypes.HWND(hwnd), attr, ctypes.byref(v), 4) == 0


def apply_backdrop(window, material: str) -> bool:
    """Pəncərəyə Mica/Acrylic şüşə fonu və yuvarlaq künclər verir."""
    if not IS_WINDOWS:
        return False
    hwnd = _hwnd(window)
    if hwnd is None:
        return False
    import ctypes
    from ctypes import wintypes

    ok = _set_attr(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
    if material not in BACKDROPS:
        return ok
    try:
        from System import Action
        from System.Drawing import Color

        form = window.native

        def make_transparent():
            # Qara forma fonu + şəffaf WebView2 → DWM materialı görünür.
            # (pywebview-in transparent=True rejimi gizli pəncərəni də açır, ona görə özümüz edirik.)
            form.BackColor = Color.Black
            webview = getattr(form, "webview", None)
            if webview is not None:
                webview.DefaultBackgroundColor = Color.Transparent

        form.Invoke(Action(make_transparent))

        class MARGINS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]

        ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(wintypes.HWND(hwnd), ctypes.byref(MARGINS(-1, -1, -1, -1)))
        _set_attr(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if is_dark_mode() else 0)
        return _set_attr(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, BACKDROPS[material])
    except Exception:  # noqa: BLE001
        log.warning("Şüşə fonu tətbiq olunmadı", exc_info=True)
        return False


def dpi_scale(window=None) -> float:
    if not IS_WINDOWS:
        return 1.0
    import ctypes
    from ctypes import wintypes

    try:
        hwnd = _hwnd(window) if window is not None else None
        dpi = ctypes.windll.user32.GetDpiForWindow(wintypes.HWND(hwnd)) if hwnd else ctypes.windll.user32.GetDpiForSystem()
        return (dpi or 96) / 96
    except Exception:  # noqa: BLE001
        return 1.0


def make_pill(window, width: int, height: int) -> bool:
    """Pəncərəni kapsul formasına salır (Dynamic Island)."""
    if not IS_WINDOWS:
        return False
    hwnd = _hwnd(window)
    if hwnd is None:
        return False
    import ctypes
    from ctypes import wintypes

    scale = dpi_scale(window)
    w, h = round(width * scale), round(height * scale)
    _set_attr(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND)
    region = ctypes.windll.gdi32.CreateRoundRectRgn(0, 0, w + 1, h + 1, h, h)
    return bool(ctypes.windll.user32.SetWindowRgn(wintypes.HWND(hwnd), region, True))


def work_area() -> tuple[int, int, int, int]:
    """Əsas monitorun iş sahəsi (taskbar xaric), məntiqi piksellərlə: (x, y, w, h)."""
    if not IS_WINDOWS:
        return 0, 0, 1440, 900
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
    scale = dpi_scale()
    return (
        round(rect.left / scale),
        round(rect.top / scale),
        round((rect.right - rect.left) / scale),
        round((rect.bottom - rect.top) / scale),
    )


def make_tool_window(window) -> None:
    """Taskbar-da və Alt+Tab siyahısında görünməsin (indikator, bildiriş)."""
    hwnd = _hwnd(window) if IS_WINDOWS else None
    if hwnd is None:
        return
    import ctypes
    from ctypes import wintypes

    GWL_EXSTYLE, WS_EX_TOOLWINDOW, WS_EX_APPWINDOW = -20, 0x00000080, 0x00040000
    user32 = ctypes.windll.user32
    style = user32.GetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE)
    user32.SetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE, (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW)


def _set_visible(window, visible: bool) -> None:
    from System import Action

    form = window.native
    form.Invoke(Action(lambda: setattr(form, "Visible", visible)))


def show_without_focus(window, x: int, y: int, width: int, height: int) -> None:
    """Pəncərəni ən üstdə göstərir, amma operatorun işlədiyi proqramdan fokusu almır.

    pywebview-in show() metodu Activate() çağırır. Biz WinForms Visible=True edirik
    (WebView2 render etsin deyə), mövqeyi SWP_NOACTIVATE ilə qoyuruq və fokus yenə də
    dəyişibsə, dərhal əvvəlki pəncərəyə qaytarırıq.
    """
    hwnd = _hwnd(window) if IS_WINDOWS else None
    if hwnd is None:
        window.move(x, y)
        window.show()
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    previous = user32.GetForegroundWindow()
    _set_visible(window, True)
    scale = dpi_scale(window)
    HWND_TOPMOST, SWP_NOACTIVATE, SWP_SHOWWINDOW = -1, 0x0010, 0x0040
    user32.SetWindowPos(
        wintypes.HWND(hwnd), wintypes.HWND(HWND_TOPMOST),
        round(x * scale), round(y * scale), round(width * scale), round(height * scale),
        SWP_NOACTIVATE | SWP_SHOWWINDOW,
    )
    if previous and user32.GetForegroundWindow() != previous:
        user32.SetForegroundWindow(previous)


def hide(window) -> None:
    if not IS_WINDOWS or _hwnd(window) is None:
        window.hide()
        return
    _set_visible(window, False)


def bring_to_front(window) -> None:
    if not IS_WINDOWS:
        return
    hwnd = _hwnd(window)
    if hwnd is None:
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.ShowWindow(wintypes.HWND(hwnd), 9)  # SW_RESTORE
    user32.SetForegroundWindow(wintypes.HWND(hwnd))
