"""Pəncərələr: əsas panel, kamera indikatoru ("Dynamic Island"), sorğu bildirişi."""

from __future__ import annotations

import logging
import os
import threading

from ..config import EMOTION_CLASSES, AppConfig
from ..labels import EMOTION_LABELS
from ..selfreport import Prompt
from ..session import Phase, Session
from . import native
from .assets import render_page
from .bridge import Bridge

log = logging.getLogger(__name__)

MAIN_W, MAIN_H = 1240, 800  # geniş panel (iş sahəsinə sığdırılır)
COMPACT_W, COMPACT_H = 400, 318
MIN_W = 1000
PROMPT_W, PROMPT_H = 360, 290
ISLAND_H = 38
EDGE = 16

# WebView2-nin öz arxa fon şəbəkə funksiyalarını söndürür (SmartScreen, komponent
# yeniləmələri, telemetriya hesabatları). Səhifənin özü CSP ilə bağlıdır.
WEBVIEW2_ARGS = " ".join(
    [
        "--disable-features=ElasticOverscroll,msSmartScreenProtection,Translate,AutofillServerCommunication",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-domain-reliability",
        "--disable-sync",
        "--disable-client-side-phishing-detection",
        "--no-pings",
    ]
)


def model_short(name: str) -> str:
    base = name.split("_8")[0].split("_va")[0]
    if name.endswith("_va_mtl"):
        return f"{base} · VA"
    suffix = name.rsplit("_best_", 1)[1] if "_best_" in name else ""
    return f"{base} · {suffix}" if suffix else base


def _fmt_minutes(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def make_boot(cfg: AppConfig, session: Session, material: str) -> dict:
    m = cfg.model
    engine = "ONNX · CPU" if m.engine == "onnx" else f"PyTorch · {m.device.upper()}"
    sr = cfg.self_report
    n_cmp = len(dict.fromkeys([m.name, *cfg.compare.models]))
    compare_label = {
        "off": "Söndürülüb (tək model)",
        "auto": f"Avto — {n_cmp} modelə qədər, gücə görə",
        "on": f"Hamısı — {n_cmp} model",
    }[cfg.compare.mode]
    return {
        "operator_id": cfg.operator_id,
        "compare_label": compare_label,
        "model": m.name,
        "model_short": model_short(m.name),
        "engine_label": engine,
        "analysis_fps": f"{cfg.capture.analysis_fps:g}",
        "window_seconds": f"{cfg.smoothing.window_seconds:g}",
        "min_frames": cfg.smoothing.min_frames,
        "source": session.source_kind,
        "prompts_enabled": sr.enabled,
        "prompt_interval_min": _fmt_minutes(sr.interval_minutes),
        "prompt_timeout_s": f"{sr.timeout_seconds:g}",
        "show_prediction": cfg.ui.show_prediction,
        "preview_fps": cfg.ui.preview_fps,
        "is_mtl": m.is_mtl,
        "material": material,
        "classes": [{"key": c, "label": EMOTION_LABELS[c]} for c in EMOTION_CLASSES],
        "categories": [{"code": k, "label": v} for k, v in sr.categories.items()],
        "decline_label": sr.decline_label,
        "initial_phase": "consent",
        "uncertain_threshold": cfg.smoothing.uncertain_threshold,
        "research_enabled": cfg.research.save_test_frames,
        "test_expressions": len(cfg.test.expressions),
        "test_rounds": cfg.test.rounds,
        "test_minutes": max(1, round((cfg.test.intro_seconds + len(cfg.test.expressions) * cfg.test.rounds
                                       * (cfg.test.prepare_seconds + cfg.test.hold_seconds + cfg.test.rest_seconds)) / 60)),
    }


class UiController:
    def __init__(self, cfg: AppConfig, video: str | None = None) -> None:
        import webview

        self.cfg = cfg
        self.session = Session(cfg, video=video)
        self.session.on_change = self._on_session_change
        self.session.on_prompt = self._on_prompt
        self.bridge = Bridge(self)
        self.compact = False
        self._quitting = False
        self._hide_delay = 0.0
        self._prompt_visible = False

        self.material = native.resolve_material(cfg.ui.material)
        wall = "#101014" if native.is_dark_mode() else "#eef0f6"
        boot = make_boot(cfg, self.session, self.material)

        wx, wy, ww, wh = native.work_area()
        self.work = (wx, wy, ww, wh)
        self.main_w = max(min(MAIN_W, ww - 3 * EDGE), min(MIN_W, ww))
        self.main_h = min(MAIN_H, wh - 2 * EDGE)
        self.island_w = 300 if cfg.ui.show_prediction else 236

        self.main = webview.create_window(
            "Emosiya Pilotu",
            html=render_page("index", boot),
            js_api=self.bridge,
            width=self.main_w,
            height=self.main_h,
            x=wx + (ww - self.main_w) // 2,
            y=wy + max(EDGE, (wh - self.main_h) // 2),
            frameless=True,
            easy_drag=False,
            resizable=False,
            background_color=wall,
            min_size=(COMPACT_W, COMPACT_H),
        )
        self.island = webview.create_window(
            "Kamera aktiv",
            html=render_page("island", boot),
            js_api=self.bridge,
            width=self.island_w,
            height=ISLAND_H,
            x=wx + (ww - self.island_w) // 2,
            y=wy + 8,
            frameless=True,
            resizable=False,
            on_top=True,
            focus=False,
            hidden=True,
            shadow=False,
            background_color="#0b0b0c",
            min_size=(100, 30),
        )
        self.prompt = webview.create_window(
            "Əhval sorğusu",
            html=render_page("prompt", boot),
            js_api=self.bridge,
            width=PROMPT_W,
            height=PROMPT_H,
            x=wx + ww - PROMPT_W - EDGE,
            y=wy + wh - PROMPT_H - EDGE,
            frameless=True,
            resizable=False,
            on_top=True,
            focus=False,
            hidden=True,
            background_color=wall,
            min_size=(200, 100),
        )
        self.main.events.closed += self._on_main_closed

    # --- başlanğıc --------------------------------------------------------------

    def run(self) -> None:
        import webview

        os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", WEBVIEW2_ARGS)
        webview.settings["ALLOW_FILE_URLS"] = False
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        # Başlıq zolağındakı düymələr sürükləməni başlatmasın
        webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
        webview.start(self._on_started, private_mode=True)

    def _on_started(self) -> None:
        for w in (self.main, self.island, self.prompt):
            w.events.loaded.wait(15)
        # Çərçivəsiz pəncərə yaradılarkən ölçü bir qədər kiçilir — dəqiq ölçünü bərpa et.
        # (Gizli pəncərələrə pywebview resize() çağırmırıq: o, SWP_SHOWWINDOW ilə onları göstərir.
        # Onların ölçüsü show_without_focus-da verilir.)
        self.main.resize(self.main_w, self.main_h)
        if self.material != "none":
            for w in (self.main, self.prompt):
                if not native.apply_backdrop(w, self.material):
                    w.evaluate_js("document.documentElement.classList.remove('material')")
        else:
            native.apply_backdrop(self.main, "none")
            native.apply_backdrop(self.prompt, "none")
        native.make_pill(self.island, self.island_w, ISLAND_H)
        native.make_tool_window(self.island)
        native.make_tool_window(self.prompt)

    # --- sessiya hadisələri -----------------------------------------------------

    def _on_session_change(self) -> None:
        if self._quitting:
            return
        if self.session.camera_active:
            wx, wy, ww, _ = self.work
            native.show_without_focus(self.island, wx + (ww - self.island_w) // 2, wy + 8, self.island_w, ISLAND_H)
        else:
            native.hide(self.island)
        if self.session.phase is not Phase.RUNNING and self._prompt_visible:
            self._hide_prompt()

    def _on_prompt(self, prompt: Prompt | None) -> None:
        if self._quitting:
            return
        if prompt is None:
            delay, self._hide_delay = self._hide_delay, 0.0
            if delay:
                threading.Timer(delay, self._hide_prompt).start()
            else:
                self._hide_prompt()
            return
        timeout = self.cfg.self_report.timeout_seconds
        self.prompt.evaluate_js(f"window.__open && window.__open({prompt.id}, {timeout}, {timeout})")
        wx, wy, ww, wh = self.work
        native.show_without_focus(
            self.prompt, wx + ww - PROMPT_W - EDGE, wy + wh - PROMPT_H - EDGE, PROMPT_W, PROMPT_H
        )
        self._prompt_visible = True

    def _hide_prompt(self) -> None:
        self._prompt_visible = False
        if not self._quitting:
            native.hide(self.prompt)

    def answer(self, prompt_id: int, code: str) -> bool:
        self._hide_delay = 0.9  # "Təşəkkürlər" animasiyası görünsün
        try:
            ok = self.session.answer(prompt_id, code)
        except ValueError:
            ok = False
        if not ok:
            self._hide_delay = 0.0
        return ok

    # --- pəncərə əməliyyatları ----------------------------------------------------

    def toggle_compact(self) -> bool:
        """Geniş panel ↔ yalnız əsas nəticəni göstərən kiçik pəncərə (sağ yuxarı küncdə)."""
        from webview.window import FixPoint

        self.compact = not self.compact
        if self.compact:
            self.main.resize(COMPACT_W, COMPACT_H, fix_point=FixPoint.NORTH | FixPoint.EAST)
        else:
            self.main.resize(self.main_w, self.main_h, fix_point=FixPoint.NORTH | FixPoint.EAST)
        return self.compact

    def focus_main(self) -> None:
        native.bring_to_front(self.main)

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self.session.shutdown()
        for w in (self.prompt, self.island, self.main):
            try:
                w.destroy()
            except Exception:  # noqa: BLE001
                log.debug("pəncərə bağlanarkən xəta", exc_info=True)

    def _on_main_closed(self) -> None:
        # Alt+F4 və s. ilə bağlananda da kamera dayanmalıdır
        threading.Thread(target=self.quit, daemon=True).start()


def run_ui(cfg: AppConfig, video: str | None = None) -> int:
    controller = UiController(cfg, video=video)
    try:
        controller.run()
    finally:
        controller.session.shutdown()
    return 0
