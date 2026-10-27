"""JavaScript → Python körpüsü (pywebview js_api).

Yalnız açıq (alt xətsiz) metodlar JS-ə görünür. Bütün qaytarılan dəyərlər
JSON-a çevrilə bilən sadə tiplərdir.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .window import UiController

log = logging.getLogger(__name__)


class Bridge:
    def __init__(self, controller: "UiController") -> None:
        self._c = controller

    # --- vəziyyət ---------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return self._c.session.state()

    def preview(self) -> str | None:
        return self._c.session.preview_data_url()

    def set_preview(self, enabled: bool) -> bool:
        self._c.session.preview_enabled = bool(enabled)
        return True

    def island_state(self) -> dict[str, Any]:
        s = self._c.session
        st = s.state()
        result = st.get("result") or {}
        label = None
        if st["phase"] == "running":
            if result.get("status") == "ok" and result.get("top_label"):
                label = result["top_label"]
            elif result.get("status") == "no_face":
                label = "Üz yoxdur"
        return {"active": st["camera_active"], "label": label}

    # --- həyat dövrü ------------------------------------------------------------
    def accept_consent(self) -> bool:
        self._c.session.accept_consent()
        return True

    def decline(self) -> bool:
        log.info("Razılıq verilmədi — kamera açılmadan çıxılır")
        self._c.quit()
        return True

    def stop(self) -> bool:
        self._c.session.stop()
        return True

    def restart(self) -> bool:
        self._c.session.restart()
        return True

    def close_app(self) -> bool:
        self._c.quit()
        return True

    # --- pəncərə ----------------------------------------------------------------
    def minimize(self) -> bool:
        self._c.main.minimize()
        return True

    def toggle_compact(self) -> bool:
        return self._c.toggle_compact()

    def focus_main(self) -> bool:
        self._c.focus_main()
        return True

    # --- kamera -----------------------------------------------------------------
    def cameras(self, refresh: bool = False) -> dict[str, Any]:
        s = self._c.session
        if refresh:
            s.refresh_cameras()
        st = s.state()
        return {"cameras": st["cameras"], "camera": st["camera"]}

    def select_camera(self, index: int) -> dict[str, Any]:
        ok, error = self._c.session.select_camera(int(index))
        return {"ok": ok, "error": error}

    # --- modellər ---------------------------------------------------------------
    def set_primary(self, key: str) -> bool:
        return self._c.session.set_primary(str(key))

    def set_model_enabled(self, key: str, enabled: bool) -> bool:
        return self._c.session.set_model_enabled(str(key), bool(enabled))

    def set_compare_mode(self, mode: str) -> bool:
        return self._c.session.set_compare_mode(str(mode))

    # --- nəzarətli test ---------------------------------------------------------
    def start_test(self, tester_id: str | None = None, save_frames: bool = False) -> dict[str, Any]:
        s = self._c.session
        ok = s.start_test(tester_id=(str(tester_id).strip() or None) if tester_id else None, save_frames=bool(save_frames))
        return {"ok": ok, "error": None if ok else (s.test_error or "Test başlamadı")}

    def cancel_test(self) -> bool:
        return self._c.session.cancel_test()

    def dismiss_test(self) -> bool:
        return self._c.session.dismiss_test()

    # --- geyim ------------------------------------------------------------------
    def check_attire(self) -> dict[str, Any]:
        ok, error = self._c.session.check_attire()
        return {"ok": ok, "error": error}

    # --- özünüqiymətləndirmə ----------------------------------------------------
    def prompt_state(self) -> dict[str, Any] | None:
        return self._c.session.prompt_state()

    def answer(self, prompt_id: int, code: str) -> bool:
        return self._c.answer(int(prompt_id), str(code))

    def demo_prompt(self) -> bool:
        return self._c.session.open_prompt_now() is not None
