"""UI qatının pəncərəsiz hissələri: səhifə yığımı, CSP, körpü, boot məlumatı."""

import json
import re
from types import SimpleNamespace

import pytest

from src.session import Phase
from src.ui import native
from src.ui.assets import CSP, WEB_DIR, render_page
from src.ui.bridge import Bridge
from src.ui.window import make_boot, model_short

PAGES = ["index", "island", "prompt"]
BOOT = {"operator_id": "OP-1", "classes": [], "categories": [], "note": "</script><b>x</b>"}


@pytest.mark.parametrize("page", PAGES)
def test_page_is_self_contained(page):
    html = render_page(page, BOOT)
    assert "<link" not in html
    assert not re.search(r"<script[^>]+src=", html), "bütün skriptlər inline olmalıdır"
    assert not re.search(r"https?://", html), "xarici URL olmamalıdır"
    assert "window.MockApi = {" not in html, "dev mock daxil edilməməlidir"
    assert 'src="mock.js"' not in html
    assert f'content="{CSP}"' in html
    assert "window.__BOOT__" in html


def test_csp_blocks_network():
    assert "default-src 'none'" in CSP
    assert "connect-src 'none'" in CSP
    assert "img-src data:" in CSP


def test_boot_json_cannot_break_out_of_script():
    html = render_page("index", BOOT)
    boot_line = next(line for line in html.splitlines() if "window.__BOOT__" in line)
    assert "</script><b>" not in boot_line
    payload = boot_line.split("=", 1)[1].rstrip(";</script>").strip()
    assert json.loads(payload.replace("<\\/", "</"))["note"] == BOOT["note"]


def test_referenced_assets_exist():
    for page in PAGES:
        src = (WEB_DIR / f"{page}.html").read_text(encoding="utf-8")
        for ref in re.findall(r'(?:href|src)="([^"]+)"', src):
            assert (WEB_DIR / ref).is_file(), f"{page}.html → {ref} yoxdur"


def test_render_rejects_path_traversal():
    from src.ui.assets import _read

    with pytest.raises(ValueError):
        _read("../../config.py")


@pytest.mark.parametrize(
    "name,short",
    [
        ("enet_b0_8_best_afew", "enet_b0 · afew"),
        ("enet_b0_8_best_vgaf", "enet_b0 · vgaf"),
        ("enet_b0_8_va_mtl", "enet_b0 · VA"),
        ("enet_b2_8", "enet_b2"),
        ("mbf_va_mtl", "mbf · VA"),
    ],
)
def test_model_short(name, short):
    assert model_short(name) == short


def test_make_boot(cfg):
    session = SimpleNamespace(source_kind="camera")
    boot = make_boot(cfg, session, "mica")
    assert boot["operator_id"] == cfg.operator_id
    assert [c["key"] for c in boot["classes"]][:2] == ["Anger", "Contempt"]
    assert {c["code"] for c in boot["categories"]} == set(cfg.self_report.categories)
    assert boot["prompt_interval_min"] == "10"
    assert boot["prompt_timeout_s"] == "30"
    assert boot["material"] == "mica"
    assert boot["attire_enabled"] is cfg.attire.live.enabled and boot["attire_frames"] == cfg.attire.live.frames
    json.dumps(boot)  # JSON-a çevrilə bilməlidir


def test_resolve_material(monkeypatch):
    monkeypatch.setattr(native, "_build", lambda: 26200)
    assert native.resolve_material("auto") == "mica"
    assert native.resolve_material("acrylic") == "acrylic"
    assert native.resolve_material("none") == "none"
    monkeypatch.setattr(native, "_build", lambda: 19045)  # Windows 10
    assert native.resolve_material("auto") == "none"


class FakeSession:
    def __init__(self, st):
        self._st = st
        self.preview_enabled = False
        self.answers = []

    def state(self):
        return self._st

    def check_attire(self):
        return (False, "Kamera açıq deyil") if self._st["phase"] != "running" else (True, None)


class FakeController:
    def __init__(self, st):
        self.session = FakeSession(st)
        self.quit_called = False

    def quit(self):
        self.quit_called = True

    def answer(self, pid, code):
        self.session.answers.append((pid, code))
        return code != "bad"


def _state(phase="running", result=None, active=True):
    return {"phase": phase, "camera_active": active, "result": result}


def test_island_shows_emotion_only_when_certain():
    b = Bridge(FakeController(_state(result={"status": "ok", "top_label": "Neytral"})))
    assert b.island_state() == {"active": True, "label": "Neytral"}
    b = Bridge(FakeController(_state(result={"status": "uncertain", "top_label": "Neytral"})))
    assert b.island_state()["label"] is None
    b = Bridge(FakeController(_state(result={"status": "no_face"})))
    assert b.island_state()["label"] == "Üz yoxdur"


def test_island_blind_mode_has_no_label():
    b = Bridge(FakeController(_state(result={"status": "ok"})))  # kor rejimdə top_label yoxdur
    assert b.island_state()["label"] is None


def test_bridge_decline_quits_without_camera():
    c = FakeController(_state(phase=Phase.CONSENT.value, active=False))
    assert Bridge(c).decline() is True
    assert c.quit_called


def test_bridge_answer_casts_types():
    c = FakeController(_state())
    b = Bridge(c)
    assert b.answer("3", "happy") is True
    assert c.session.answers == [(3, "happy")]
    assert b.answer(4, "bad") is False


def test_bridge_check_attire_returns_plain_dict():
    assert Bridge(FakeController(_state())).check_attire() == {"ok": True, "error": None}
    res = Bridge(FakeController(_state(phase="stopped", active=False))).check_attire()
    assert res == {"ok": False, "error": "Kamera açıq deyil"}


def test_attire_card_is_on_page_and_in_consent():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="attire-card"' in html and 'data-action="check-attire"' in html
    consent = html.split('id="consent"')[1].split("</section>")[0]
    assert "Geyim yoxlaması" in consent and "yalnız ekranda" in consent


def test_bridge_exposes_no_private_members():
    public = [n for n in dir(Bridge) if not n.startswith("_")]
    assert set(public) == {
        "state", "preview", "set_preview", "island_state", "accept_consent", "decline", "stop",
        "restart", "close_app", "minimize", "toggle_compact", "focus_main", "prompt_state",
        "answer", "demo_prompt", "set_primary", "set_model_enabled", "set_compare_mode",
        "start_test", "cancel_test", "dismiss_test", "cameras", "select_camera", "check_attire",
    }  # fmt: skip
