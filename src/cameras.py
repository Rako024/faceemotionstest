"""Kameraların siyahısı və seçilmiş kameranın yadda saxlanması.

Windows-da siyahı DirectShow-dan oxunur — CameraSource da DirectShow (cv2.CAP_DSHOW) ilə açır,
ona görə siyahıdakı sıra OpenCV index-i ilə eynidir. Siyahılama kameranı AÇMIR (işıq yanmır):
yalnız cihaz adları oxunur, razılıqdan əvvəl də təhlükəsizdir.

Seçim local_state.json-da (ad + cihaz yolu) saxlanır: USB kamera çıxarılıb başqa porta taxılsa,
index dəyişsə də eyni kamera tapılır. Fayl yoxdursa və ya kamera taxılmayıbsa — capture.camera_index.
"""

from __future__ import annotations

import ctypes
import json
import logging
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)

STATE_PATH = PROJECT_ROOT / "local_state.json"


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    path: str = ""  # DirectShow DevicePath (virtual kameralarda boş ola bilər)


# --- DirectShow (ctypes COM, əlavə asılılıq yoxdur) -----------------------------

_CLSID_SystemDeviceEnum = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"
_IID_ICreateDevEnum = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VideoInputDeviceCategory = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
_IID_IPropertyBag = "{55272A00-42CB-11CE-8135-00AA004BB851}"
_VT_BSTR = 8


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]


class _VARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_uint16), ("r1", ctypes.c_uint16), ("r2", ctypes.c_uint16), ("r3", ctypes.c_uint16),
                ("val", ctypes.c_void_p), ("pad", ctypes.c_void_p)]


def _vcall(obj: ctypes.c_void_p, index: int, argtypes: list, *args) -> int:
    """COM vtable çağırışı: obj->lpVtbl[index](obj, *args)."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])
    return fn(obj, *args)


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _vcall(obj, 2, [])


def _dshow_devices() -> list[CameraInfo]:
    ole32, oleaut32 = ctypes.windll.ole32, ctypes.windll.oleaut32

    def guid(text: str) -> _GUID:
        g = _GUID()
        ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(g))
        return g

    out: list[CameraInfo] = []
    hr = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED — öz thread-imizdə
    try:
        dev_enum = ctypes.c_void_p()
        if ole32.CoCreateInstance(ctypes.byref(guid(_CLSID_SystemDeviceEnum)), None, 1,
                                  ctypes.byref(guid(_IID_ICreateDevEnum)), ctypes.byref(dev_enum)) != 0:
            return out
        try:
            enum = ctypes.c_void_p()
            # ICreateDevEnum::CreateClassEnumerator — kamera yoxdursa S_FALSE və NULL
            r = _vcall(dev_enum, 3, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32],
                       ctypes.byref(guid(_CLSID_VideoInputDeviceCategory)), ctypes.byref(enum), 0)
            if r != 0 or not enum:
                return out
            try:
                iid_bag = guid(_IID_IPropertyBag)
                while True:
                    moniker, fetched = ctypes.c_void_p(), ctypes.c_uint32()
                    # IEnumMoniker::Next
                    if _vcall(enum, 3, [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p],
                              1, ctypes.byref(moniker), ctypes.byref(fetched)) != 0 or not fetched.value:
                        break
                    try:
                        bag = ctypes.c_void_p()
                        # IMoniker::BindToStorage(NULL, NULL, IID_IPropertyBag, &bag)
                        if _vcall(moniker, 9, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
                                  None, None, ctypes.byref(iid_bag), ctypes.byref(bag)) != 0:
                            continue
                        try:
                            props = {}
                            for key in ("FriendlyName", "DevicePath"):
                                var = _VARIANT()
                                # IPropertyBag::Read
                                if _vcall(bag, 3, [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p],
                                          key, ctypes.byref(var), None) == 0 and var.vt == _VT_BSTR and var.val:
                                    props[key] = ctypes.wstring_at(var.val)
                                oleaut32.VariantClear(ctypes.byref(var))
                            out.append(CameraInfo(len(out), props.get("FriendlyName", f"Kamera {len(out) + 1}"),
                                                  props.get("DevicePath", "")))
                        finally:
                            _release(bag)
                    finally:
                        _release(moniker)
            finally:
                _release(enum)
        finally:
            _release(dev_enum)
    finally:
        if hr in (0, 1):  # S_OK / S_FALSE — başlatmışıqsa bağlayırıq
            ole32.CoUninitialize()
    return out


def list_cameras() -> list[CameraInfo]:
    """Qoşulmuş kameralar (OpenCV index sırası ilə). Kameranı açmır."""
    if sys.platform != "win32":
        return []
    result: list[CameraInfo] = []

    def run() -> None:
        try:
            result.extend(_dshow_devices())
        except Exception:  # noqa: BLE001 - siyahı olmasa da kamera index ilə işləyir
            log.exception("Kamera siyahısı oxunmadı")

    # COM-u ayrıca thread-də başladırıq ki, UI/körpü thread-lərinin apartment-inə toxunmasın
    t = threading.Thread(target=run, name="CameraList", daemon=True)
    t.start()
    t.join(timeout=5.0)
    return _number_duplicates(result)


def _number_duplicates(cams: list[CameraInfo]) -> list[CameraInfo]:
    """Eyni adlı iki kamera: "USB Camera", "USB Camera (2)"."""
    seen: dict[str, int] = {}
    out = []
    for c in cams:
        seen[c.name] = seen.get(c.name, 0) + 1
        out.append(c if seen[c.name] == 1 else CameraInfo(c.index, f"{c.name} ({seen[c.name]})", c.path))
    return out


# --- seçimin yadda saxlanması -----------------------------------------------------


def load_choice(path: Path | None = None) -> dict | None:
    path = path or STATE_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cam = data.get("camera") if isinstance(data, dict) else None
    return cam if isinstance(cam, dict) and "name" in cam else None


def save_choice(cam: CameraInfo, path: Path | None = None) -> None:
    path = path or STATE_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data["camera"] = {"name": cam.name, "path": cam.path}
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        log.warning("Kamera seçimi yadda saxlanmadı: %s", path)


def resolve(cams: list[CameraInfo], choice: dict | None, default_index: int) -> CameraInfo | None:
    """Yadda qalan kamera (əvvəl cihaz yolu, sonra ad üzrə), yoxdursa config index-i."""
    if choice:
        if choice.get("path"):
            for c in cams:
                if c.path == choice["path"]:
                    return c
        for c in cams:
            if c.name == choice.get("name"):
                return c
    for c in cams:
        if c.index == default_index:
            return c
    return cams[0] if cams else None


def as_dicts(cams: list[CameraInfo]) -> list[dict]:
    return [asdict(c) for c in cams]
