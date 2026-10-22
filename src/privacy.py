"""Məxfilik qoruyucuları — qaydaları sözlə yox, kodla təmin edir.

1. NetworkGuard: modellər yükləndikdən sonra loopback-dən başqa bütün şəbəkə
   bağlantılarını (DNS daxil) bloklayır.
2. DiskImageGuard: kadrların diskə yazıla biləcəyi API-ləri (cv2.imwrite,
   cv2.VideoWriter, PIL Image.save) iş vaxtı bağlayır.
3. ConsentToken: kamera yalnız istifadəçinin razılığı ilə yaradılan token
   təqdim edildikdə açılır.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


class NetworkBlockedError(PermissionError):
    pass


class DiskWriteBlockedError(PermissionError):
    pass


class ConsentError(PermissionError):
    pass


# --- şəbəkə ---------------------------------------------------------------------

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", ""}


def _is_loopback_host(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="ignore")
    if not isinstance(host, str):
        return False
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _address_allowed(sock: socket.socket, address: Any) -> bool:
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return True
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


class NetworkGuard:
    """Proses daxilində Python səviyyəsində bütün xarici şəbəkə çağırışlarını bloklayır.

    Loopback (127.0.0.1/::1) icazəlidir — Windows-da asyncio/socketpair daxili
    oyanma üçün ondan istifadə edir. Bloklanan hər cəhd log-a düşür.
    """

    def __init__(self) -> None:
        self._originals: dict[str, Any] = {}
        self.blocked_attempts: list[str] = []
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return bool(self._originals)

    def _block(self, what: str) -> None:
        with self._lock:
            self.blocked_attempts.append(what)
        log.error("Şəbəkə çağırışı bloklandı: %s", what)
        raise NetworkBlockedError(f"Şəbəkə bloklanıb (privacy.block_network): {what}")

    def install(self) -> "NetworkGuard":
        if self.active:
            return self
        guard = self
        orig_connect = socket.socket.connect
        orig_connect_ex = socket.socket.connect_ex
        orig_sendto = socket.socket.sendto
        orig_getaddrinfo = socket.getaddrinfo
        orig_create_connection = socket.create_connection

        def connect(self, address):
            if not _address_allowed(self, address):
                guard._block(f"connect {address!r}")
            return orig_connect(self, address)

        def connect_ex(self, address):
            if not _address_allowed(self, address):
                guard._block(f"connect_ex {address!r}")
            return orig_connect_ex(self, address)

        def sendto(self, data, *args):
            address = args[-1] if args else None
            if not _address_allowed(self, address):
                guard._block(f"sendto {address!r}")
            return orig_sendto(self, data, *args)

        def getaddrinfo(host, *args, **kwargs):
            if not _is_loopback_host(host if host is not None else ""):
                guard._block(f"getaddrinfo {host!r}")
            return orig_getaddrinfo(host, *args, **kwargs)

        def create_connection(address, *args, **kwargs):
            if not (isinstance(address, tuple) and _is_loopback_host(address[0])):
                guard._block(f"create_connection {address!r}")
            return orig_create_connection(address, *args, **kwargs)

        self._originals = {
            "connect": orig_connect,
            "connect_ex": orig_connect_ex,
            "sendto": orig_sendto,
            "getaddrinfo": orig_getaddrinfo,
            "create_connection": orig_create_connection,
        }
        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.socket.sendto = sendto
        socket.getaddrinfo = getaddrinfo
        socket.create_connection = create_connection
        log.info("Şəbəkə qoruyucusu aktivdir — yalnız loopback icazəlidir")
        return self

    def uninstall(self) -> None:
        if not self.active:
            return
        socket.socket.connect = self._originals["connect"]
        socket.socket.connect_ex = self._originals["connect_ex"]
        socket.socket.sendto = self._originals["sendto"]
        socket.getaddrinfo = self._originals["getaddrinfo"]
        socket.create_connection = self._originals["create_connection"]
        self._originals = {}

    def __enter__(self) -> "NetworkGuard":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()


# --- disk -----------------------------------------------------------------------


class DiskImageGuard:
    """Kadrları diskə yaza biləcək API-ləri bağlayır. cv2.imencode (yaddaşda) icazəlidir."""

    def __init__(self) -> None:
        self._patches: list[tuple[Any, str, Any]] = []
        self.blocked_attempts: list[str] = []

    @property
    def active(self) -> bool:
        return bool(self._patches)

    def _blocker(self, name: str):
        guard = self

        def blocked(*args, **kwargs):
            guard.blocked_attempts.append(name)
            log.error("Diskə şəkil/video yazma cəhdi bloklandı: %s", name)
            raise DiskWriteBlockedError(f"{name} qadağandır: şəkil və video diskə yazılmır")

        return blocked

    def _patch(self, owner: Any, attr: str, label: str) -> None:
        if hasattr(owner, attr):
            self._patches.append((owner, attr, getattr(owner, attr)))
            setattr(owner, attr, self._blocker(label))

    def install(self) -> "DiskImageGuard":
        if self.active:
            return self
        import cv2

        for attr in ("imwrite", "imwritemulti", "imwriteanimation", "VideoWriter"):
            self._patch(cv2, attr, f"cv2.{attr}")
        try:
            from PIL import Image

            self._patch(Image.Image, "save", "PIL.Image.save")
        except ImportError:
            pass
        return self

    def uninstall(self) -> None:
        for owner, attr, original in reversed(self._patches):
            setattr(owner, attr, original)
        self._patches = []

    def __enter__(self) -> "DiskImageGuard":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()


# --- razılıq --------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentToken:
    """İstifadəçinin kameranın açılmasına verdiyi razılıq."""

    granted_at: datetime
    source: str  # "ui" | "cli"
    _key: object = field(repr=False, compare=False, default=None)


_CONSENT_KEY = object()


def grant_consent(source: str) -> ConsentToken:
    """Yalnız istifadəçinin açıq hərəkətindən sonra çağırılmalıdır (UI düyməsi / CLI təsdiqi)."""
    log.info("Razılıq verildi (%s)", source)
    return ConsentToken(datetime.now().astimezone(), source, _CONSENT_KEY)


def require_consent(token: Any) -> ConsentToken:
    if not isinstance(token, ConsentToken) or token._key is not _CONSENT_KEY:
        raise ConsentError("Kamera razılıq olmadan açıla bilməz")
    return token


# --- hamısı birlikdə ------------------------------------------------------------


@dataclass
class RuntimeGuards:
    network: NetworkGuard | None
    disk: DiskImageGuard

    def uninstall(self) -> None:
        if self.network:
            self.network.uninstall()
        self.disk.uninstall()


def install_runtime_guards(block_network: bool) -> RuntimeGuards:
    """Modellər yükləndikdən SONRA çağırın."""
    network = NetworkGuard().install() if block_network else None
    return RuntimeGuards(network, DiskImageGuard().install())
