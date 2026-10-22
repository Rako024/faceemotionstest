import socket
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from src.privacy import (
    ConsentError,
    ConsentToken,
    DiskImageGuard,
    DiskWriteBlockedError,
    NetworkBlockedError,
    NetworkGuard,
    grant_consent,
    install_runtime_guards,
    require_consent,
)


@pytest.fixture
def net_guard():
    g = NetworkGuard().install()
    yield g
    g.uninstall()


@pytest.fixture
def disk_guard():
    g = DiskImageGuard().install()
    yield g
    g.uninstall()


# --- şəbəkə ---------------------------------------------------------------------


def test_http_request_blocked(net_guard):
    with pytest.raises(urllib.error.URLError) as exc:
        urllib.request.urlopen("https://example.com", timeout=2)
    assert isinstance(exc.value.reason, NetworkBlockedError)
    assert net_guard.blocked_attempts


def test_dns_lookup_blocked(net_guard):
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("github.com", 443)


def test_raw_connect_to_public_ip_blocked(net_guard):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            s.connect(("1.1.1.1", 443))
        with pytest.raises(NetworkBlockedError):
            s.connect_ex(("8.8.8.8", 53))
    finally:
        s.close()


def test_udp_sendto_blocked(net_guard):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkBlockedError):
            s.sendto(b"x", ("8.8.8.8", 53))
    finally:
        s.close()


def test_loopback_still_works(net_guard):
    a, b = socket.socketpair()  # Windows-da loopback connect istifadə edir
    try:
        a.sendall(b"ping")
        assert b.recv(4) == b"ping"
    finally:
        a.close()
        b.close()


def test_uninstall_restores(net_guard):
    net_guard.uninstall()
    assert socket.getaddrinfo.__name__ == "getaddrinfo"
    assert socket.socket.connect.__qualname__ != "NetworkGuard.install.<locals>.connect"


# --- disk -----------------------------------------------------------------------


def test_imwrite_blocked(disk_guard, tmp_path):
    img = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(DiskWriteBlockedError):
        cv2.imwrite(str(tmp_path / "frame.jpg"), img)
    assert list(tmp_path.iterdir()) == []


def test_video_writer_blocked(disk_guard, tmp_path):
    with pytest.raises(DiskWriteBlockedError):
        cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (10, 10))
    assert list(tmp_path.iterdir()) == []


def test_pil_save_blocked(disk_guard, tmp_path):
    from PIL import Image

    with pytest.raises(DiskWriteBlockedError):
        Image.new("RGB", (4, 4)).save(tmp_path / "x.png")


def test_in_memory_encoding_still_allowed(disk_guard):
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok and len(buf) > 0


def test_disk_guard_uninstall(disk_guard, tmp_path):
    disk_guard.uninstall()
    assert cv2.imwrite(str(tmp_path / "ok.png"), np.zeros((2, 2, 3), np.uint8))


def test_install_runtime_guards_respects_config():
    guards = install_runtime_guards(block_network=False)
    try:
        assert guards.network is None
        assert guards.disk.active
    finally:
        guards.uninstall()
    guards = install_runtime_guards(block_network=True)
    try:
        assert guards.network is not None and guards.network.active
    finally:
        guards.uninstall()


# --- razılıq --------------------------------------------------------------------


def test_consent_token_from_grant():
    token = grant_consent("ui")
    assert require_consent(token) is token
    assert token.source == "ui"


@pytest.mark.parametrize("bad", [None, True, "yes", object()])
def test_consent_required(bad):
    with pytest.raises(ConsentError):
        require_consent(bad)


def test_forged_token_rejected():
    from datetime import datetime

    with pytest.raises(ConsentError):
        require_consent(ConsentToken(datetime.now(), "forged"))
