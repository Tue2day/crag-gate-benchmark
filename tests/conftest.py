import socket
import threading
import httpx
import pytest

from jev_gate.config import Config


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original_pair, original_connect = socket.socketpair, socket.socket.connect
    state = threading.local()

    def local_socketpair(*args, **kwargs):
        # Windows implements asyncio's self-pipe with a localhost TCP socketpair.
        # Permit only that stdlib operation, on its calling thread, not arbitrary localhost IO.
        state.creating_pair = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            state.creating_pair = False

    def blocked(sock, address, *args, **kwargs):
        if (getattr(state, "creating_pair", False) and isinstance(address, tuple)
                and address[0] in {"127.0.0.1", "::1"}):
            return original_connect(sock, address, *args, **kwargs)
        raise AssertionError("Tests must not open external or arbitrary localhost connections")

    def blocked_http(*args, **kwargs):
        raise AssertionError("Tests must use MockTransport, never a real HTTP transport")

    async def blocked_async_http(*args, **kwargs):
        blocked_http()

    monkeypatch.setattr(socket, "socketpair", local_socketpair)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked_http)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async_http)


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=str(tmp_path / "data"), run_dir=str(tmp_path / "runs"),
                  bootstrap_samples=100, requests_per_second=100000,
                  development_size=2, validation_size=2, test_size=3,
                  challenge_size=2, challenge_pool_size=20)
