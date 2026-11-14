"""The direct lane: calls sent straight to the fastest machine while nobody is sitting at it.

PAIR assigns a call to a node as it arrives and has no setting for speed.
Its own record showed the fastest node, the developer's desktop (154 tokens
a second, against 61 and 34 for the other two), idle more than half the
night while calls waited in the queues of the slower ones. The desktop is
also somebody's computer, so it cannot simply be given all the work.

The lane has two halves.

On the desktop, a gateway (``tradingagents lane``): a small HTTP server on
the LAN that passes a request on to the machine's own Ollama only while the
lane is open, which is when the desktop has seen no keyboard or mouse for
``idle_minutes`` or its screen is locked. Otherwise it answers 503. It
reads the desktop's own idle time (GNOME's idle monitor), so the machine it
runs on decides, and a nap or an evening out opens the lane with no
schedule to keep. A call in flight when its owner comes back is left to
finish, a matter of seconds; nothing new is taken. Only the addresses in
``allow`` are served.

On the worker's machine, every model call first asks for the lane
(:func:`claim`): if it is configured (``ollama_lane_url``), a place in it is
free (one call at a time by default; a file lock shared by the sweep's
processes and the worker) and the gateway says it is open, the call goes
down the lane; otherwise it goes through PAIR as it always has. A lane that
refuses a call, or cannot be reached, costs that call nothing but the
detour: it is sent through PAIR, and the lane is left alone for a while.
"""

from __future__ import annotations

import fcntl
import http.client
import json
import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The gateway, on the desktop
# ---------------------------------------------------------------------------

READ_EVERY = 3.0        # seconds a reading of the desktop's idle time is good for
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


def _gdbus(dest: str, path: str, method: str) -> str | None:
    """A session-bus call's reply as text, or None when the desktop does not answer."""
    env = {**os.environ}
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    try:
        out = subprocess.run(["gdbus", "call", "--session", "--dest", dest, "--object-path", path, "--method", method],
                             capture_output=True, text=True, timeout=5, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def idle_seconds() -> float | None:
    """Seconds since the desktop last saw a key or the mouse; None when it cannot be read."""
    reply = _gdbus("org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core",
                   "org.gnome.Mutter.IdleMonitor.GetIdletime")          # "(uint64 11156,)", in milliseconds
    try:
        return int(reply.split()[1].rstrip(",)")) / 1000 if reply else None
    except (IndexError, ValueError):
        return None


def screen_locked() -> bool:
    return (_gdbus("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver", "org.gnome.ScreenSaver.GetActive") or "").startswith("(true")


class Gate:
    """Whether the lane is open: the desktop idle for ``idle_minutes``, or its screen locked.

    A desktop that cannot be asked is taken as in use: the lane stays shut.
    """

    def __init__(self, idle_minutes: float, read_idle=idle_seconds, read_locked=screen_locked, clock=time.monotonic):
        self.need = idle_minutes * 60
        self._read_idle, self._read_locked, self._clock = read_idle, read_locked, clock
        self._at, self._state = None, {}

    def state(self) -> dict:
        now = self._clock()
        if self._at is None or now - self._at >= READ_EVERY:
            idle, locked = self._read_idle(), self._read_locked()
            if locked:
                is_open, why = True, "the screen is locked"
            elif idle is None:
                is_open, why = False, "the desktop's idle time cannot be read"
            elif idle >= self.need:
                is_open, why = True, f"no key or mouse for {idle / 60:.0f} min"
            else:
                is_open, why = False, f"in use ({idle:.0f} s since the last key or mouse; the lane opens at {self.need / 60:g} min)"
            self._state = {"open": is_open, "reason": why, "idle_seconds": idle, "locked": locked,
                           "node": socket.gethostname()}
            self._at = now
        return self._state


def make_handler(gate: Gate, upstream: str, allow: set[str]):
    target = urlparse(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):       # the journal has the refusals; a line per call is noise
            logger.debug("%s %s", self.address_string(), fmt % args)

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve(self) -> None:
            if self.path == "/lane":
                return self._json(200, gate.state())
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.client_address[0] not in allow:
                logger.warning("Refused %s: not one of the lane's callers", self.client_address[0])
                return self._json(403, {"error": "this address is not one of the lane's callers"})
            state = gate.state()
            if not state["open"]:
                return self._json(503, {"error": f"the lane is closed: {state['reason']}"})
            conn = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=900)
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
                conn.request(self.command, self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP_BY_HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                # As it comes: the caller's idle limit watches for silence, and a
                # caller that hangs up must stop the model (the write fails, the
                # connection to Ollama closes, and Ollama drops the call).
                while chunk := resp.read1(65536):
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except OSError as exc:
                logger.warning("The machine's own model server did not answer: %s", exc)
                self.close_connection = True
                try:
                    self._json(502, {"error": f"the model server behind the lane did not answer: {exc}"})
                except OSError:
                    pass
            finally:
                conn.close()

        do_GET = do_POST = do_DELETE = do_HEAD = _serve

    return Handler


def serve(port: int, upstream: str, idle_minutes: float, allow: list[str], host: str = "0.0.0.0",
          gate: Gate | None = None) -> ThreadingHTTPServer:
    """The gateway's server, ready for ``serve_forever``; loopback may always call it."""
    gate = gate or Gate(idle_minutes)
    server = ThreadingHTTPServer((host, port), make_handler(gate, upstream, {*allow, "127.0.0.1", "::1"}))
    server.daemon_threads = True
    return server


# ---------------------------------------------------------------------------
# The caller's side, on the worker's machine
# ---------------------------------------------------------------------------

ASK_EVERY = 10.0        # seconds the gateway's answer to "are you open?" is good for
SHUT_AFTER_REFUSAL = 30.0
SHUT_AFTER_FAULT = 600.0

_asked: dict[str, tuple[float, dict]] = {}
_shut_until: dict[str, float] = {}


def ask(url: str, now: float | None = None) -> dict:
    """The gateway's state (``open``, ``reason``, ``node``); closed when it cannot be asked."""
    now = time.monotonic() if now is None else now
    cached = _asked.get(url)
    if cached and now - cached[0] < ASK_EVERY:
        return cached[1]
    target = urlparse(url)
    state = {"open": False, "reason": "the lane's gateway could not be reached", "node": ""}
    try:
        conn = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=2)
        try:
            conn.request("GET", "/lane")
            resp = conn.getresponse()
            if resp.status == 200:
                state = json.loads(resp.read())
        finally:
            conn.close()
    except (OSError, ValueError):
        pass
    _asked[url] = (now, state)
    return state


def shut(url: str, seconds: float, why: str = "") -> None:
    """Leave the lane alone for ``seconds``: it refused a call, or failed one."""
    _shut_until[url] = time.monotonic() + seconds
    _asked.pop(url, None)
    if why:
        logger.info("The lane at %s is left alone for %d s: %s", url, seconds, why)


@contextmanager
def claim(config: dict):
    """A place in the lane for one call: ``(url, node)`` while it is held, or None when the call goes through PAIR.

    None when no lane is configured, it was shut a moment ago, every place is
    taken (by this process or another: the lock is a file), or the gateway
    says the desktop is in use.
    """
    url = (config.get("ollama_lane_url") or "").strip().rstrip("/")
    if not url or time.monotonic() < _shut_until.get(url, 0.0):
        yield None
        return
    folder = Path(config.get("data_cache_dir") or ".") / "lane"
    held = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for slot in range(max(1, int(config.get("ollama_lane_slots") or 1))):
            fh = (folder / f"slot{slot}.lock").open("w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = fh
                break
            except OSError:
                fh.close()
    except OSError as exc:
        logger.debug("No lane lock: %s", exc)
    if held is None:
        yield None
        return
    try:
        state = ask(url)
        yield (url, f"{state.get('node') or urlparse(url).hostname} (lane)") if state.get("open") else None
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
