"""The direct lane: calls go straight to the fast desktop while nobody is at it, and through PAIR otherwise."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tradingagents import lane


@pytest.fixture(autouse=True)
def _loopback(monkeypatch):
    """The suite refuses the network; the lane's tests talk to servers of their own on loopback."""
    monkeypatch.undo()
    lane._asked.clear()
    lane._shut_until.clear()
    yield
    lane._asked.clear()
    lane._shut_until.clear()


def _gate(idle, locked=False, minutes=10.0):
    clock = iter(range(0, 10_000, 5))
    return lane.Gate(minutes, read_idle=lambda: idle() if callable(idle) else idle, read_locked=lambda: locked, clock=lambda: next(clock))


@pytest.mark.unit
def test_the_lane_opens_after_ten_idle_minutes_or_a_locked_screen_and_shuts_at_the_first_key():
    assert _gate(599.0).state()["open"] is False
    opened = _gate(600.0).state()
    assert opened["open"] is True and opened["reason"] == "no key or mouse for 10 min"
    assert _gate(3.0, locked=True).state()["reason"] == "the screen is locked"         # locked a moment ago: away
    assert _gate(None).state() == {**_gate(None).state(), "open": False, "reason": "the desktop's idle time cannot be read"}

    idle = iter([1800.0, 0.4])                                                         # away half an hour; then a key
    gate = _gate(lambda: next(idle))
    assert gate.state()["open"] is True
    back = gate.state()
    assert back["open"] is False and "in use (0 s since the last key or mouse" in back["reason"]


class _Ollama(BaseHTTPRequestHandler):
    """Stands for the desktop's own Ollama: answers a chat in two streamed lines."""
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Ollama.seen.append((self.path, json.loads(body)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for line in (b'{"message":{"content":"Hel"},"done":false}\n', b'{"message":{"content":"lo"},"done":true}\n'):
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def gateway():
    """A gateway on loopback in front of a stand-in Ollama; ``state['idle']`` is the desktop's idle time."""
    _Ollama.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Ollama)
    state = {"idle": 3600.0}
    ticks = iter(range(0, 10**9, 5))
    gate = lane.Gate(10.0, read_idle=lambda: state["idle"], read_locked=lambda: False, clock=lambda: next(ticks))
    server = lane.serve(0, f"http://127.0.0.1:{upstream.server_address[1]}", 10.0, [], host="127.0.0.1", gate=gate)
    for s in (upstream, server):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", state
    server.shutdown()
    upstream.shutdown()


def _post(url, path="/api/chat", body=None):
    conn = http.client.HTTPConnection(url.split("//")[1], timeout=5)
    conn.request("POST", path, body=json.dumps(body or {"model": "m", "messages": []}), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, resp.read().decode()


@pytest.mark.unit
def test_the_gateway_passes_a_call_to_the_desktops_ollama_while_open_and_refuses_it_in_use(gateway):
    url, state = gateway
    status, text = _post(url, body={"model": "qwen3.8:27b", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and [json.loads(line)["message"]["content"] for line in text.splitlines()] == ["Hel", "lo"]
    assert _Ollama.seen == [("/api/chat", {"model": "qwen3.8:27b", "messages": [{"role": "user", "content": "hi"}]})]

    state["idle"] = 2.0                                                               # somebody sat down
    status, text = _post(url)
    assert status == 503 and "the lane is closed: in use" in json.loads(text)["error"] and len(_Ollama.seen) == 1
    assert lane.ask(url)["open"] is False


@pytest.mark.unit
def test_only_the_lanes_callers_are_served():
    gate = lane.Gate(10.0, read_idle=lambda: 3600.0, read_locked=lambda: False)
    handler = lane.make_handler(gate, "http://127.0.0.1:9", allow={"192.168.0.98"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)                           # loopback is not on this list
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, text = _post(f"http://127.0.0.1:{server.server_address[1]}")
        assert status == 403 and "not one of the lane's callers" in text
    finally:
        server.shutdown()


@pytest.mark.unit
def test_a_call_claims_the_lane_only_when_it_is_configured_free_and_open(gateway, tmp_path):
    url, state = gateway
    cfg = {"ollama_lane_url": url, "data_cache_dir": str(tmp_path), "ollama_lane_slots": 1}
    with lane.claim({"data_cache_dir": str(tmp_path)}) as place:
        assert place is None                                                          # no lane configured
    with lane.claim(cfg) as place:
        assert place[0] == url and place[1].endswith("(lane)")
        with lane.claim(cfg) as second:
            assert second is None                                                     # one call at a time: the place is taken
    with lane.claim({**cfg, "ollama_lane_slots": 2}) as a, lane.claim({**cfg, "ollama_lane_slots": 2}) as b:
        assert a and b
    lane.shut(url, 30)
    with lane.claim(cfg) as place:
        assert place is None                                                          # it refused a call a moment ago
    lane._shut_until.clear()
    state["idle"] = 1.0
    lane._asked.clear()
    with lane.claim(cfg) as place:
        assert place is None                                                          # the desktop is in use
    with lane.claim({**cfg, "ollama_lane_url": "http://127.0.0.1:9"}) as place:
        assert place is None                                                          # no gateway there


class _Reply:
    usage_metadata = {"input_tokens": 5, "output_tokens": 7}
    response_metadata = {"total_duration": 2e9, "eval_duration": 1e9}
    content = "ok"


@pytest.mark.unit
def test_the_model_client_sends_a_call_down_an_open_lane_and_through_pair_when_it_is_refused(gateway, tmp_path, monkeypatch):
    from langchain_ollama import ChatOllama

    from tradingagents.dataflows.config import run_config
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import ollama_client as oc

    url, state = gateway
    sent, recorded = [], []

    class Refused(Exception):
        status_code = 503

    def fake_invoke(self, input, config=None, **kwargs):
        sent.append(self.base_url)
        if self.base_url == url and state.get("refuse"):
            raise Refused("the lane is closed")
        return _Reply()

    monkeypatch.setattr(ChatOllama, "invoke", fake_invoke)
    monkeypatch.setattr(oc.usage, "record_response", lambda response, model, config, started, ended, node="": recorded.append(node))
    monkeypatch.setattr(oc, "normalize_content", lambda r: r)
    llm = oc.NormalizedChatOllama(model="qwen3.8:27b", base_url="http://127.0.0.1:11434")
    with run_config({**DEFAULT_CONFIG, "ollama_lane_url": url, "data_cache_dir": str(tmp_path)}):
        llm.invoke("hi")
        assert sent == [url] and recorded[0].endswith("(lane)")                      # down the lane, and recorded as such

        state["refuse"] = True                                                        # its owner came back as the call arrived
        llm.invoke("hi")
        assert sent[1:] == [url, "http://127.0.0.1:11434"] and recorded[1] == ""     # the same call, through PAIR
        llm.invoke("hi")
        assert sent[3:] == ["http://127.0.0.1:11434"]                                 # and the lane is left alone for a while

    with run_config({**DEFAULT_CONFIG, "data_cache_dir": str(tmp_path)}):             # no lane: nothing changes
        llm.invoke("hi")
        assert sent[-1] == "http://127.0.0.1:11434"
