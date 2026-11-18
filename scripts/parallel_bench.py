"""How much more a node writes when Ollama serves several calls at once.

Sends the same analyst-sized call (about 6k tokens in, up to 1,300 out) to this
machine's plain Ollama, first one at a time, then N at once, and prints each
level's total output rate. Run it after restarting Ollama with
OLLAMA_NUM_PARALLEL set; with the default of 1, the calls simply queue and the
total stays flat.

    .venv/bin/python scripts/parallel_bench.py --levels 1 2 3

Live sweep calls reaching the same GPU through PAIR make the numbers
conservative, not inflated.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.request

FILLER = (
    "Quarterly revenue rose on stronger services demand while hardware margins narrowed; "
    "management reiterated full-year guidance and flagged currency headwinds in Europe. "
)


def prompt(seed: int) -> str:
    random.seed(seed)
    body = "".join(f"[{i}] {FILLER}" for i in range(260))   # about 6k tokens
    return (
        f"Report #{seed}-{random.random():.6f}. Read the notes below and write a detailed analyst "
        "report of at least 900 words on the company's outlook, risks and a trading view.\n\n" + body
    )


def call(url: str, model: str, seed: int, num_ctx: int, max_out: int) -> dict:
    payload = {
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt(seed)}],
        "options": {"num_ctx": num_ctx, "num_predict": max_out, "temperature": 0.7},
    }
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        r = json.load(resp)
    return {"out": r.get("eval_count", 0), "in": r.get("prompt_eval_count", 0),
            "gen_s": r.get("eval_duration", 0) / 1e9, "wall": time.monotonic() - t0}


def level(url: str, model: str, n: int, rounds: int, num_ctx: int, max_out: int) -> None:
    seeds = [random.randrange(10**9) for _ in range(n * rounds)]
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(n) as pool:
        results = list(pool.map(lambda s: call(url, model, s, num_ctx, max_out), seeds))
    wall = time.monotonic() - t0
    out = sum(r["out"] for r in results)
    per_call = sum(r["out"] / r["gen_s"] for r in results if r["gen_s"]) / len(results)
    print(f"{n} at once: {len(results)} calls, {out} tokens out in {wall:.0f} s -> "
          f"{out / wall:.0f} tok/s total, {per_call:.0f} tok/s per call, "
          f"{sum(r['wall'] for r in results) / len(results):.0f} s per call", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--rounds", type=int, default=2, help="calls per slot at each level")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--max-out", type=int, default=1300)
    a = ap.parse_args()
    call(a.url, a.model, 0, a.num_ctx, 16)   # load the model at this context first
    for n in a.levels:
        level(a.url, a.model, n, a.rounds, a.num_ctx, a.max_out)


if __name__ == "__main__":
    main()
