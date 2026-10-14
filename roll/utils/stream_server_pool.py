"""Payoff-cell generation on long-lived vLLM servers (CELL_SERVERS=url,url).

Drop-in for EnginePool (same submit / poll / in_flight / capacity / size /
close), but instead of spawning one engine per GPU for every cell it sends
requests to OpenAI-compatible vLLM servers that stay up for the whole PSRO
job (any vLLM version with /v1/load_lora_adapter and return_token_ids).

* No engine start-up per cell; all cards serve every cell.
* Several cells run concurrently against the same servers, so one cell's
  tail overlaps another's bulk and the servers stay saturated.
* Each cell's attacker/defender adapter is loaded on every server under a
  name derived from its path, so concurrent cells share adapters and a
  later cell reuses what an earlier one loaded.

Sampling matches EnginePool's engine exactly: temperature 1, top_p 1,
top_k -1, min_tokens 1, skip_special_tokens False,
include_stop_str_in_output True, per-request seed, prompt truncation.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import queue
import threading
from typing import Any


def adapter_name(path: str | None) -> str | None:
    if not path:
        return None
    return "cell_" + hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]


class ServerPool:
    def __init__(self, urls: list[str], *, loras: dict[str, str | None],
                 base_name: str = "base", max_inflight_per_server: int = 512) -> None:
        import aiohttp  # noqa: F401  (fail early if missing)

        self.urls = [u.rstrip("/") for u in urls if u.strip()]
        if not self.urls:
            raise RuntimeError("server pool: no CELL_SERVERS")
        self.base_name = base_name
        self.models = {role: (adapter_name(path) or base_name) for role, path in loras.items()}
        self.max_num_seqs = max_inflight_per_server
        self._load = [0] * len(self.urls)
        self._owner: dict[str, int] = {}
        self._done: "queue.Queue[tuple]" = queue.Queue()
        self._error: str | None = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._setup(), self.loop).result(timeout=60)
        for role, path in loras.items():
            if path:
                self._load_adapter(self.models[role], path)
        print(f"server pool ready on {len(self.urls)} server(s); adapters {self.models}", flush=True)

    async def _setup(self) -> None:
        import aiohttp

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=4),
        )

    def _load_adapter(self, name: str, path: str) -> None:
        if os.environ.get("ONEJOB_SHARED_BASE", "0") == "1":  # concat(pi0, adapter), see shared_base_lora
            from roll.utils.shared_base_lora import merged_path
            path = merged_path(path)
        async def go():
            for url in self.urls:
                async with self.session.post(f"{url}/v1/load_lora_adapter",
                                             json={"lora_name": name, "lora_path": path}) as resp:
                    text = await resp.text()
                    if resp.status != 200 and "already" not in text:
                        raise RuntimeError(f"{url}: load {name} failed HTTP {resp.status}: {text[:300]}")

        asyncio.run_coroutine_threadsafe(go(), self.loop).result(timeout=600)

    # -- EnginePool interface ------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.urls)

    @property
    def in_flight(self) -> int:
        return sum(self._load)

    def capacity(self) -> int:
        return len(self.urls) * (self.max_num_seqs + self.max_num_seqs // 4)

    def submit(self, key: str, *, role: str, token_ids: list[int], seed: int,
               max_new_tokens: int, prompt_max_tokens: int) -> None:
        if self._error:
            raise RuntimeError(f"server pool: {self._error}")
        server = min(range(len(self.urls)), key=self._load.__getitem__)
        self._load[server] += 1
        self._owner[key] = server
        ids = list(token_ids)[-prompt_max_tokens:] if prompt_max_tokens else list(token_ids)
        payload = {
            "model": self.models.get(role, self.base_name),
            "prompt": ids,
            "max_tokens": max_new_tokens,
            "min_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": int(seed) % (2**63),
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
            "return_token_ids": True,
        }
        if os.environ.get("FORMAT_GRAMMAR") == "1":
            # same constrained policy and budget as training rollouts (see format_grammar)
            from roll.utils.format_grammar import rescue_reserve, structured_outputs
            payload["structured_outputs"] = structured_outputs()
            payload["max_tokens"] = max(1, payload["max_tokens"] - rescue_reserve())
        asyncio.run_coroutine_threadsafe(self._one(server, key, payload), self.loop)

    async def _one(self, server: int, key: str, payload: dict) -> None:
        import aiohttp

        try:
            for attempt in range(8):
                try:
                    async with self.session.post(f"{self.urls[server]}/v1/completions", json=payload) as resp:
                        body = await resp.json(content_type=None)
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}: {str(body)[:300]}")
                    break
                except (asyncio.TimeoutError, OSError, aiohttp.ClientError):
                    if attempt == 7:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))
            choice = body["choices"][0]
            from roll.utils.format_grammar import needs_completion
            if os.environ.get("FORMAT_GRAMMAR") == "1" and needs_completion(
                    choice.get("text", ""), choice.get("finish_reason")):
                from roll.utils.format_grammar import budget_force
                ids, text, _ = await budget_force(self.session, self.urls[server], payload, choice, False)
                choice = dict(choice, token_ids=ids, text=text, finish_reason="stop",
                              stop_reason="budget_forced")
            self._done.put((key, list(choice.get("token_ids") or []), choice.get("text", ""),
                            choice.get("finish_reason"), choice.get("stop_reason")))
        except BaseException as exc:
            self._error = f"{key}: {type(exc).__name__}: {exc}"
            self._done.put(None)

    def poll(self, timeout: float = 1.0) -> list[tuple[str, list[int], str, Any, Any]]:
        results = []
        try:
            item = self._done.get(timeout=timeout)
            while True:
                if item is None:
                    raise RuntimeError(f"server pool: {self._error}")
                self._load[self._owner.pop(item[0])] -= 1
                results.append(item)
                item = self._done.get_nowait()
        except queue.Empty:
            pass
        return results

    def close(self) -> None:
        # Adapters stay loaded for later cells; only the client goes away.
        async def shut():
            await self.session.close()

        try:
            asyncio.run_coroutine_threadsafe(shut(), self.loop).result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)

    kill = close
