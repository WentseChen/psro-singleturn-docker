"""Data-parallel, continuously batched vLLM generation for payoff cells.

One engine process per GPU. The parent submits single requests and collects
them as they finish, so a finished sequence is replaced immediately (vLLM's
continuous batching) and no request ever waits for the slowest member of a
fixed batch. Every request carries its own seed, so a completion does not
depend on which engine ran it or what it was batched with.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any

_STOP = "__stop__"


def _die_with_parent() -> None:
    """SIGKILL this engine if the cell worker dies, even by SIGKILL, so an
    orphaned engine can never keep a GPU."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _engine_main(gpu: str, util: float, config: dict[str, Any], inbox, outbox) -> None:
    _die_with_parent()
    parent = os.getppid()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        if config.get("prepare"):
            import importlib

            module_name, _, attr = config["prepare"].partition(":")
            getattr(importlib.import_module(module_name), attr)()
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import RequestOutputKind

        loras = {
            role: LoRARequest(f"payoff_{role}", slot, path)
            for slot, (role, path) in enumerate(sorted(config["loras"].items()), start=1)
            if path
        }
        llm = LLM(
            model=config["model"],
            tokenizer=config["model"],
            trust_remote_code=True,
            dtype=config.get("dtype", "bfloat16"),
            tensor_parallel_size=1,
            gpu_memory_utilization=util,
            max_model_len=config["max_model_len"],
            max_num_seqs=config["max_num_seqs"],
            enable_lora=bool(loras),
            max_loras=2,
            max_lora_rank=64,
            enforce_eager=config["enforce_eager"],
        )
        engine = llm.llm_engine
    except BaseException as exc:  # report, then die: the parent must not hang
        outbox.put(("error", gpu, f"{type(exc).__name__}: {exc}"))
        raise
    outbox.put(("ready", gpu, None))

    def add(item: dict[str, Any]) -> None:
        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=item["max_new_tokens"],
            min_tokens=1,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            seed=item["seed"],
            truncate_prompt_tokens=item["prompt_max_tokens"],
        )
        params.output_kind = RequestOutputKind.FINAL_ONLY
        engine.add_request(
            item["key"],
            TokensPrompt(prompt_token_ids=item["token_ids"]),
            params,
            lora_request=loras.get(item["role"]),
        )

    stopping = False
    while True:
        busy = engine.has_unfinished_requests()
        try:
            # Block only when idle; otherwise drain whatever has arrived.
            if busy or stopping:
                item = inbox.get_nowait()
            else:
                while True:
                    try:
                        item = inbox.get(timeout=5)
                        break
                    except queue.Empty:
                        if os.getppid() != parent:  # orphaned: exit, free the card
                            return
            while True:
                if item == _STOP:
                    stopping = True
                else:
                    add(item)
                item = inbox.get_nowait()
        except queue.Empty:
            pass
        if stopping and not engine.has_unfinished_requests():
            break
        if not engine.has_unfinished_requests():
            continue
        finished = []
        for output in engine.step():
            if not output.finished:
                continue
            candidate = output.outputs[0]
            finished.append(
                (
                    output.request_id,
                    list(candidate.token_ids or []),
                    candidate.text,
                    candidate.finish_reason,
                    candidate.stop_reason,
                )
            )
        if finished:
            outbox.put(("done", gpu, finished))
    outbox.put(("closed", gpu, None))


def _free_fraction(gpu: str) -> float:
    """Free memory fraction of one card, probed in a throwaway process so the
    parent never initialises CUDA."""
    import subprocess
    import sys

    probe = (
        "import torch; f, t = torch.cuda.mem_get_info(0); print(f / t)"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=300
    )
    if out.returncode != 0:
        return 0.0
    return float(out.stdout.strip().splitlines()[-1])


def _util_is_total_cap() -> bool:
    from importlib.metadata import version

    major, minor = (int(part) for part in version("vllm").split(".")[:2])
    return (major, minor) < (0, 9)


class EnginePool:
    """Round-robin-by-load dispatch over one vLLM engine per GPU."""

    def __init__(
        self,
        gpus: list[str],
        *,
        model: str,
        loras: dict[str, str | None],
        max_model_len: int,
        max_num_seqs: int,
        enforce_eager: bool,
        util_by_gpu: dict[str, float],
        min_util: float = 0.30,
        headroom: float = 0.05,
        startup_timeout: float = 1800.0,
        prepare: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        ctx = mp.get_context("spawn")
        import atexit

        # Engines are non-daemon (vLLM may fork helpers). Kill them on any exit
        # of the parent, including an exception, so a dead cell never strands
        # a GPU; atexit is LIFO, so this runs before multiprocessing joins.
        atexit.register(self.kill)
        self._outbox = ctx.Queue()
        self._inboxes: dict[str, Any] = {}
        self._procs: dict[str, Any] = {}
        self._load: dict[str, int] = {}
        self._owner: dict[str, str] = {}
        self.max_num_seqs = max_num_seqs
        config = {
            "model": model,
            "loras": loras,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "prepare": prepare,
            "dtype": dtype,
        }
        total_cap = _util_is_total_cap()
        for gpu in gpus:
            wanted = util_by_gpu.get(gpu, 0.90)
            free = _free_fraction(gpu)
            # A card shared with a judge only has what the judge left free.
            # vLLM < 0.9 reads gpu_memory_utilization as a cap on the whole
            # card (other processes included); newer releases read it as this
            # engine's own budget and refuse to start if it is not free.
            if total_cap:
                util = min(wanted, 1.0 - headroom)
                usable = util - (1.0 - free)
            else:
                util = usable = min(wanted, free - headroom)
            if usable < min_util:
                print(
                    f"engine pool: skip GPU {gpu} (usable fraction {usable:.2f})",
                    flush=True,
                )
                continue
            inbox = ctx.Queue()
            proc = ctx.Process(
                target=_engine_main, args=(gpu, round(util, 3), config, inbox, self._outbox)
            )
            # Set the card before the child exists: spawn re-imports the
            # parent's main module, and anything there that touches CUDA must
            # already see only this GPU.
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            # The engine's target lives in this module, so the child has no
            # use for the parent's __main__; hide it so spawn does not re-run
            # the caller's script (or fail on `python -` / `-c`).
            main = sys.modules["__main__"]
            hidden = {k: main.__dict__.pop(k) for k in ("__file__",) if k in main.__dict__}
            spec, main.__spec__ = getattr(main, "__spec__", None), None
            try:
                proc.start()
            finally:
                main.__dict__.update(hidden)
                main.__spec__ = spec
                if previous is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous
            self._inboxes[gpu], self._procs[gpu], self._load[gpu] = inbox, proc, 0
            print(
                f"engine pool: GPU {gpu} util {util:.2f} (usable {usable:.2f})",
                flush=True,
            )
        if not self._procs:
            raise RuntimeError(f"engine pool: no usable GPU among {gpus}")
        pending = set(self._procs)
        deadline = time.time() + startup_timeout
        failed: dict[str, str] = {}
        while pending and time.time() < deadline:
            try:
                kind, gpu, payload = self._outbox.get(timeout=10)
            except queue.Empty:
                for gpu in list(pending):
                    if not self._procs[gpu].is_alive():
                        failed[gpu] = f"exit {self._procs[gpu].exitcode}"
                        pending.discard(gpu)
                continue
            if kind == "ready":
                pending.discard(gpu)
            elif kind == "error":
                failed[gpu] = payload
                pending.discard(gpu)
        for gpu in list(pending) + list(failed):
            self._drop(gpu)
        if failed or pending:
            print(f"engine pool: engines failed {failed} timed out {sorted(pending)}", flush=True)
        if not self._procs:
            raise RuntimeError(f"engine pool: every engine failed: {failed}")
        print(f"engine pool ready on GPUs {sorted(self._procs)}", flush=True)

    def _drop(self, gpu: str) -> None:
        proc = self._procs.pop(gpu, None)
        self._inboxes.pop(gpu, None)
        self._load.pop(gpu, None)
        if proc is not None and proc.is_alive():
            proc.kill()
            proc.join(30)

    @property
    def size(self) -> int:
        return len(self._procs)

    @property
    def in_flight(self) -> int:
        return sum(self._load.values())

    def capacity(self) -> int:
        """Requests to keep outstanding: a full batch per engine plus a small
        queue so a finished slot is refilled without a round trip."""
        return len(self._procs) * (self.max_num_seqs + self.max_num_seqs // 4)

    def submit(self, key: str, *, role: str, token_ids: list[int], seed: int,
               max_new_tokens: int, prompt_max_tokens: int) -> None:
        gpu = min(self._load, key=self._load.get)
        self._inboxes[gpu].put(
            {
                "key": key, "role": role, "token_ids": token_ids, "seed": seed,
                "max_new_tokens": max_new_tokens, "prompt_max_tokens": prompt_max_tokens,
            }
        )
        self._load[gpu] += 1
        self._owner[key] = gpu

    def poll(self, timeout: float = 1.0) -> list[tuple[str, list[int], str, Any, Any]]:
        results = []
        try:
            item = self._outbox.get(timeout=timeout)
            while True:
                kind, gpu, payload = item
                if kind == "done":
                    for row in payload:
                        self._load[self._owner.pop(row[0])] -= 1
                    results.extend(payload)
                elif kind == "error":
                    raise RuntimeError(f"engine on GPU {gpu} failed: {payload}")
                item = self._outbox.get_nowait()
        except queue.Empty:
            pass
        if not results:
            dead = [g for g, p in self._procs.items() if not p.is_alive()]
            if dead:
                raise RuntimeError(f"engine process died on GPU(s) {dead}")
        return results

    def close(self) -> None:
        for inbox in self._inboxes.values():
            inbox.put(_STOP)
        for gpu, proc in list(self._procs.items()):
            proc.join(120)
            if proc.is_alive():
                proc.kill()
                proc.join(30)
        self._procs.clear()

    def kill(self) -> None:
        for gpu in list(getattr(self, "_procs", {})):
            self._drop(gpu)
