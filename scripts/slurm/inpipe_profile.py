"""Phase profiler for the cell(+in-pipeline eval) phase (ONEJOB_PHASE_PROFILE / ONEJOB_INPIPE_EVAL, 10-09).

Samples every INPIPE_PROFILE_SECONDS (15): per-card GPU util / memory (nvidia-smi), per-server vLLM
running / waiting / KV usage (/metrics), cell progress (callback), eval worker stage (status json).
stop() writes run_dir/evals/phase_profile_<tag>.json and returns a one-line summary."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def _gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        rows = {}
        for line in out.strip().splitlines():
            i, u, m, t = [x.strip() for x in line.split(",")]
            rows[int(i)] = (float(u), float(m), float(t))
        return rows
    except Exception:  # noqa: BLE001
        return {}


def _vllm(url):
    try:
        text = urllib.request.urlopen(f"{url}/metrics", timeout=3).read().decode()
    except Exception:  # noqa: BLE001
        return None
    vals = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, name in (("vllm:num_requests_running", "run"), ("vllm:num_requests_waiting{", "wait"),
                          ("vllm:kv_cache_usage_perc", "kv")):
            if line.startswith(key) and name not in vals:
                try:
                    vals[name] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return vals


class PhaseProfiler:
    def __init__(self, out: Path, tag: str, cards_phys: list[int], server_urls: list[str],
                 progress=lambda: {}, status_files=()):
        self.out, self.tag = out, tag
        self.cards, self.urls = cards_phys, server_urls
        self.progress, self.status_files = progress, list(status_files)
        self.samples, self.events = [], []
        self.t0 = time.time()
        self.period = float(os.environ.get("INPIPE_PROFILE_SECONDS", "15"))
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def event(self, name: str):
        self.events.append((round(time.time() - self.t0, 1), name))

    def add_status(self, path):
        self.status_files.append(Path(path))

    def _run(self):
        while not self._stop.is_set():
            g = _gpus()
            s = {"t": round(time.time() - self.t0, 1),
                 "gpu": {c: g.get(c) for c in self.cards},
                 "srv": {u: _vllm(u) for u in self.urls},
                 "cells": self.progress()}
            stages = {}
            for p in self.status_files:
                try:
                    d = json.loads(Path(p).read_text())
                    stages[d.get("label", str(p))] = d.get("stage")
                except Exception:  # noqa: BLE001
                    pass
            s["eval"] = stages
            self.samples.append(s)
            self._stop.wait(self.period)

    def stop(self, extra: dict | None = None) -> str:
        self._stop.set()
        self.thread.join(timeout=30)
        dur = time.time() - self.t0
        util, mem, idle = {}, {}, {}
        for c in self.cards:
            us = [s["gpu"][c][0] for s in self.samples if s["gpu"].get(c)]
            ms = [s["gpu"][c][1] / s["gpu"][c][2] for s in self.samples if s["gpu"].get(c)]
            if us:
                util[c] = round(sum(us) / len(us), 1)
                idle[c] = round(100 * sum(1 for u in us if u < 10) / len(us), 1)
                mem[c] = round(100 * sum(ms) / len(ms), 1)
        summary = {"tag": self.tag, "duration_s": round(dur, 1), "n_samples": len(self.samples),
                   "gpu_util_mean_pct": util, "gpu_idle_pct": idle, "gpu_mem_ratio_pct": mem,
                   "events": self.events, **(extra or {})}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / f"phase_profile_{self.tag}.json").write_text(
            json.dumps({"summary": summary, "samples": self.samples}, default=str))
        mu = sum(util.values()) / max(1, len(util))
        mi = sum(idle.values()) / max(1, len(idle))
        return (f"phase {self.tag}: {dur / 60:.1f} min, mean GPU util {mu:.0f}%, idle {mi:.0f}% "
                f"(per card util {util}, mem% {mem})")
