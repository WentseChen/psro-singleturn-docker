"""Long-lived inference services for a one-job PSRO run (ONEJOB=1).

Started once by the driver and kept up for the whole run:

* WildGuard on cards 0,1 (two replicas behind one URL, --num_gpus 2);
* the local topic judge on card 3 (CUDA graphs, 256 sequences);
* one vLLM 0.30 server per card, with sleep mode, serving the base model
  plus runtime-loaded LoRAs (payoff cells and the training rollout share them).

Role switching is by vLLM sleep/wake (level 1: weights to CPU, KV freed,
~6 s sleep / ~2 s wake measured on H200), so a card moves between
inference and training without restarting anything.

Memory note: vLLM's gpu_memory_utilization counts every process on the
card, so a server's share is what is left after the judges on its card.
The object also stands in for cold_psro_local.Judge (url / start / stop):
cells call start() and get the persistent URL; stop() is a no-op.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict | None = None, timeout: float = 600) -> str:
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _ok(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def phys_cards(cards) -> str:
    """Logical card list (e.g. "0,1" or 2) -> physical CUDA indices.
    ONEJOB_CARD_MAP="4,5,6,7" lets two 4-card runs share one 8-GPU node;
    unset keeps logical == physical."""
    cmap = os.environ.get("ONEJOB_CARD_MAP", "").strip()
    items = [c.strip() for c in str(cards).split(",") if c.strip() != ""]
    if not cmap:
        return ",".join(items)
    table = [c.strip() for c in cmap.split(",")]
    return ",".join(table[int(c)] for c in items)


class OneJobServices:
    def __init__(self, cfg, log) -> None:
        self.cfg = cfg
        self.log = log
        self.gpus = int(cfg.gpus)
        base_port = int(os.environ.get("ONEJOB_PORT_BASE", "7700"))
        self.judge_port = base_port
        self.topic_port = base_port + 10
        self.server_ports = [base_port + 20 + c for c in range(self.gpus)]
        self.url = f"http://127.0.0.1:{self.judge_port}/classify"
        self.server_urls = [f"http://127.0.0.1:{p}" for p in self.server_ports]
        self.renv = os.environ.get("ROLLOUT_ENV", "/scratch/wentsec/envs/vllm30")
        self.model = os.environ.get("BASE_MODEL_PATH") or cfg.base_model
        self.judge_gpus = os.environ.get("ONEJOB_JUDGE_GPUS", "0,1")
        self.topic_gpu = os.environ.get("ONEJOB_TOPIC_GPU", str(self.gpus - 1))
        utils = os.environ.get("ONEJOB_SERVER_UTILS", "0.80,0.80,0.90,0.70").split(",")
        self.server_utils = [float(u) for u in utils][: self.gpus]
        self.procs: list[subprocess.Popen] = []
        self.logdir = Path(cfg.run_root) / "logs" / "services"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.asleep: set[int] = set()

    # -- Judge interface ------------------------------------------------------
    def start(self) -> str:
        return self.url

    def stop(self) -> None:  # the judge outlives every phase
        return None

    # -- lifecycle ------------------------------------------------------------
    def _spawn(self, cmd, env, name):
        handle = open(self.logdir / f"{name}.log", "ab")
        proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.procs.append(proc)
        return proc

    def launch(self) -> None:
        t0 = time.time()
        base = self.cfg.base_env()
        # WildGuard, two replicas (the proxy round-robins them)
        env = dict(base, CUDA_VISIBLE_DEVICES=phys_cards(self.judge_gpus),
                   RAY_TMPDIR=f"{os.environ.get('SSP_TMP', '/tmp')}/ssp-onejob-judge-{self.judge_port}")
        replicas = len(self.judge_gpus.split(","))
        self._spawn([self.cfg.python, "-m", "openrlhf.cli.serve_wildguard",
                     "--model_path", self.cfg.judge_model, "--port", str(self.judge_port),
                     "--host", "127.0.0.1", "--bf16", "--num_gpus", str(replicas),
                     "--tensor_parallel_size", "1", "--seed", "42", "--max_len", "8192",
                     "--max_num_seqs", "128", "--batch_size", "128",
                     "--gpu_memory_utilization", os.environ.get("ONEJOB_JUDGE_UTIL", "0.13")],
                    env, "wildguard")
        probe = {"queries": [{"prompt": "hi", "response": "hello", "game_idx": 0}], "prompts": None}
        self._wait(lambda: "labels" in _try_post(self.url, probe), "wildguard", 1800)
        for _ in range(replicas):  # one request per replica: all loaded before the next engine
            _try_post(self.url, probe, timeout=900)
        self.log(f"onejob: wildguard x{replicas} up after {time.time() - t0:.0f}s")
        shared = os.environ.get("ONEJOB_SHARED_BASE", "0") == "1"
        if not shared:
            self._launch_topic(base, t0)
        self._launch_servers(base, t0, shared)
        if shared:
            self._launch_topic(base, t0, backends=self.server_urls)
        self._start_mem_logger()
    def _launch_topic(self, base, t0, backends=None) -> None:
        # the topic/fulfil judge stays on the ORIGINAL base even when the policy
        # base is pi0: a judge must not move with the players.  ONEJOB_SHARED_BASE=1:
        # the rollout servers ARE that base (pi0 is the LoRA "base"), so the judge
        # front-end runs on CPU and forwards to them (serve_topic_judge.py backend mode).
        env = dict(base, CUDA_VISIBLE_DEVICES="" if backends else phys_cards(self.topic_gpu),
                   TOPIC_JUDGE_MODEL=os.environ.get("TOPIC_JUDGE_BASE_MODEL") or self.model,
                   TOPIC_JUDGE_PORT=str(self.topic_port),
                   TOPIC_JUDGE_UTIL=os.environ.get("ONEJOB_TOPIC_UTIL", "0.20"),
                   TOPIC_JUDGE_MAX_SEQS="256", TOPIC_JUDGE_MAX_BATCH="256",
                   TOPIC_JUDGE_MAX_LEN=os.environ.get("ONEJOB_TOPIC_MAX_LEN", "8192"), TOPIC_JUDGE_EAGER="0",
                   **({"TOPIC_JUDGE_BACKEND_URLS": ",".join(backends)} if backends else {}))
        self._spawn([self.cfg.python, str(Path(__file__).parent / "serve_topic_judge.py")], env, "topic")
        self._wait(lambda: _ok(f"http://127.0.0.1:{self.topic_port}/health"), "topic judge", 1800)
        self.log(f"onejob: topic judge up after {time.time() - t0:.0f}s" + (" (CPU front-end on the rollout servers)" if backends else ""))
    def _launch_servers(self, base, t0, shared) -> None:
        if shared:
            judge_base = os.environ["TOPIC_JUDGE_BASE_MODEL"]
            served = ["--served-model-name", "local-topic-judge",
                      "--lora-modules", f"base={os.environ['SHARED_BASE_PI0_LORA']}"]
            rank = os.environ.get("SHARED_BASE_LORA_RANK", "128")
            max_len = os.environ.get("ONEJOB_TOPIC_MAX_LEN", "8192")
        for card, (port, util) in enumerate(zip(self.server_ports, self.server_utils)):
            env = {k: v for k, v in base.items()
                   if k not in ("PYTHONPATH", "VLLM_USE_V1", "VLLM_ENABLE_V1_MULTIPROCESSING")}
            env.update(CUDA_VISIBLE_DEVICES=phys_cards(card), VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
                       VLLM_SERVER_DEV_MODE="1", PATH=f"{self.renv}/bin:" + base.get("PATH", ""))
            if os.environ.get("ROLLOUT_CUDA_HOME"):  # rollout env (torch cu13x) must not see the trainer toolkit
                env["CUDA_HOME"] = os.environ["ROLLOUT_CUDA_HOME"]
            self._spawn([f"{self.renv}/bin/python", "-m", "vllm.entrypoints.openai.api_server",
                         "--model", judge_base if shared else self.model,
                         *(served if shared else ["--served-model-name", "base"]),
                         "--host", "127.0.0.1", "--port", str(port),
                         "--enable-lora", "--max-loras", os.environ.get("ONEJOB_MAX_LORAS", "16"),
                         "--max-cpu-loras", os.environ.get("ONEJOB_MAX_CPU_LORAS", "16"), "--max-lora-rank", rank if shared else str(os.environ.get("LORA_RANK", "64")),
                         "--gpu-memory-utilization", str(util), "--max-model-len", max_len if shared else "4096",
                         "--enable-prefix-caching", "--enable-sleep-mode",
                         "--max-num-seqs", os.environ.get("ONEJOB_MAX_NUM_SEQS", "1024"),
                         "--max-num-batched-tokens", os.environ.get("ONEJOB_MAX_BATCHED_TOKENS", "16384"),
                         "--seed", "0", "--dtype", "bfloat16"]
                        # ONEJOB_KV_DTYPE=fp8 (10-08): ~2x KV tokens per card; defender-phase rollout servers were KV-full
                        + (["--kv-cache-dtype", os.environ["ONEJOB_KV_DTYPE"]] if os.environ.get("ONEJOB_KV_DTYPE") else [])
                        # ONEJOB_SCHED_PRIORITY=1 (10-09): priority scheduling so the shared-base topic judge (served by these
                        # same servers) is not starved by cell/eval generation (cells failed TopicAPIError 3x on 10-09)
                        + (["--scheduling-policy", "priority"] if os.environ.get("ONEJOB_SCHED_PRIORITY") == "1" else []),
                        env, f"server_{card}")
        for url in self.server_urls:
            self._wait(lambda u=url: _ok(f"{u}/health"), f"server {url}", 1800)
        self.log(f"onejob: {len(self.server_urls)} rollout servers up after {time.time() - t0:.0f}s"
                 + (" (shared base: abliterated weights, pi0 = LoRA 'base')" if shared else ""))

    def _start_mem_logger(self) -> None:
        """Every 15 s: job cgroup memory, node free memory, /dev/shm use --
        evidence if the job is killed without a message again."""
        import threading

        def cgroup_bytes():
            try:
                for line in open("/proc/self/cgroup"):
                    path = line.strip().split(":", 2)[-1]
                    for f in (f"/sys/fs/cgroup{path}/memory.current",
                              f"/sys/fs/cgroup/memory{path}/memory.usage_in_bytes"):
                        if os.path.exists(f):
                            return int(open(f).read())
            except Exception:
                return -1
            return -1

        def loop():
            out = open(self.logdir / "mem.log", "a")
            while True:
                try:
                    meminfo = dict(l.split(":", 1) for l in open("/proc/meminfo"))
                    avail = int(meminfo["MemAvailable"].split()[0]) // (1024 ** 2)
                    shm = os.statvfs("/dev/shm")
                    shm_used = (shm.f_blocks - shm.f_bfree) * shm.f_frsize // (1024 ** 3)
                    out.write(f"{time.strftime('%H:%M:%S')} cgroup_GB={cgroup_bytes() / 1024 ** 3:.1f} "
                              f"node_avail_GB={avail} shm_used_GB={shm_used}\n")
                    out.flush()
                except Exception as exc:
                    out.write(f"mem logger error {exc!r}\n")
                time.sleep(15)

        threading.Thread(target=loop, daemon=True).start()

    def _wait(self, check, what, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if check():
                return
            for proc in self.procs:
                if proc.poll() is not None:
                    raise RuntimeError(f"onejob: a service exited (rc={proc.returncode}) while waiting for {what}")
            time.sleep(5)
        raise RuntimeError(f"onejob: {what} not up after {timeout}s")

    def shutdown(self) -> None:
        self.stop_cell_judges() if getattr(self, "cell_judge_procs", None) else None
        for proc in self.procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(3)
        for proc in self.procs:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # -- cell WildGuard replicas (ONEJOB_CELL_WG_REPLICAS, 10-09) ---------------
    # Payoff cells were judge-bound: one WildGuard on card 0 at 98% while the cards' generation finished
    # minutes earlier (2 cells x 16 judge threads, chunk 1 s -> 225 s, ~19 min per cell pair). During the
    # cell phase only, the rollout server on each replica card is put to sleep (level 1 frees its weights
    # and KV) and an extra WildGuard (same model/args, greedy -> identical labels) is started there;
    # each cell worker gets one judge URL (round-robin). Replicas are killed before training resumes.
    def start_cell_judges(self, cards) -> list[str]:
        self.cell_judge_procs = getattr(self, "cell_judge_procs", [])
        urls = []
        base = self.cfg.base_env()
        base.pop("RAY_ADDRESS", None)
        probe = {"queries": [{"prompt": "hi", "response": "hello", "game_idx": 0}], "prompts": None}
        self.sleep(cards)
        for i, card in enumerate(cards):
            port = self.judge_port + 5 + i
            url = f"http://127.0.0.1:{port}/classify"
            env = dict(base, CUDA_VISIBLE_DEVICES=phys_cards(str(card)),
                       RAY_TMPDIR=f"{os.environ.get('SSP_TMP', '/tmp')}/ssp-onejob-celljudge-{port}")
            handle = open(self.logdir / f"wildguard_cell{i}.log", "ab")
            proc = subprocess.Popen(
                [self.cfg.python, "-m", "openrlhf.cli.serve_wildguard",
                 "--model_path", self.cfg.judge_model, "--port", str(port),
                 "--host", "127.0.0.1", "--bf16", "--num_gpus", "1",
                 "--tensor_parallel_size", "1", "--seed", "42", "--max_len", "8192",
                 "--max_num_seqs", "128", "--batch_size", "128",
                 "--gpu_memory_utilization", os.environ.get("ONEJOB_CELL_WG_UTIL", "0.60")],
                env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            self.cell_judge_procs.append(proc)
            deadline = time.time() + float(os.environ.get("ONEJOB_CELL_WG_TIMEOUT", "900"))
            while time.time() < deadline and proc.poll() is None and "labels" not in _try_post(url, probe):
                time.sleep(5)
            if proc.poll() is None and "labels" in _try_post(url, probe, timeout=600):
                urls.append(url)
                self.log(f"onejob: cell judge replica on card {card} -> {url}")
            else:
                self.log(f"onejob: cell judge replica on card {card} FAILED (rc={proc.poll()}); continuing without it")
        return urls
    def stop_cell_judges(self) -> None:
        procs, self.cell_judge_procs = getattr(self, "cell_judge_procs", []), []
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(5)
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
        if procs:
            time.sleep(10)  # let the driver release the replica's memory before a server wakes there
            self.log(f"onejob: stopped {len(procs)} cell judge replica(s)")
    # -- role switching -------------------------------------------------------
    def sleep(self, cards) -> None:
        for card in cards:
            if card not in self.asleep:
                _post(f"{self.server_urls[card]}/sleep?level=1")
                self.asleep.add(card)
        self.log(f"onejob: servers asleep on cards {sorted(self.asleep)}")

    def wake(self, cards=None) -> None:
        for card in sorted(self.asleep if cards is None else cards):
            if card in self.asleep:
                _post(f"{self.server_urls[card]}/wake_up")
                self.asleep.discard(card)
        self.log("onejob: all servers awake" if not self.asleep else f"onejob: asleep {sorted(self.asleep)}")

    def awake_urls(self) -> list[str]:
        return [u for c, u in enumerate(self.server_urls) if c not in self.asleep]

    # -- environment for clients ------------------------------------------------
    def client_env(self) -> dict[str, str]:
        return {
            "TOPIC_JUDGE_BACKEND": "local",
            "TOPIC_JUDGE_MANAGED": "1",
            "TOPIC_API_PROVIDER": "openai_compatible",
            "TOPIC_API_MODEL": "local-topic-judge",
            "TOPIC_API_BASE_URL": f"http://127.0.0.1:{self.topic_port}/v1",
            "TOPIC_API_KEY": "local",
            "TOPIC_API_CONCURRENCY": "32",
        }


def _try_post(url: str, payload: dict, timeout: float = 10) -> str:
    try:
        return _post(url, payload, timeout=timeout)
    except Exception:
        return ""
