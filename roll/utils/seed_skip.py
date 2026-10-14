"""V7_SEED_SKIP (10-11, user): skip benign seeds the cooperative reference cannot fulfil.

Under V7_REF_GATE=benign, label_games asks the reference model to answer the SEED and the fulfil judge
checks it (labels["seed_reference_fulfils"]); V7_CONCISE drops a benign game whose seed fails
(DropGame("benign_seed_unfulfillable")) after its rollout and judges were already paid for.  The drop
stays.  This module adds the user's rule: "if a task's pass rate < 50% and visited > 4 times, skip that
seed in the future".

* VISIT = one (seed, label_games call) occurrence: the GROUP_SIZE copies of a seed inside one scoring
  call count once (seed_reference_fulfils is computed once per unique seed per call).
* stats: one JSON per run (V7_SEED_STATS; default <dir of GAME_DUMP>/seed_stats.json), keyed by
  sha1(seed text) -> [visits, passes]; flock + tmp/rename, so ranks / judge threads can share it.
  Persists across oracles of a run (the reference model is fixed: this is a property of the seed).
* blacklist: visits >= V7_SEED_SKIP_MIN_VISITS (5) and passes/visits < V7_SEED_SKIP_MAX_PASS (0.5);
  sticky; never grows past V7_SEED_SKIP_MAX_FRAC (0.2) of the benign pool (loud log, stop adding).
* sampler (trainer): blacklisted benign seeds drawn by the dataloader are replaced by a random
  non-blacklisted benign seed, keeping the slot's label, so batch size and H/B balance are unchanged.
  Harmful seeds are never touched.
Off (V7_SEED_SKIP unset/0): nothing here is imported or called.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import threading
import time

_TLOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("V7_SEED_SKIP", "0") == "1"


def min_visits() -> int:
    return int(os.environ.get("V7_SEED_SKIP_MIN_VISITS", "5"))


def max_pass() -> float:
    return float(os.environ.get("V7_SEED_SKIP_MAX_PASS", "0.5"))


def max_frac() -> float:
    return float(os.environ.get("V7_SEED_SKIP_MAX_FRAC", "0.2"))


def seed_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stats_path() -> str:
    p = os.environ.get("V7_SEED_STATS")
    if p:
        return p
    dump = os.environ.get("GAME_DUMP")
    if dump:
        return os.path.join(os.path.dirname(os.path.abspath(dump)), "seed_stats.json")
    return os.path.join(os.environ.get("RUNNER_LOG_DIR") or os.getcwd(), "seed_stats.json")


def _empty() -> dict:
    return {"version": 1, "pool": None, "seeds": {}, "blacklist": [], "guard_tripped": False}


def read_state(path: str | None = None) -> dict:
    """Lock-free read (writers rename atomically)."""
    path = path or stats_path()
    for _ in range(3):
        try:
            with open(path) as f:
                st = json.load(f)
            base = _empty()
            base.update(st)
            return base
        except FileNotFoundError:
            return _empty()
        except (OSError, ValueError):
            time.sleep(0.05)
    return _empty()


def _update(fn, path: str | None = None):
    """Read-modify-write under a thread lock + flock on <path>.lock; atomic tmp+rename."""
    path = path or stats_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with _TLOCK, open(path + ".lock", "a+") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            st = read_state(path)
            out = fn(st)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            with open(tmp, "w") as f:
                json.dump(st, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return out
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


def _cap(pool) -> int | None:
    return int(max_frac() * pool) if pool else None


def record(outcomes: dict, path: str | None = None) -> dict:
    """outcomes: {seed_text: bool seed_reference_fulfils} for ONE scoring call (one visit per seed)."""
    if not outcomes:
        return {}
    mv, mp = min_visits(), max_pass()

    def fn(st):
        seeds, bl = st["seeds"], st["blacklist"]
        bls, new, guarded = set(bl), [], 0
        cap = _cap(st.get("pool"))
        for text, ok in outcomes.items():
            k = seed_key(text)
            v, p = seeds.get(k, (0, 0))
            v, p = v + 1, p + int(bool(ok))
            seeds[k] = [v, p]
            if k in bls or v < mv or p / v >= mp:
                continue
            if cap is not None and len(bl) >= cap:
                guarded += 1
                continue
            bl.append(k)
            bls.add(k)
            new.append(k)
        if guarded:
            st["guard_tripped"] = True
        return {"recorded": len(outcomes), "new_blacklisted": len(new), "guarded": guarded,
                "blacklisted": len(bl), "tracked": len(seeds), "pool": st.get("pool")}

    res = _update(fn, path)
    if res.get("guarded"):
        print(f"[V7_SEED_SKIP] !!!!! GUARD: blacklist at cap ({res['blacklisted']} >= {max_frac():g} x pool "
              f"{res['pool']}); NOT adding {res['guarded']} more failing seeds -- check the reference/fulfil judge "
              f"!!!!!", flush=True)
    if res.get("new_blacklisted"):
        print("[V7_SEED_SKIP] record " + json.dumps(res, sort_keys=True), flush=True)
    return res


def record_labels(games, labels, path: str | None = None) -> dict:
    """language_game hook: games = label_games input, labels = its output.  Benign seeds only."""
    outcomes = {}
    for g in games:
        if g.get("prompt_type") != "generated_benign":
            continue
        v = labels.get(g["game_idx"], {}).get("seed_reference_fulfils")
        if v is None:
            continue
        outcomes.setdefault(g["seed"], bool(v))
    return record(outcomes, path)


class SeedSkipSampler:
    """Trainer-side filter on dataloader batches ((prompts, completions), labels)."""

    def __init__(self, prompts, labels, *, rng_seed=0, rank=0, path: str | None = None, log=print):
        self.path = path or stats_path()
        self.rank = rank
        self.log = log
        self.prompts = prompts  # list of (text, completion)
        self.benign_idx = [i for i, l in enumerate(labels) if "benign" in l]
        self.keys = {i: seed_key(prompts[i][0]) for i in self.benign_idx}
        self.pool_keys = set(self.keys.values())
        self.pool = len(self.pool_keys)
        self.rng = random.Random(f"v7_seed_skip|{rng_seed}|{rank}")
        self._loud = False

        def reg(st):
            st["pool"] = max(int(st.get("pool") or 0), self.pool)
        _update(reg, self.path)
        self.log(f"[V7_SEED_SKIP] on: stats={self.path} pool={self.pool} min_visits={min_visits()} "
                 f"max_pass={max_pass():g} max_frac={max_frac():g}")

    def blacklist(self):
        st = read_state(self.path)
        bl = [k for k in st["blacklist"] if k in self.pool_keys]
        cap = _cap(self.pool)
        if len(bl) > cap:
            if not self._loud:
                self.log(f"[V7_SEED_SKIP] !!!!! GUARD: blacklist {len(bl)} > {max_frac():g} x pool {self.pool}; "
                         f"using the first {cap} only !!!!!")
                self._loud = True
            bl = bl[:cap]
        return set(bl), st

    def filter(self, rand_prompts, labels):
        bl, st = self.blacklist()
        texts, comps = list(rand_prompts[0]), list(rand_prompts[1])
        skipped = benign = 0
        if bl:
            for j, lab in enumerate(labels):
                if "benign" not in lab:
                    continue  # never skip harmful seeds
                benign += 1
                if seed_key(texts[j]) not in bl:
                    continue
                for _ in range(10000):
                    i = self.rng.choice(self.benign_idx)
                    if self.keys[i] not in bl:
                        break
                else:
                    continue  # pathological: keep the original
                texts[j], comps[j] = self.prompts[i][0], self.prompts[i][1]
                skipped += 1
        else:
            benign = sum("benign" in l for l in labels)
        info = {"tracked": len(st["seeds"]), "blacklisted": len(bl), "skipped_draws": skipped,
                "pool": self.pool, "benign_draws": benign, "rank": self.rank}
        self.log("[V7_SEED_SKIP] " + json.dumps(info))
        typ = type(rand_prompts)
        new_prompts = typ([texts, comps]) if typ in (list, tuple) else [texts, comps]
        return new_prompts, labels, info
