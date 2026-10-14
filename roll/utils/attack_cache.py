"""Cache of frozen-attacker rewrites, keyed so a hit is bit-for-bit exact.

In this single-turn game the attacker moves first against an empty history
(``language_game.py`` builds turn 0 from ``game["history"] == []`` plus the seed
prompt), so its rewrite cannot depend on the defender. Payoff cells additionally
generate with an explicit per-request ``seed`` (``attacker = seed_base +
candidate_index``) at temperature 1.0, which makes the rewrite a deterministic
function of

    (attacker adapter, seed prompt, prompt type, template, decoding, seed)

Every one of those is in the cache key below, so reusing an entry reproduces the
generation exactly rather than approximating it. A cell that re-runs the same
attacker against a different defender therefore gets the identical attack it
would have generated, and only the defender has to run.

Training-time generation is NOT seeded (``experience_maker.py`` builds
SamplingParams without ``seed=``), so a cache hit there is one fixed draw from a
stochastic policy rather than a fresh one. ``allow_unseeded`` must be set
explicitly to accept that, and the entry records ``seeded: false`` so the
difference stays visible downstream.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

CACHE_VERSION = "attack-cache-v1"
_LOCK = threading.Lock()


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def decoding_fingerprint(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    min_new_tokens: int,
    prompt_max_tokens: int,
    skip_special_tokens: bool = False,
    include_stop_str_in_output: bool = True,
) -> str:
    return _sha(
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "prompt_max_tokens": prompt_max_tokens,
            "skip_special_tokens": skip_special_tokens,
            "include_stop_str_in_output": include_stop_str_in_output,
            # Constrained decoding changes the attack distribution: never reuse
            # unconstrained attacks for a grammar run (or vice versa).
            **_format_grammar_identity(),
            # Seeded sampling is only bit-exact within one engine: the vLLM 0.30
            # servers must not reuse the in-process 0.8.2 engines' attacks.
            **({"engine": "vllm-server"} if os.environ.get("CELL_SERVERS") else {}),
            # the policy base (raw base vs pi0) changes every attacker's outputs
            **({"policy_base": os.path.realpath(os.environ["BASE_MODEL_PATH"])}
               if os.environ.get("BASE_MODEL_PATH") else {}),
        }
    )


def _format_grammar_identity() -> dict:
    if os.environ.get("FORMAT_GRAMMAR") != "1":
        return {}
    from roll.utils.format_grammar import FORMAT_GRAMMAR_VERSION, GRAMMAR, rescue_reserve
    return {"format_grammar": FORMAT_GRAMMAR_VERSION, "grammar_sha": _sha({"g": GRAMMAR}),
            "rescue_reserve": rescue_reserve()}


class AttackCache:
    """One shard per (attacker, template, decoding) triple."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        attacker_label: str,
        attacker_sha256: str,
        base_model: str,
        template_sha256: str,
        decoding_sha256: str,
        seed_scheme: str,
        prompt_dataset_sha256: str,
        seeded: bool,
        allow_unseeded: bool = False,
    ) -> None:
        if not seeded and not allow_unseeded:
            raise ValueError(
                "refusing to cache unseeded attacker generations: decoding is "
                "stochastic, so a cache hit would replace a fresh draw with one "
                "fixed sample. Pass allow_unseeded=True to accept that."
            )
        self.meta = {
            "cache_version": CACHE_VERSION,
            "attacker_label": attacker_label,
            "attacker_sha256": attacker_sha256,
            "base_model": base_model,
            "template_sha256": template_sha256,
            "decoding_sha256": decoding_sha256,
            "seed_scheme": seed_scheme,
            "prompt_dataset_sha256": prompt_dataset_sha256,
            "seeded": seeded,
        }
        self.shard = _sha(self.meta)[:16]
        self.dir = Path(root) / attacker_sha256[:16] / self.shard
        self.path = self.dir / "attacks.jsonl"
        self.meta_path = self.dir / "meta.json"
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        if self.meta_path.is_file():
            stored = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if stored != self.meta:            # shard hash collision or edited metadata
                raise ValueError(
                    f"attack cache metadata mismatch at {self.meta_path}; refusing to reuse"
                )
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._entries[row["key"]] = row

    @staticmethod
    def key(seed_prompt_id: Any, prompt_type: str, request_seed: int | None) -> str:
        return f"{seed_prompt_id}|{prompt_type}|{'' if request_seed is None else request_seed}"

    #: every field the episode record reads off a generation result
    OUTPUT_FIELDS = (
        "text", "vllm_raw_text", "finish_reason", "stop_reason", "token_count",
        "output_token_ids_sha256", "rendered_prompt_char_count",
        "rendered_prompt_token_count", "tokenized_prompt_token_count",
        "tokenized_prompt_ids_sha256", "prompt_truncated",
    )

    def get(self, seed_prompt_id: Any, prompt_type: str, request_seed: int | None,
            seed_prompt: str) -> tuple[str, dict[str, Any]] | None:
        """Return (parsed_attack, generation_output) or None."""
        row = self._entries.get(self.key(seed_prompt_id, prompt_type, request_seed))
        if row is None:
            self.misses += 1
            return None
        # guard against a pool edit that shifted indices under a stable id
        if row.get("seed_prompt_sha256") != _sha(seed_prompt):
            self.misses += 1
            return None
        output = row.get("output")
        if not isinstance(output, dict) or any(f not in output for f in self.OUTPUT_FIELDS):
            self.misses += 1          # entry predates the full-record format
            return None
        self.hits += 1
        return row["attack"], output

    def put(self, seed_prompt_id: Any, prompt_type: str, request_seed: int | None,
            seed_prompt: str, attack: str, output: Mapping[str, Any]) -> None:
        missing = [f for f in self.OUTPUT_FIELDS if f not in output]
        if missing:
            raise ValueError(f"generation output is missing {missing}; refusing to cache")
        row = {
            "key": self.key(seed_prompt_id, prompt_type, request_seed),
            "seed_prompt_id": seed_prompt_id,
            "prompt_type": prompt_type,
            "request_seed": request_seed,
            "seed_prompt_sha256": _sha(seed_prompt),
            "attack": attack,
            "output": {f: output[f] for f in self.OUTPUT_FIELDS},
        }
        self._entries[row["key"]] = row
        with _LOCK:
            self.dir.mkdir(parents=True, exist_ok=True)
            if not self.meta_path.is_file():
                self.meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "shard": self.shard,
            "path": str(self.path),
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "seeded": self.meta["seeded"],
        }
