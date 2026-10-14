#!/usr/bin/env python3
"""Import the Modal cold-PSRO coordinator with no Modal runtime attached.

The coordinator, the payoff-cell generator and the pure scheduling/reward
helpers are the scientific contract of the archived 3x3 run.  Re-implementing
them locally would fork that contract, so this module imports them verbatim and
replaces only the two things Modal owns: the ``modal`` package itself and the
paths the functions read and write.

``_stub_modal`` mirrors scripts/slurm/prepare_sources.py.  The decorators become
pass-throughs, so ``@app.function(...) def f`` leaves ``f`` a plain function;
``_bind_local`` then re-attaches the ``.remote``/``.local``/``.get_raw_f``
handles that the coordinator calls, and turns ``.spawn`` into a no-op so the
self-detaching controller loop becomes an ordinary ``while`` loop here.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Any:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, _name: str) -> "_Any":
        return _Any()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _Any()

    def __or__(self, _other: Any) -> "_Any":
        return _Any()

    def __bool__(self) -> bool:
        return False


def _stub_modal() -> None:
    if "modal" in sys.modules:
        return
    stub = types.ModuleType("modal")

    def _stub_getattr(name: str) -> Any:
        # torch.library walks every module looking for a source file; a dunder
        # that answers with _Any() instead of raising breaks inspect.getmodule.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Any()

    stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
    stub.__file__ = str(Path(__file__).resolve())
    stub.__spec__ = None
    stub.__loader__ = None
    stub.__package__ = ""
    for attr in (
        "App", "Image", "Secret", "Volume", "Function", "Sandbox", "Cls",
        "Dict", "Queue", "CloudBucketMount", "Retries", "Period", "Cron",
    ):
        setattr(stub, attr, _Any())
    sys.modules["modal"] = stub
    for extra in ("modal.runner", "modal.io_streams", "modal.exception"):
        sys.modules.setdefault(extra, types.ModuleType(extra))


def _bind_local(function: Any) -> Any:
    """Give a de-Modal'd function the call handles the coordinator uses."""

    function.remote = function
    function.local = function
    function.get_raw_f = lambda: function
    function.spawn = lambda *a, **k: types.SimpleNamespace(
        object_id="local-inline"
    )
    return function


COLD_COORDINATOR = "modal_role_lora_zero_sum_psro"


def load_modules(coordinator: str = COLD_COORDINATOR) -> tuple[Any, Any]:
    """Return ``(psro, payoff)`` with local call handles already attached.

    ``coordinator`` selects the coordinator module. The hot-start run uses a
    derived copy (see make_hot_psro_coordinator.py) whose contract records the
    warm-start initialization; everything else about it is byte-identical.
    """

    import importlib

    _stub_modal()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import modal_upstream_v2_payoff as payoff

    # The coordinator unwraps this one at import time, so bind before importing.
    _bind_local(payoff.evaluate_upstream_v2_raw_payoff_cell)

    psro = importlib.import_module(coordinator)

    for name in (
        "evaluate_zero_sum_psro_cell",
        "train_cold_psro_oracle",
        "advance_cold_psro",
    ):
        _bind_local(getattr(psro, name))
    return psro, payoff


def retarget(
    psro: Any,
    payoff: Any,
    *,
    psro_root: Path,
    raw_payoff_root: Path,
    upstream_source: Path,
    judge_url: str,
) -> None:
    """Point the imported functions at this filesystem instead of Modal's."""

    psro_root.mkdir(parents=True, exist_ok=True)
    raw_payoff_root.mkdir(parents=True, exist_ok=True)
    psro.PSRO_OUTPUT_ROOT = psro_root
    psro.RAW_PAYOFF_ROOT = raw_payoff_root
    payoff.PAYOFF_ROOT = raw_payoff_root
    payoff.UPSTREAM_SOURCE = upstream_source
    psro._stable_wildguard_rm_url = lambda *a, **k: judge_url
    payoff._stable_wildguard_rm_url = lambda *a, **k: judge_url

    base_dir = os.environ.get("COLD_PSRO_BASE_MODEL_DIR")
    if base_dir:
        # vLLM resolves the base by repo id inside the payoff evaluator; on a
        # cluster without egress the snapshot directory is the same model.
        payoff.LLAMA_ABLITERATED_MODEL = base_dir


def implementation_hashes(psro: Any) -> dict[str, str]:
    return psro._psro_implementation_hashes()
