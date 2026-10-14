"""Upload each role's final checkpoint to the Hugging Face Hub.

Off unless ``HF_CHECKPOINT_REPO`` names a repo. Uploading is a side effect of a
finished role, never a precondition for one: a 300-step oracle must not be lost
because a token expired or the Hub was briefly unreachable, so every failure
here is reported and swallowed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def _hub_id(path: str) -> str:
    """Recover `org/name` from a local snapshot path, for a usable record."""
    for part in Path(path).parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/", 1)
    return path


def checkpoint_publish_target() -> str:
    return os.environ.get("HF_CHECKPOINT_REPO", "").strip()


def publish_role_checkpoint(
    checkpoint: Path,
    *,
    run_suffix: str,
    target: str,
    role: str,
    extra: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> str | None:
    """Upload ``checkpoint`` as ``<run_suffix>/<target>``; return its URL or None.

    Returns None when publishing is switched off or when it failed. The caller
    treats that as "not published", never as "training failed".
    """
    repo_id = checkpoint_publish_target()
    if not repo_id:
        return None
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        log(f"  checkpoint publish skipped: HF_CHECKPOINT_REPO={repo_id} but HF_TOKEN is unset")
        return None
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        log(f"  checkpoint publish skipped: {checkpoint} is not a directory")
        return None

    path_in_repo = f"{run_suffix}/{target}"
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        private = os.environ.get("HF_CHECKPOINT_PRIVATE", "1") != "0"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        # Provenance travels with the weights: a bare adapter directory does not
        # say which run, role, step or reward produced it.
        metadata = {
            "run_suffix": run_suffix, "target": target, "role": role,
            "checkpoint": checkpoint.name,
            "base_model": _hub_id(os.environ.get("BASE_MODEL_PATH", "")),
            **(extra or {}),
        }
        api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(checkpoint),
            path_in_repo=path_in_repo,
            commit_message=f"{run_suffix}: final {target} ({role}) checkpoint",
            # PEFT writes a README.md whose `base_model:` is the ABSOLUTE LOCAL
            # path the adapter was trained from. That is not a model id, so the
            # Hub rejects the whole commit with "Invalid metadata in README.md"
            # and no part of the checkpoint lands. The card says nothing our
            # metadata file does not, so leave it behind.
            ignore_patterns=["README.md"],
        )
        api.upload_file(
            repo_id=repo_id, repo_type="model",
            path_or_fileobj=json.dumps(metadata, indent=2, sort_keys=True).encode(),
            path_in_repo=f"{path_in_repo}/safeselfplay_metadata.json",
            commit_message=f"{run_suffix}: {target} provenance",
        )
    except Exception as exc:  # noqa: BLE001 - never lose a finished role to an upload
        log(f"  checkpoint publish FAILED for {target}: {type(exc).__name__}: {exc}")
        return None
    url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
    log(f"  published {target} -> {url}")
    return url
