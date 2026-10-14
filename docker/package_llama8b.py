#!/usr/bin/env python3
"""Package runtime sources; include fixed W&B credentials only when requested."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
TOP = "SafeSelfPlay-llama8b-h200"
ADDED = {
    "EXPERIMENTS.md", "TOPIC_JUDGE_AND_SHARED_REWARD.md", "roll/utils/history_selfplay.py",
    "roll/utils/shared_game_reward.py", "roll/utils/topic_api.py",
    "roll/utils/local_topic_runtime.py", "tests/test_local_topic_runtime.py",
    "tests/test_shared_game_reward.py", "tests/test_topic_api.py",
    "scripts/slurm/history_selfplay.py",
    "tests/test_history_selfplay.py", "tests/test_startup_resilience.py", "tests/test_docker_llama8b.py",
    "tests/test_local_early_stopping.py",
    "docker/Dockerfile.llama8b", "docker/Dockerfile.llama8b.dockerignore",
    "docker/requirements-llama8b.txt", "docker/llama8b_entrypoint.py",
    "docker/run_llama8b.sh", "docker/export_llama8b.sh", "docker/package_llama8b.py",
    "docker/download_to_mac.command",
    "docker/download_source_to_mac.command", "docker/QUICKSTART.source.llama8b.md",
    "docker/README.llama8b.md", "docker/QUICKSTART.llama8b.md",
}


def source_paths() -> list[str]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    selected = set(ADDED)
    for name in tracked:
        path = Path(name)
        if (name.startswith(("roll/", "_modal_patches/", "scripts/slurm/", "data/safety_selfplay/", "tests/"))
                or (len(path.parts) == 1 and path.suffix in {".py", ".sh"})
                or name in {"README.md", "LICENSE", "LICENSE.md", ".gitignore", "pyproject.toml"}):
            selected.add(name)
    for name in selected:
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Expected a regular source file: {name}")
        if any(part in {".git", "__pycache__"} or part.startswith(".env") for part in path.parts):
            raise RuntimeError(f"Refusing non-source content: {name}")
        if path.suffix in {".safetensors", ".pt", ".pth", ".bin", ".pyc"}:
            raise RuntimeError(f"Refusing model/cache content: {name}")
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/releases/llama8b-labels-20260919/safeselfplay-llama8b-h200-labels.build.tar.gz")
    parser.add_argument("--include-wandb", action="store_true", help="include the private teacher-handoff W&B configuration")
    parser.add_argument("--source-only-guide", action="store_true",
                        help="make START_HERE.md describe building from source without a prebuilt image")
    args = parser.parse_args()
    files = source_paths()
    if args.include_wandb:
        private = ROOT / "docker/wandb.fixed.json"
        if not private.is_file() or private.is_symlink():
            raise RuntimeError("Missing private W&B configuration")
        files.append("docker/wandb.fixed.json")
    metadata = {
        "kind": "docker-build-bundle", "image_included": False, "model_weights_included": False,
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "contains_current_working_tree_changes": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip()),
        "fixed_wandb_credential_included": args.include_wandb,
        "quickstart": "source-build" if args.source_only_guide else "prebuilt-image",
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": len(files),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=f"{TOP}/{name}", recursive=False)
        for name in ("BUILD_VALIDATION.json", "runtime-packages.txt"):
            report = args.output.parent / name
            if report.is_file():
                archive.add(report, arcname=f"{TOP}/{name}", recursive=False)
        for name, text in {
            "BUILD_BUNDLE.json": json.dumps(metadata, indent=2) + "\n",
            "START_HERE.md": (ROOT / "docker" / ("QUICKSTART.source.llama8b.md" if args.source_only_guide
                                                else "QUICKSTART.llama8b.md")).read_text(),
        }.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{TOP}/{name}")
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    if args.include_wandb:
        args.output.chmod(0o600)
    print(f"Docker build bundle: {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
