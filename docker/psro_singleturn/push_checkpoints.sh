#!/usr/bin/env bash
# Push the per-oracle LoRA adapters from a run to a HuggingFace repo, so they can be shared back.
# Each oracle's final adapter lives at: $RUN/training/<ORACLE>/ckpt/global_step*_hf  (HF/PEFT format).
#
# Usage:  HF_PUSH_REPO=<org>/<repo> HF_TOKEN=hf_xxx ./push_checkpoints.sh [RUN_TAG]
# Safe to re-run (re-uploads only; HF dedupes). Called automatically at the end of entrypoint.sh
# when HF_PUSH_REPO is set.
set -eu -o pipefail
HERE=/zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker
source "$HERE/psro.env"
SUF=${1:-${RUN_TAG:-psro_docker}}
RUN=$WORKSPACE/psro/$SUF

if [ -z "${HF_PUSH_REPO:-}" ]; then
  echo "[push] HF_PUSH_REPO not set -- skipping. To share checkpoints, set HF_PUSH_REPO=<org>/<repo>"
  echo "       and HF_TOKEN, then run: ./push_checkpoints.sh $SUF"
  exit 0
fi
[ -n "${HF_TOKEN:-}" ] || { echo "[push] HF_PUSH_REPO set but HF_TOKEN missing -- cannot authenticate."; exit 6; }

echo "[push] uploading LoRA adapters from $RUN to hf.co/$HF_PUSH_REPO ..."
$SSP_PYTHON - "$RUN" "$HF_PUSH_REPO" <<'PY'
import sys, glob, os
from huggingface_hub import HfApi
run, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(repo, repo_type="model", exist_ok=True, private=True)
ckpts = sorted(glob.glob(f"{run}/training/*/ckpt/global_step*_hf"))
if not ckpts:
    print("  no adapters found under", run + "/training/*/ckpt/global_step*_hf"); raise SystemExit(0)
tag = os.path.basename(run.rstrip("/"))
for c in ckpts:
    oracle = c.split("/training/")[1].split("/")[0]            # e.g. A1, D2
    dest = f"{tag}/{oracle}/{os.path.basename(c)}"
    print("  push", oracle, "->", f"{repo}/{dest}", flush=True)
    api.upload_folder(folder_path=c, repo_id=repo, path_in_repo=dest,
                      commit_message=f"{tag} {oracle} adapter")
# also push the payoff matrix + state so the meta-game is reproducible
for extra in ("state.json",):
    p = os.path.join(run, extra)
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, repo_id=repo, path_in_repo=f"{tag}/{extra}")
print("  done:", len(ckpts), "adapters ->", repo)
PY
echo "[push] done. Share the repo hf.co/$HF_PUSH_REPO with us."
