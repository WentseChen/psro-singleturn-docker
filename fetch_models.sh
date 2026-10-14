#!/usr/bin/env bash
# Idempotent setup. Pulls ONLY the public judge models (no policy weights are shipped or baked).
# pi0 -- the base policy the co-worker trains from scratch on -- is NOT fetched here: mount it at
# $BASE_MODEL_PATH. Cheap to re-run on every container start (skips what is already present).
#
# IMPORTANT: psro.env pins each judge by an EXACT snapshot commit (used as a literal filesystem
# path), so we download that exact revision -- not "main" -- or the pinned path would not exist.
set -eu -o pipefail
source /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/psro.env
PY=$SSP_PYTHON
export HF_HOME=$SSP_HF_HOME

have () { [ -n "$(ls "$1"/*.safetensors 2>/dev/null | head -1)" ]; }

# repo <TAB> pinned-revision (must match the snapshot hashes in psro.env)
JUDGES="\
allenai/wildguard	cbba4823f3e8020e5a74a5e29bf85072def6f2ff
Qwen/Qwen3.5-9B	c202236235762e1c871ad0ccb60c8ee5ba337b9a
mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated	368c8ed94ce4c986e7b9ca5c159651ef753908ce"

echo "[fetch] judge models into $SSP_HF_HOME (pinned revisions) ..."
while IFS=$'\t' read -r repo rev; do
  [ -z "$repo" ] && continue
  dir="$SSP_HF_HOME/hub/models--${repo//\//--}/snapshots/$rev"
  if have "$dir" 2>/dev/null; then echo "  have $repo @ $rev"; continue; fi
  echo "  pull $repo @ $rev"
  $PY - "$repo" "$rev" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2],
                  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model"],
                  max_workers=8)
PY
  [ -d "$dir" ] || { echo "FATAL: downloaded $repo but $dir is missing (revision/path mismatch)"; exit 5; }
done <<< "$JUDGES"

# Base policy = the public abliterated Llama-3.1-8B, fetched above (off-the-shelf; nothing of ours).
if have "$BASE_MODEL_PATH"; then
  echo "[fetch] base policy (off-the-shelf abliterated Llama) present at $BASE_MODEL_PATH"
else
  echo "FATAL: base policy not found at $BASE_MODEL_PATH"
  echo "       This should be the abliterated Llama snapshot the judge loop just pulled;"
  echo "       if you changed BASE_MODEL_PATH, point it at a fetched open-source model."
  exit 4
fi
echo "[fetch] done: wildguard, Qwen3.5-9B (faithful judge), abliterated (base + topic-judge); base=$BASE_MODEL_PATH"
