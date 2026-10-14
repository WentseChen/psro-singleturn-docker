#!/usr/bin/env bash
# Export an already-built image for docker load on another machine.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${SSP_IMAGE:-safeselfplay:llama8b-h200-invalidmove-20260924-v5}
DEST=${1:-$REPO/output/releases/llama8b-labels-20260919/safeselfplay-llama8b-h200-labels.image.tar.gz}
if [ -e "$DEST" ]; then
  echo "Output already exists: $DEST; choose a different filename." >&2
  exit 1
fi
docker image inspect "$IMAGE" >/dev/null
mkdir -p "$(dirname "$DEST")"
TEMP=$(mktemp "${DEST}.partial.XXXXXX")
trap 'rm -f "$TEMP"' EXIT
docker image save "$IMAGE" | gzip -1 > "$TEMP"
mv "$TEMP" "$DEST"
echo "Exported Docker image: $DEST"
echo "On the target machine: docker load -i '$DEST'"
