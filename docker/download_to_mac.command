#!/bin/bash
# Run this on janus's Mac. Uses the existing ~/.ssh/config alias.
set -euo pipefail
SERVER=${SSP_SSH_HOST:-sz-4090-d02}
REMOTE=/export/pgs/wuxudong/work/selfplay/SafeSelfPlay/output/releases/llama8b-labels-20260919
DEST=${SSP_DOWNLOAD_DIR:-$HOME/Downloads/SafeSelfPlay-llama8b-labels-20260919}
mkdir -p "$DEST"
chmod 700 "$DEST"
echo "Downloading the private handoff to $DEST"
rsync -avP -e ssh "$SERVER:$REMOTE/" "$DEST/"
chmod 600 "$DEST"/*.tar.gz
echo "Download complete: $DEST"
