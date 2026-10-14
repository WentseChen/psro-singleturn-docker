#!/bin/bash
# Download only the source build handoff using janus's existing SSH alias.
set -euo pipefail
SERVER=${SSP_SSH_HOST:-sz-4090-d02}
REMOTE=/export/pgs/wuxudong/work/selfplay/SafeSelfPlay/output/releases/llama8b-source-20260919
DEST=${SSP_DOWNLOAD_DIR:-$HOME/Downloads/SafeSelfPlay-llama8b-source-20260919}
mkdir -p "$DEST"
chmod 700 "$DEST"
rsync -avP -e ssh "$SERVER:$REMOTE/" "$DEST/"
chmod 600 "$DEST"/*.tar.gz
(cd "$DEST" && shasum -a 256 -c SHA256SUMS)
echo "Source build package downloaded and verified: $DEST"
