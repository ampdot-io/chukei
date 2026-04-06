#!/bin/bash
set -euo pipefail

INSTALL_DIR="$HOME"
BINARY_NAME="koboldcpp"
ASSET_NAME="koboldcpp-mac-arm64"
REPO="LostRuins/koboldcpp"

echo "Checking latest release..."
LATEST=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
echo "Latest release: $LATEST"

DOWNLOAD_URL="https://github.com/$REPO/releases/download/$LATEST/$ASSET_NAME"
TMP="$INSTALL_DIR/$ASSET_NAME.tmp"

echo "Downloading $DOWNLOAD_URL..."
curl -L -o "$TMP" "$DOWNLOAD_URL"
chmod +x "$TMP"

# Keep one backup
mv -f "$INSTALL_DIR/$ASSET_NAME" "$INSTALL_DIR/$ASSET_NAME.old" 2>/dev/null || true

mv "$TMP" "$INSTALL_DIR/$ASSET_NAME"
cp "$INSTALL_DIR/$ASSET_NAME" "$INSTALL_DIR/$BINARY_NAME"
chmod +x "$INSTALL_DIR/$BINARY_NAME"

echo "Updated koboldcpp to $LATEST"
