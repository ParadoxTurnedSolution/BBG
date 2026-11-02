#!/bin/bash
# Builds BBG.app: a small app that starts BBG without a Terminal window.
#
#     bash macos/make-app.sh BBG_HOME DEST_DIR
#
# BBG_HOME is BBG's installed folder (with .venv); DEST_DIR receives BBG.app,
# replacing any older one. install.sh runs this. The app is built on the Mac
# it runs on, so macOS doesn't treat it as a download to be checked.
# Prints the app's path.

set -euo pipefail

BBG_HOME="$1"
DEST="$2"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

VERSION="$("$BBG_HOME/.venv/bin/python" -c 'import bbg; print(bbg.__version__)' 2>/dev/null || echo 0.0.0)"

mkdir -p "$DEST"
TMP="$(mktemp -d "$DEST/.BBG.app.XXXXXX")"
trap 'rm -rf "$TMP"' ERR
mkdir -p "$TMP/Contents/MacOS" "$TMP/Contents/Resources"
sed "s/@VERSION@/$VERSION/g" "$SRC/Info.plist" >"$TMP/Contents/Info.plist"
printf 'APPL????' >"$TMP/Contents/PkgInfo"
cp "$SRC/launcher.sh" "$TMP/Contents/MacOS/BBG"
chmod 755 "$TMP/Contents/MacOS/BBG"
cp "$SRC/BBG.icns" "$TMP/Contents/Resources/BBG.icns"
# Python's own app binary, copied into the bundle so macOS counts BBG's
# process as BBG (menu bar name, Dock icon) rather than as "Python". The
# launcher checks it works before using it, and falls back to the
# environment's python otherwise.
PYAPP="$("$BBG_HOME/.venv/bin/python" -c 'import os, sys
p = os.path.join(sys.base_prefix, "Resources", "Python.app", "Contents", "MacOS", "Python")
print(p if os.path.isfile(p) else "")' 2>/dev/null || true)"
if [ -n "$PYAPP" ]; then
  cp "$PYAPP" "$TMP/Contents/MacOS/BBG-python"
  chmod 755 "$TMP/Contents/MacOS/BBG-python"
fi
printf '%s' "$BBG_HOME" >"$TMP/Contents/Resources/bbg-home"
chmod 755 "$TMP"

# Swap in the new app. A running copy keeps working: its files stay readable
# until it exits.
rm -rf "$DEST/BBG.app"
mv "$TMP" "$DEST/BBG.app"
touch "$DEST/BBG.app" # so Finder notices the new icon

# Tell Launch Services at once, so Spotlight and Launchpad find it (macOS only).
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
if [ -x "$LSREGISTER" ]; then
  "$LSREGISTER" -f "$DEST/BBG.app" >/dev/null 2>&1 || true
fi
echo "$DEST/BBG.app"
