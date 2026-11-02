#!/bin/bash
# BBG.app's executable (Contents/MacOS/BBG): starts BBG with no Terminal
# window. macos/make-app.sh puts it in the bundle, with the path of BBG's
# folder in Contents/Resources/bbg-home.
#
# What would have gone to a Terminal window goes to ~/Library/Logs/BBG/bbg.log;
# anything the user must know is shown as a normal macOS alert.
# BBG_OSASCRIPT overrides osascript (tests).

BUNDLE="$(cd "$(dirname "$0")/../.." && pwd -P)"
BBG_HOME="$(cat "$BUNDLE/Contents/Resources/bbg-home" 2>/dev/null)"
LOG_DIR="$HOME/Library/Logs/BBG"
LOG="$LOG_DIR/bbg.log"
OSASCRIPT="${BBG_OSASCRIPT:-/usr/bin/osascript}"

alert() { # title, message
  "$OSASCRIPT" - "$1" "$2" >/dev/null 2>&1 <<'OSA'
on run argv
  display alert (item 1 of argv) message (item 2 of argv) as critical buttons {"OK"} default button "OK"
end run
OSA
}

notify() { # message
  "$OSASCRIPT" - "$1" >/dev/null 2>&1 <<'OSA'
on run argv
  display notification (item 1 of argv) with title "BBG"
end run
OSA
}

mkdir -p "$LOG_DIR"
size="$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null || echo 0)"
[ "$size" -gt 5000000 ] && mv -f "$LOG" "$LOG.old"
{
  echo
  echo "=== BBG started $(date)"
} >>"$LOG"

if [ -z "$BBG_HOME" ] || [ ! -f "$BBG_HOME/install.sh" ]; then
  alert "BBG can't find its files" \
    "BBG's folder (${BBG_HOME:-unknown}) has been moved, renamed or deleted. To fix this, install BBG again: follow the Install steps in BBG's instructions."
  exit 1
fi

# The environment only works where it was installed, with the Python it was
# made from. If either changed, re-run the installer quietly, then carry on.
if [ "$(cat "$BBG_HOME/.venv/.bbg-home" 2>/dev/null)" != "$BBG_HOME" ] ||
  ! "$BBG_HOME/.venv/bin/python" -c "" >/dev/null 2>&1; then
  notify "Repairing BBG after a change to this Mac. This takes a few minutes; BBG opens when it's done."
  echo "--- repairing" >>"$LOG"
  export BBG_HOME # the installer repairs this folder in place
  if ! BBG_SKIP_MODEL=1 BBG_NO_OPEN=1 /bin/bash "$BBG_HOME/install.sh" >>"$LOG" 2>&1; then
    alert "BBG couldn't repair itself" \
      "Install BBG again: follow the Install steps in BBG's instructions. (Details are in $LOG)"
    exit 1
  fi
fi

export BBG_LOG="$LOG"
cd "$BBG_HOME" || exit 1
exec >>"$LOG" 2>&1
PY="$BBG_HOME/.venv/bin/python"
# Run as BBG (see make-app.sh) when the bundled copy of Python's app binary
# works with BBG's environment; __PYVENV_LAUNCHER__ is how Python's own
# launcher tells that binary which environment to use.
APP_PY="$BUNDLE/Contents/MacOS/BBG-python"
if [ -x "$APP_PY" ] && __PYVENV_LAUNCHER__="$PY" "$APP_PY" -c "import bbg" 2>/dev/null; then
  export __PYVENV_LAUNCHER__="$PY"
  exec "$APP_PY" -m bbg "$@"
fi
exec "$PY" -m bbg "$@"
