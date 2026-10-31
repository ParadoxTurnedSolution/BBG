#!/bin/bash
# Double-click this file to start BBG. (install.sh must have run once first.)
# The Terminal window that opens with it shows progress and any errors:
# leave it open while you use BBG, and close it after you quit.

cd "$(dirname "$0")" || exit 1

stop() {
  echo
  for line in "$@"; do echo "$line"; done
  echo
  read -r -p "Press Return to close this window. "
  exit 1
}

if [ ! -e .venv/bin/bbg ]; then
  stop "BBG isn't installed yet. In Terminal, run:" "" "    bash \"$(pwd)/install.sh\""
fi
# The environment only works where it was installed (install.sh records
# where). If the folder was moved or renamed since, re-run the installer here.
if [ "$(cat .venv/.bbg-home 2>/dev/null)" != "$(pwd -P)" ]; then
  echo "BBG's folder was moved or renamed after it was installed, so it needs a"
  echo "quick repair (a few minutes; nothing is downloaded again except libraries)."
  read -r -p "Repair it now? [Y/n] " answer
  case "$answer" in [nN]*) exit 1 ;; esac
  BBG_HOME="$(pwd -P)" bash ./install.sh || stop "The repair didn't finish: see the messages above."
fi
exec .venv/bin/bbg "$@"
