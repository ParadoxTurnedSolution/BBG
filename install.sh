#!/bin/bash
# BBG installer for Apple Silicon Macs.
#
# Run it from Terminal (the README says how), straight from the download:
#     bash ~/Downloads/BBG-main/install.sh
#
# It checks the Mac, moves BBG to ~/BBG (moving any older copy aside), makes
# BBG its own Python environment inside that folder (.venv, so nothing is
# installed system-wide), installs BBG into it, and offers to download the
# SAM 3.1 model. Installing and updating are the same command, and running it
# again from ~/BBG is safe.
#
# For testing elsewhere: BBG_SKIP_MAC_CHECK=1 skips the Mac checks,
# BBG_SKIP_MODEL=1 skips the model download, BBG_EXTRAS overrides "[sam3,gui]",
# BBG_HOME sets where BBG goes (default ~/BBG on a Mac; off when skipping checks).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() {
  printf '\n\033[31m✗ %s\033[0m\n' "$1"
  shift
  for line in "$@"; do printf '  %s\n' "$line"; done
  printf '\n'
  exit 1
}
trap 'printf "\n\033[31m✗ Something went wrong (see the messages above).\033[0m\n  Check the internet connection and run the same command again.\n  If it fails again, send the messages above to whoever looks after BBG.\n\n"' ERR

MAC_CHECK=1
[ "${BBG_SKIP_MAC_CHECK:-}" = "1" ] && MAC_CHECK=0
EXTRAS="${BBG_EXTRAS:-[sam3,gui]}"

# -- 1. This Mac ---------------------------------------------------------------

bold "1/4  Checking this Mac"
if [ "$MAC_CHECK" = "1" ]; then
  [ "$(uname -s)" = "Darwin" ] || fail "This isn't a Mac." "BBG runs on Apple Silicon Macs only."
  # hw.optional.arm64 is 1 on Apple Silicon even when Terminal runs under Rosetta.
  [ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" = "1" ] ||
    fail "This Mac doesn't have an Apple chip (M1 or later)." \
      "BBG needs Apple's MLX, which only runs on Apple Silicon. Intel Macs can't run it."
  if [ "$(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
    fail "Terminal is running under Rosetta." \
      "Quit Terminal. In Finder, go to Applications → Utilities, select Terminal," \
      "press ⌘I and untick \"Open using Rosetta\". Then open Terminal and run this again."
  fi
  macos="$(sw_vers -productVersion)"
  if [ "${macos%%.*}" -lt 14 ]; then
    fail "macOS $macos is too old: BBG needs macOS 14 (Sonoma) or later." \
      "Update it in Apple menu → System Settings → General → Software Update."
  fi
  ok "Apple Silicon, macOS $macos"
  mem_gb=$(($(sysctl -n hw.memsize) / 1073741824))
  if [ "$mem_gb" -lt 16 ]; then
    warn "This Mac has ${mem_gb} GB of memory. BBG is designed for 16 GB or more: it may be slow."
  else
    ok "${mem_gb} GB of memory"
  fi
fi

# -- 2. Python ------------------------------------------------------------------

bold "2/4  Looking for Python 3.11 or newer"
want_arm=1
[ "$MAC_CHECK" = "0" ] && want_arm=0
python_ok() {
  "$1" -c "import sys, platform
ok = sys.version_info >= (3, 11) and ($want_arm == 0 or platform.machine() == 'arm64')
sys.exit(0 if ok else 1)" 2>/dev/null
}
PY=""
for name in python3.14 python3.13 python3.12 python3.11 python3; do
  for candidate in "$(command -v "$name" 2>/dev/null || true)" \
    /Library/Frameworks/Python.framework/Versions/3.1[1-9]/bin/"$name" \
    /opt/homebrew/bin/"$name"; do
    # /usr/bin/python3 is Apple's old 3.9 (or a stub that pops up an install
    # dialog): never use it.
    [ -n "$candidate" ] && [ -x "$candidate" ] && [ "$candidate" != "/usr/bin/python3" ] || continue
    if python_ok "$candidate"; then
      PY="$candidate"
      break 2
    fi
  done
done
if [ -z "$PY" ]; then
  [ "$MAC_CHECK" = "1" ] && open "https://www.python.org/downloads/macos/" 2>/dev/null || true
  fail "Python 3.11 or newer isn't installed." \
    "1. On the page that just opened (python.org), download the latest Python 3 for macOS." \
    "2. Open the downloaded file and click Continue / Agree / Install until it's done." \
    "3. Run this again:  bash \"$HERE/install.sh\""
fi
ok "$("$PY" --version) at $PY"

# -- Into place -----------------------------------------------------------------
# BBG lives in ~/BBG. Run from anywhere else (the unzipped download), this
# script moves its folder there; if an older BBG is there, that is moved
# aside first, so installing and updating are the same command.

TARGET="${BBG_HOME:-}"
[ -z "$TARGET" ] && [ "$MAC_CHECK" = "1" ] && TARGET="$HOME/BBG"
if [ -n "$TARGET" ] && [ "$HERE" != "$TARGET" ]; then
  bold "Moving BBG to $TARGET"
  if [ -e "$TARGET" ]; then
    if [ -f "$TARGET/install.sh" ] && [ -d "$TARGET/src/bbg" ]; then
      OLD="$TARGET-old-$(date +%Y%m%d-%H%M%S)"
      mv "$TARGET" "$OLD"
      ok "Your previous BBG was moved to $OLD (drag it to the Trash once this works)"
    else
      fail "There is already a folder called $(basename "$TARGET") in your home folder, and it isn't BBG." \
        "Rename or move that folder, then run this again:" \
        "    bash \"$HERE/install.sh\""
    fi
  fi
  mv "$HERE" "$TARGET"
  cd "$TARGET"
  HERE="$(pwd -P)"
  ok "BBG is now in $HERE"
fi

# -- 3. BBG ---------------------------------------------------------------------

bold "3/4  Installing BBG and its libraries (a few minutes, about 1 GB)"
# An environment only works where it was made: .venv/.bbg-home records where.
if [ -d .venv ] && { [ "$(cat .venv/.bbg-home 2>/dev/null)" != "$HERE" ] ||
  ! .venv/bin/python -c "import sys" 2>/dev/null; }; then
  warn "BBG's folder moved, or Python changed, since the last install: making a fresh environment"
  rm -rf .venv
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade --quiet pip
.venv/bin/python -m pip install --upgrade -e ".$EXTRAS"
echo "$HERE" >.venv/.bbg-home
chmod +x "$HERE/BBG.command"
if [ "$MAC_CHECK" = "1" ]; then
  # Files from a downloaded ZIP are "quarantined", and macOS refuses to run a
  # double-clicked script from it. This folder is now trusted.
  xattr -dr com.apple.quarantine "$HERE" 2>/dev/null || true
fi
ok "BBG $(.venv/bin/python -c 'import bbg; print(bbg.__version__)') installed in $HERE/.venv"

# -- 4. The model -----------------------------------------------------------------

bold "4/4  The SAM 3.1 model"
if [ "${BBG_SKIP_MODEL:-}" = "1" ]; then
  warn "Skipped (BBG_SKIP_MODEL=1)"
elif .venv/bin/python -c "from bbg.backends.sam3_mlx import model_cached; import sys; sys.exit(0 if model_cached() else 1)" 2>/dev/null; then
  ok "Already downloaded"
else
  echo "  BBG needs Meta's SAM 3.1 model (about 1.8 GB, downloaded once)."
  .venv/bin/bbg-cli download-model ||
    warn "The model wasn't downloaded. BBG will offer to download it when it starts."
fi

trap - ERR
bold "Done: BBG is installed."
echo "  To start it, double-click BBG.command in this folder:"
echo "    $HERE"
if [ "$MAC_CHECK" = "1" ]; then
  open "$HERE" 2>/dev/null || true
fi
echo
