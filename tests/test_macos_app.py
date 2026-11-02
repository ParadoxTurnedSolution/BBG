"""BBG.app: the icon, the bundle make-app.sh builds, and the launcher's paths.

These run anywhere: the launcher is a bash script, and osascript (macOS's
dialog tool) is replaced by a stub that records what would have been shown.
What only a Mac can check: that macOS opens the bundle, and that the bundled
Python binary makes the menu bar say "BBG".
"""

import os
import plistlib
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_icns_has_every_size():
    data = (ROOT / "macos" / "BBG.icns").read_bytes()
    assert data[:4] == b"icns" and struct.unpack(">I", data[4:8])[0] == len(data)
    sizes, pos = {}, 8
    while pos < len(data):
        kind, length = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        img = cv2.imdecode(np.frombuffer(data[pos + 8:pos + length], np.uint8), cv2.IMREAD_UNCHANGED)
        sizes[kind] = img.shape[:2]
        pos += length
    assert sizes == {b"ic04": (16, 16), b"ic05": (32, 32), b"ic07": (128, 128), b"ic08": (256, 256),
                     b"ic09": (512, 512), b"ic10": (1024, 1024), b"ic11": (32, 32), b"ic12": (64, 64),
                     b"ic13": (256, 256), b"ic14": (512, 512)}
    assert (ROOT / "src" / "bbg" / "gui" / "resources" / "bbg.png").exists()


def _make_app(home: Path, dest: Path) -> Path:
    out = subprocess.run(["bash", str(ROOT / "macos" / "make-app.sh"), str(home), str(dest)],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip().splitlines()[-1])


def test_bundle_structure(tmp_path):
    home = tmp_path / "BBG"
    (home / ".venv" / "bin").mkdir(parents=True)
    py = home / ".venv" / "bin" / "python"
    py.write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    app = _make_app(home, tmp_path / "Applications")
    assert app == tmp_path / "Applications" / "BBG.app"
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "BBG" and info["CFBundleIconFile"] == "BBG"
    assert info["CFBundlePackageType"] == "APPL" and info["CFBundleShortVersionString"] == "0.1.0"
    launcher = app / "Contents" / "MacOS" / "BBG"
    assert os.access(launcher, os.X_OK)
    assert (app / "Contents" / "Resources" / "bbg-home").read_text() == str(home)
    assert (app / "Contents" / "Resources" / "BBG.icns").stat().st_size > 10_000
    # Rebuilding replaces the app and leaves no temporary folders behind.
    _make_app(home, tmp_path / "Applications")
    assert sorted(p.name for p in (tmp_path / "Applications").iterdir()) == ["BBG.app"]


@pytest.fixture
def fake(tmp_path):
    """An installed-looking BBG folder with stub python, installer and osascript."""
    home = (tmp_path / "BBG").resolve()
    (home / ".venv" / "bin").mkdir(parents=True)
    rec = tmp_path / "record.txt"
    python = home / ".venv" / "bin" / "python"
    python.write_text(f"""#!/bin/bash
[ "$1" = "-c" ] && exit 0
echo "python $* LOG=$BBG_LOG" >>"{rec}"
""")
    python.chmod(0o755)
    (home / ".venv" / ".bbg-home").write_text(str(home) + "\n")
    installer = home / "install.sh"
    installer.write_text(f"""#!/bin/bash
echo "install HOME=$BBG_HOME SKIP_MODEL=$BBG_SKIP_MODEL NO_OPEN=$BBG_NO_OPEN" >>"{rec}"
[ -f "{tmp_path}/fail-repair" ] && exit 1
echo "$BBG_HOME" >"$BBG_HOME/.venv/.bbg-home"
""")
    osa = tmp_path / "osascript"
    osa.write_text(f"""#!/bin/bash
shift
echo "dialog $*" >>"{rec}"
""")
    osa.chmod(0o755)
    app = _make_app(home, tmp_path / "Applications")
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "BBG_OSASCRIPT": str(osa)}

    def run():
        subprocess.run([str(app / "Contents" / "MacOS" / "BBG")], env=env, check=False, timeout=60)
        return rec.read_text() if rec.exists() else ""

    return home, tmp_path, run


def test_launcher_starts_bbg(fake):
    home, tmp, run = fake
    out = run()
    assert f"python -m bbg LOG={tmp}/Library/Logs/BBG/bbg.log" in out
    assert "dialog" not in out and "install" not in out
    assert "=== BBG started" in (tmp / "Library" / "Logs" / "BBG" / "bbg.log").read_text()


def test_launcher_repairs_after_a_move(fake):
    home, tmp, run = fake
    (home / ".venv" / ".bbg-home").write_text("/somewhere/else\n")
    out = run()
    assert f"install HOME={home} SKIP_MODEL=1 NO_OPEN=1" in out
    assert "dialog Repairing BBG" in out  # a notification, not a question
    assert "python -m bbg" in out  # and then BBG opens


def test_launcher_reports_a_failed_repair(fake):
    home, tmp, run = fake
    (home / ".venv" / ".bbg-home").write_text("/somewhere/else\n")
    (tmp / "fail-repair").touch()
    out = run()
    assert "dialog BBG couldn't repair itself" in out and "python -m bbg" not in out


def test_launcher_reports_missing_folder(fake):
    home, tmp, run = fake
    home.rename(tmp / "moved")
    out = run()
    assert "dialog BBG can't find its files" in out and "python" not in out
