import json
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from bbg.export import check_format, export
from bbg.selection import Selection
from bbg.tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]


def _tracked(p, be):
    sel = Selection(p, be)
    sel.find(0, "ball")
    sel.find(0, "box")
    sel.close()
    Tracker(p, be).run()
    return p


def test_export_alpha_matches_ground_truth(project, backend, clip, tmp_path):
    p = _tracked(project, backend)
    for o in p.objects.values():
        o.keep = o.label.startswith("box")  # keep only the box
    p.edges.feather = 0
    p.edges.temporal = 0
    out = tmp_path / "box.mov"
    rep = export(p, str(out), "prores4444")
    assert rep.frames == 60 and not rep.warnings
    with av.open(str(out)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i != 25:
                continue
            alpha = frame.to_ndarray(format="rgba")[..., 3]
            gt = cv2.imread(str(clip[1] / "box" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            inter = ((alpha > 127) & (gt > 127)).sum()
            union = ((alpha > 127) | (gt > 127)).sum()
            assert inter / union > 0.95


def test_flattened_needs_background(project, backend, tmp_path):
    p = _tracked(project, backend)
    with pytest.raises(ValueError):
        check_format(p, "h264")
    p.background.mode = "colour"
    rep = export(p, str(tmp_path / "flat.mp4"), "h264")
    assert rep.frames == 60


def test_untracked_warning(project, backend, tmp_path):
    sel = Selection(project, backend)
    sel.find(0, "ball")
    sel.close()
    rep = export(project, str(tmp_path / "x.mov"), "matte")
    assert rep.untracked_frames == 59 and rep.warnings


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "bbg.cli", *args], capture_output=True, text=True,
                          cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})


def test_cli_track_and_export(clip, tmp_path):
    proj = tmp_path / "c.bbg"
    r = _cli("track", str(clip[0]), "--project", str(proj), "--backend", "synthetic",
             "--phrase", "ball", "--click", "544,180@0", "--report", str(tmp_path / "r.json"))
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["track"]["frames"] >= 60 and len(rep["objects"]) >= 2
    r = _cli("export", str(proj), str(tmp_path / "o.webm"), "--format", "webm", "--keep", "ball 1")
    assert r.returncode == 0, r.stderr
    r = _cli("export", str(proj), str(tmp_path / "o.mp4"), "--format", "h264")
    assert r.returncode == 2 and "choose a background" in r.stderr


def test_spike_dry_run(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "spike" / "run_spike.py"), "--dry-run",
                        "--out", str(tmp_path / "out"), "--workdir", str(tmp_path / "w")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["passed"] and len(summary["runs"]) == 2
    run = json.loads((tmp_path / "out" / "synthetic__synthetic__balanced.json").read_text())
    assert run["quality_metrics"]["mean_JF"] > 0.9
