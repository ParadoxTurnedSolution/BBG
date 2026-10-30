import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A 60-frame synthetic clip with ground truth: (video path, gt folder)."""
    from bbg.synthetic import make_clip

    d = tmp_path_factory.mktemp("clip")
    path = make_clip(d / "clip.mp4", n_frames=60, gt_dir=d / "gt")
    return path, d / "gt"


@pytest.fixture
def project(clip, tmp_path):
    from bbg.project import Project

    p = Project.create(tmp_path / "p.bbg", clip[0])
    p.backend = "synthetic"
    return p


@pytest.fixture
def backend():
    from bbg.backends.synthetic import SyntheticBackend

    return SyntheticBackend()
