"""A project: one clip, its objects, their keyframes, settings and mask cache.

On disk a project is a folder `<name>.bbg/`:

    project.json        objects, keyframes, phrases, settings, video info
    anchors/            keyframe masks, one PNG per (frame, object)
    cache/masks/        tracked masks, one PNG per (object, frame)
    cache/tracks/       per-object score, area and "which keyframe wrote this
                        frame" arrays

A keyframe ("anchor") is a frame where the object's mask came from the user
(a phrase, clicks or a box), not from tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .composite import BackgroundSettings
from .masks.cache import MaskCache
from .masks.refine import EdgeSettings
from .video.reader import VideoInfo, VideoReader, work_size

FORMAT_VERSION = 1

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195),
]


@dataclass
class TrackedObject:
    id: int
    label: str
    colour: tuple[int, int, int]
    keep: bool = True
    source: str = "click"  # phrase | click | box | similar | auto
    phrase: str | None = None
    # Objects found by re-detection mid-clip are only tracked back this far:
    # had they been visible earlier, the previous detection would have found them.
    backward_limit: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["colour"] = list(self.colour)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrackedObject":
        d = dict(d)
        d["colour"] = tuple(d["colour"])
        return cls(**d)


@dataclass
class TrackSettings:
    quality: str = "balanced"  # balanced | fast (fast tracks every 2nd frame)
    detect_threshold: float = 0.4
    redetect_every: int = 15  # frames between re-detections of typed phrases; 0 = off
    new_object_iou: float = 0.3  # a detection overlapping a track more than this is not new

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackSettings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def mask_version(mask: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(mask).tobytes()).hexdigest()[:12]


class Project:
    def __init__(self, root: Path, video_path: str, info: VideoInfo):
        self.root = Path(root)
        self.video_path = video_path
        self.info = info
        self.work_size = work_size(info.width, info.height)
        self.objects: dict[int, TrackedObject] = {}
        self.anchors: dict[int, dict[int, str]] = {}  # frame -> {object: mask version}
        self.phrases: list[str] = []
        self.edges = EdgeSettings()
        self.background = BackgroundSettings()
        self.track = TrackSettings()
        self.backend = "sam3"
        # owners[obj] = list of owner keys; written[obj][frame] = index into it, -1 = none
        self.owner_tables: dict[int, list[str]] = {}
        self._written: dict[int, np.ndarray] = {}
        self._next_id = 1
        (self.root / "anchors").mkdir(parents=True, exist_ok=True)
        self.cache = MaskCache(self.root / "cache", info.n_frames, self.work_size)

    # -- create / open / save -------------------------------------------

    @classmethod
    def create(cls, root: str | Path, video_path: str | Path, overwrite: bool = False) -> "Project":
        root = Path(root)
        if root.exists():
            if not overwrite:
                raise FileExistsError(root)
            shutil.rmtree(root)
        video_path = str(Path(video_path).resolve())
        with VideoReader(video_path, hwaccel=False) as r:
            info = r.info
        p = cls(root, video_path, info)
        p.save()
        return p

    @classmethod
    def open(cls, root: str | Path) -> "Project":
        root = Path(root)
        data = json.loads((root / "project.json").read_text())
        if data.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(f"{root} was saved by a newer BBG")
        p = cls(root, data["video_path"], VideoInfo.from_dict(data["video_info"]))
        p.objects = {o["id"]: TrackedObject.from_dict(o) for o in data["objects"]}
        p.anchors = {int(f): {int(o): v for o, v in objs.items()} for f, objs in data["anchors"].items()}
        p.phrases = list(data.get("phrases", []))
        p.edges = EdgeSettings.from_dict(data.get("edges", {}))
        p.background = BackgroundSettings.from_dict(data.get("background", {}))
        p.track = TrackSettings.from_dict(data.get("track", {}))
        p.backend = data.get("backend", "sam3")
        p.owner_tables = {int(k): v for k, v in data.get("owner_tables", {}).items()}
        p._next_id = data.get("next_id", max(p.objects, default=0) + 1)
        for obj in p.objects:
            path = p.root / "cache" / "tracks" / f"{obj}.owner.npy"
            if path.exists():
                p._written[obj] = np.load(path)
        return p

    def save(self) -> None:
        self.cache.flush()
        for obj, arr in self._written.items():
            path = self.root / "cache" / "tracks" / f"{obj}.owner.npy"
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, path)
        data = {
            "format_version": FORMAT_VERSION,
            "video_path": self.video_path,
            "video_info": self.info.to_dict(),
            "objects": [o.to_dict() for o in self.objects.values()],
            "anchors": {str(f): {str(o): v for o, v in objs.items()} for f, objs in self.anchors.items()},
            "phrases": self.phrases,
            "edges": self.edges.to_dict(),
            "background": self.background.to_dict(),
            "track": self.track.to_dict(),
            "backend": self.backend,
            "owner_tables": {str(k): v for k, v in self.owner_tables.items()},
            "next_id": self._next_id,
        }
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.root / "project.json")

    # -- objects ----------------------------------------------------------

    def add_object(self, label: str, source: str = "click", phrase: str | None = None,
                   backward_limit: int | None = None) -> TrackedObject:
        oid = self._next_id
        self._next_id += 1
        obj = TrackedObject(oid, label, PALETTE[(oid - 1) % len(PALETTE)], True, source,
                            phrase, backward_limit)
        self.objects[oid] = obj
        return obj

    def next_label(self, phrase: str) -> str:
        n = sum(1 for o in self.objects.values() if o.phrase == phrase) + 1
        return f"{phrase} {n}"

    def remove_object(self, oid: int) -> None:
        self.objects.pop(oid, None)
        for frame in list(self.anchors):
            if self.anchors[frame].pop(oid, None) is not None:
                (self.root / "anchors" / f"{frame:06d}_{oid}.png").unlink(missing_ok=True)
            if not self.anchors[frame]:
                del self.anchors[frame]
        self.cache.clear(oid)
        self.owner_tables.pop(oid, None)
        self._written.pop(oid, None)
        (self.root / "cache" / "tracks" / f"{oid}.owner.npy").unlink(missing_ok=True)

    # -- keyframes --------------------------------------------------------

    def set_anchor(self, frame: int, oid: int, mask: np.ndarray) -> None:
        """Make `mask` (work size, uint8 or float 0..1) the object's keyframe on `frame`."""
        if mask.dtype != np.uint8:
            mask = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
        w, h = self.work_size
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(self.root / "anchors" / f"{frame:06d}_{oid}.png"), mask)
        self.anchors.setdefault(frame, {})[oid] = mask_version(mask)
        # The keyframe itself is shown at once, before any tracking.
        self.cache.put(oid, frame, mask, 1.0 if mask.max() > 127 else 0.0)
        self.mark_written(oid, frame, frame, self.owner_key(oid, frame, +1))

    def remove_anchor(self, frame: int, oid: int) -> None:
        objs = self.anchors.get(frame, {})
        if objs.pop(oid, None) is not None:
            (self.root / "anchors" / f"{frame:06d}_{oid}.png").unlink(missing_ok=True)
        if not objs:
            self.anchors.pop(frame, None)

    def anchor_mask(self, frame: int, oid: int) -> np.ndarray:
        m = cv2.imread(str(self.root / "anchors" / f"{frame:06d}_{oid}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise KeyError((frame, oid))
        return m

    def anchor_frames(self, oid: int) -> list[int]:
        return sorted(f for f, objs in self.anchors.items() if oid in objs)

    def owner_key(self, oid: int, start: int, direction: int) -> str:
        return f"{start}:{direction}:{self.anchors[start][oid]}"

    # -- which keyframe wrote each frame -----------------------------------

    def written(self, oid: int) -> np.ndarray:
        if oid not in self._written:
            self._written[oid] = np.full(self.info.n_frames, -1, np.int32)
        return self._written[oid]

    def _owner_index(self, oid: int, key: str) -> int:
        table = self.owner_tables.setdefault(oid, [])
        if key not in table:
            table.append(key)
        return table.index(key)

    def mark_written(self, oid: int, lo: int, hi: int, key: str) -> None:
        lo, hi = sorted((lo, hi))
        self.written(oid)[lo:hi + 1] = self._owner_index(oid, key)

    def written_keys(self, oid: int) -> list[str | None]:
        table = self.owner_tables.get(oid, [])
        return [table[i] if i >= 0 else None for i in self.written(oid)]

    # -- convenience --------------------------------------------------------

    def kept_masks(self, frame: int) -> list[np.ndarray]:
        return [m for o in self.objects.values() if o.keep
                for m in [self.cache.get(o.id, frame)] if m is not None]

    def reader(self, hwaccel: bool = True) -> VideoReader:
        return VideoReader(self.video_path, hwaccel=hwaccel)
