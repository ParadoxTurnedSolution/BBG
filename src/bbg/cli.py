"""bbg-cli: track and export without the GUI.

    bbg-cli info clip.mov
    bbg-cli track clip.mov --project clip.bbg --phrase dog --click 640,360@0
    bbg-cli export clip.bbg out.mov --format prores4444
    bbg-cli demo /tmp/bbg-demo            # synthetic clip, no model needed
    bbg-cli download-model                # SAM 3.1 weights (SAM License)

Click and box coordinates are in source pixels, optionally `@frame`.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path


def _point(s: str) -> tuple[list[float], int]:
    coords, _, frame = s.partition("@")
    return [float(v) for v in coords.split(",")], int(frame or 0)


def _background(s: str):
    from .composite import BackgroundSettings

    mode, _, arg = s.partition(":")
    if mode == "transparent":
        return BackgroundSettings("transparent")
    if mode in ("colour", "color"):
        h = (arg or "#00b140").lstrip("#")
        return BackgroundSettings("colour", colour=tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    if mode == "image":
        return BackgroundSettings("image", image_path=arg)
    if mode == "video":
        return BackgroundSettings("video", video_path=arg)
    if mode == "blur":
        return BackgroundSettings("blur", blur=float(arg) if arg else 0.02)
    raise argparse.ArgumentTypeError(f"bad background {s!r}")


def _progress(prefix: str):
    def show(p):
        eta = "" if p.eta == float("inf") else f" · ~{p.eta:.0f}s left"
        print(f"\r{prefix} {p.done}/{p.total} frames · {p.fps:.1f} fps{eta}   ", end="", file=sys.stderr)
    return show


def cmd_info(a) -> int:
    from .video.reader import probe

    print(json.dumps(probe(a.video).to_dict(), indent=1))
    return 0


def _make_backend(name: str, a):
    from .backends import create_backend

    kwargs = {}
    if name == "sam3":
        kwargs = {"memory_limit_gb": a.memory_limit_gb, "model_id": a.model}
    be = create_backend(name, **kwargs)
    be.load()
    return be


def cmd_track(a) -> int:
    from .memory import MemoryMonitor
    from .project import Project
    from .selection import Selection
    from .tracking import Tracker, find_flags

    root = Path(a.project or Path(a.video).with_suffix(".bbg"))
    p = Project.open(root) if (root / "project.json").exists() else Project.create(root, a.video)
    p.backend = a.backend
    p.track.quality = a.quality
    if a.threshold is not None:
        p.track.detect_threshold = a.threshold
    if a.redetect_every is not None:
        p.track.redetect_every = a.redetect_every
    sx = p.work_size[0] / p.info.width
    sy = p.work_size[1] / p.info.height
    mon = MemoryMonitor().start()
    be = _make_backend(a.backend, a)
    sel = Selection(p, be)
    for phrase in a.phrase:
        made = sel.find(a.frame, phrase)
        print(f"'{phrase}' on frame {a.frame}: {len(made)} object(s)", file=sys.stderr)
    for c in a.click:
        (x, y), f = _point(c)
        obj = sel.click(f, x * sx, y * sy)
        print(f"click {x:.0f},{y:.0f} on frame {f}: {obj.label}", file=sys.stderr)
    for b in a.box:
        coords, f = _point(b)
        x1, y1, x2, y2 = coords
        obj = sel.box(f, (x1 * sx, y1 * sy, x2 * sx, y2 * sy))
        print(f"box on frame {f}: {obj.label}", file=sys.stderr)
    sel.close()
    p.save()
    if not p.objects:
        print("nothing selected: give --phrase, --click or --box", file=sys.stderr)
        return 2
    cancel = threading.Event()
    try:
        rep = Tracker(p, be, on_progress=_progress("tracking"), cancel=cancel).run()
    except KeyboardInterrupt:
        cancel.set()
        p.save()
        return 130
    finally:
        mon.stop()
    print(file=sys.stderr)
    flags = find_flags(p)
    print(f"tracked {rep.frames} frames in {rep.seconds:.1f}s ({rep.fps:.1f} fps); "
          f"{len(p.objects)} object(s); {len(flags)} flag(s); "
          f"peak memory {mon.peak_footprint / 1e9:.2f} GB", file=sys.stderr)
    if a.report:
        out = {"track": rep.to_dict(), "memory": mon.to_dict(), "backend": be.memory_stats(),
               "objects": [o.to_dict() for o in p.objects.values()],
               "flags": [f.__dict__ for f in flags]}
        Path(a.report).write_text(json.dumps(out, indent=1))
    return 0


def cmd_export(a) -> int:
    from .export import export
    from .project import Project

    p = Project.open(a.project)
    if a.background:
        p.background = a.background
    for attr in ("grow", "feather", "temporal"):
        v = getattr(a, attr)
        if v is not None:
            setattr(p.edges, attr, v)
    if a.keep is not None:
        keep = set(a.keep)
        for o in p.objects.values():
            o.keep = o.label in keep or str(o.id) in keep
    p.save()

    def show(done, total):
        print(f"\rexporting {done}/{total}", end="", file=sys.stderr)

    rep = export(p, a.out, a.format, on_progress=show)
    print(file=sys.stderr)
    for w in rep.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if rep.audio_error:
        print(f"warning: audio not carried over: {rep.audio_error}", file=sys.stderr)
    print(f"wrote {rep.path}: {rep.frames} frames in {rep.seconds:.1f}s", file=sys.stderr)
    return 0


def cmd_demo(a) -> int:
    from .export import export
    from .project import Project
    from .selection import Selection
    from .synthetic import make_clip
    from .tracking import Tracker

    out = Path(a.outdir)
    clip = make_clip(out / "demo.mp4", n_frames=a.frames)
    p = Project.create(out / "demo.bbg", clip, overwrite=True)
    p.backend = "synthetic"
    be = _make_backend("synthetic", a)
    sel = Selection(p, be)
    sel.find(0, "ball")
    sel.find(0, "box")
    p.phrases.append("dog")
    sel.close()
    rep = Tracker(p, be, on_progress=_progress("tracking")).run()
    print(file=sys.stderr)
    print(f"tracked {rep.frames} frames; objects: {[o.label for o in p.objects.values()]}", file=sys.stderr)
    for o in p.objects.values():
        o.keep = o.label.startswith(("ball", "dog"))
    rep = export(p, str(out / "demo_prores4444.mov"), "prores4444")
    print(f"wrote {rep.path} (box removed, ball and dog kept)", file=sys.stderr)
    return 0


def cmd_download(a) -> int:
    from .backends.sam3_mlx import MODEL_ID, SAM_LICENSE_URL, download

    print(f"SAM 3.1 weights ({a.model or MODEL_ID}) are Meta's, under the SAM License:\n"
          f"  {SAM_LICENSE_URL}\n"
          "It permits use and redistribution, but forbids military/warfare and other\n"
          "trade-controlled end uses, and Meta may change its terms.", file=sys.stderr)
    if not a.yes and input("Accept and download? [y/N] ").strip().lower() != "y":
        return 1
    print(download(a.model or MODEL_ID))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bbg-cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="describe a video")
    s.add_argument("video")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("track", help="select objects and track them")
    s.add_argument("video")
    s.add_argument("--project", help="project folder (default: <video>.bbg)")
    s.add_argument("--phrase", action="append", default=[], help="select every match of a phrase")
    s.add_argument("--click", action="append", default=[], help="X,Y[@FRAME] in source pixels")
    s.add_argument("--box", action="append", default=[], help="X1,Y1,X2,Y2[@FRAME]")
    s.add_argument("--frame", type=int, default=0, help="frame for --phrase (default 0)")
    s.add_argument("--backend", default="sam3", choices=["sam3", "synthetic"])
    s.add_argument("--model", default=None, help="SAM 3.1 weights (Hugging Face id or path)")
    s.add_argument("--quality", default="balanced", choices=["balanced", "fast"])
    s.add_argument("--threshold", type=float, default=None, help="phrase detection threshold")
    s.add_argument("--redetect-every", type=int, default=None)
    s.add_argument("--memory-limit-gb", type=float, default=6.0)
    s.add_argument("--report", help="write a JSON report here")
    s.set_defaults(fn=cmd_track)

    s = sub.add_parser("export", help="render a tracked project")
    s.add_argument("project")
    s.add_argument("out")
    s.add_argument("--format", default="prores4444",
                   choices=["prores4444", "hevc_alpha", "png", "matte", "h264", "hevc", "webm"])
    s.add_argument("--background", type=_background,
                   help="transparent | colour:#RRGGBB | image:PATH | video:PATH | blur[:SIGMA]")
    s.add_argument("--keep", nargs="*", help="labels or ids to keep (default: as saved)")
    s.add_argument("--grow", type=float)
    s.add_argument("--feather", type=float)
    s.add_argument("--temporal", type=float)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("demo", help="synthetic end-to-end run (no model needed)")
    s.add_argument("outdir")
    s.add_argument("--frames", type=int, default=90)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("download-model", help="download the SAM 3.1 weights")
    s.add_argument("--model", default=None)
    s.add_argument("--yes", action="store_true", help="accept the SAM License without asking")
    s.set_defaults(fn=cmd_download)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        print(f"bbg-cli: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
