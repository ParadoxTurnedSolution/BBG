"""Phase 0 spike: measure SAM 3.1 on this Mac against the scope's targets.

    python spike/run_spike.py spike/manifest.json --out spike/reports
    python spike/run_spike.py --dry-run --out /tmp/spike   # synthetic, no model

For every clip × model × quality in the manifest it selects the objects
(phrases and clicks), tracks the whole clip, and records:

- memory: peak physical footprint (what Activity Monitor shows), MLX peak,
  footprint growth per minute after warm-up, swap growth;
- speed: model load, phrase-to-objects and click-to-mask latency, tracking fps
  and where the time went (decode / backbone / tracker / detection / writes);
- quality, where ground truth exists: J&F per object, phrase recall and
  precision on the selection frame, flicker (dtSSD with consecutive ground
  truth, warp error without).

Each run writes a JSON report with a top-level `passed` (the memory and speed
targets from SCOPE.md §7–8; quality is reported, not judged, since Phase 0
sets that bar), and the script exits non-zero if any run failed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bbg import __version__  # noqa: E402
from bbg.backends import create_backend  # noqa: E402
from bbg.memory import GB, MemoryMonitor  # noqa: E402
from bbg.metrics import WarpError, dtssd, jf, phrase_recall_precision  # noqa: E402
from bbg.project import Project  # noqa: E402
from bbg.selection import Selection  # noqa: E402
from bbg.tracking import Tracker, find_flags  # noqa: E402

TARGETS = {
    "peak_footprint_gb": 6.0,  # SCOPE §7: app peak ≤ 6 GB
    "swap_growth_gb": 0.25,  # no swap growth (allowing noise)
    "growth_gb_per_min": 0.05,  # memory flat over a long clip
    "click_s": 0.3,  # SCOPE §8: click to mask
    "phrase_s": 1.0,  # phrase to objects
    "fps_balanced_1080p": 5.0,
    "fps_fast_1080p": 12.0,
}


def machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "bbg": __version__}
    if sys.platform == "darwin":
        def sysctl(k):
            try:
                return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
            except Exception:
                return None
        info.update(chip=sysctl("machdep.cpu.brand_string"),
                    memory_gb=int(sysctl("hw.memsize") or 0) / GB,
                    macos=platform.mac_ver()[0])
    else:
        try:
            kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
            info["memory_gb"] = kb * 1024 / GB
        except Exception:
            pass
    for mod in ("mlx", "mlx_vlm", "av", "cv2", "numpy"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", None) or getattr(getattr(m, "core", None), "__version__", None)
        except Exception:
            info[f"{mod}_version"] = None
    return info


def _gt_frames(gt_dir: Path, name: str) -> dict[int, Path]:
    d = gt_dir / name
    return {int(p.stem): p for p in sorted(d.glob("*.png"))} if d.exists() else {}


def _read_gt(path: Path, size) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)


def quality(p: Project, clip: dict, base: Path, selection_frame: int, found: dict) -> dict:
    out: dict = {}
    gt_dir = clip.get("gt_dir")
    size = p.work_size
    if gt_dir:
        gt_dir = (base / gt_dir).resolve()
        names = sorted(d.name for d in gt_dir.iterdir() if d.is_dir())
        per_obj = {}
        for name in names:
            frames = _gt_frames(gt_dir, name)
            if not frames:
                continue
            # The tracked object that best matches this ground-truth object.
            best, best_j, best_scores = None, -1.0, None
            for o in p.objects.values():
                js, fs = [], []
                for f, path in frames.items():
                    g = _read_gt(path, size)
                    m = p.cache.get(o.id, f)
                    m = np.zeros_like(g) if m is None else m
                    j, fsc = jf(m, g)
                    js.append(j)
                    fs.append(fsc)
                if np.mean(js) > best_j:
                    best, best_j, best_scores = o, float(np.mean(js)), (js, fs)
            if best is None:
                continue
            js, fs = best_scores
            entry = {"object": best.label, "J": float(np.mean(js)), "F": float(np.mean(fs)),
                     "JF": float((np.mean(js) + np.mean(fs)) / 2), "frames": len(frames)}
            consecutive = sorted(frames)
            if len(consecutive) > 2 and consecutive[-1] - consecutive[0] == len(consecutive) - 1:
                al = [(p.cache.get(best.id, f) if p.cache.get(best.id, f) is not None
                       else np.zeros(size[::-1], np.uint8)).astype(np.float32) / 255 for f in consecutive]
                gl = [_read_gt(frames[f], size).astype(np.float32) / 255 for f in consecutive]
                entry["dtSSD"] = dtssd(al, gl)
            per_obj[name] = entry
        out["objects"] = per_obj
        if per_obj:
            out["mean_JF"] = float(np.mean([e["JF"] for e in per_obj.values()]))
        # Phrase recall/precision on the selection frame.
        pr = {}
        for phrase, gt_names in clip.get("phrase_gt", {}).items():
            gts = [_read_gt(_gt_frames(gt_dir, n)[selection_frame], size) for n in gt_names
                   if selection_frame in _gt_frames(gt_dir, n)]
            dets = [p.cache.get(o.id, selection_frame) for o in found.get(phrase, [])]
            r, prec = phrase_recall_precision([d for d in dets if d is not None], gts)
            pr[phrase] = {"recall": r, "precision": prec, "found": len(dets), "expected": len(gts)}
        if pr:
            out["phrases"] = pr
    # Flicker without ground truth: on the union of kept objects.
    we = WarpError()
    reader = p.reader()
    try:
        step = max(1, p.info.n_frames // 300)  # ≤ 300 pairs keeps this quick
        for idx, rgb in reader.iter_frames(0, p.info.n_frames - 1, size=size):
            if idx % step:
                continue
            ms = [m for m in p.kept_masks(idx)]
            u = np.max(np.stack(ms), axis=0) if ms else np.zeros(size[::-1], np.uint8)
            we.add(rgb, u)
    finally:
        reader.close()
    out["warp_error"] = we.mean
    return out


def run_one(clip: dict, base: Path, backend_name: str, model: str | None, quality_tier: str,
            memory_limit_gb: float, workdir: Path) -> dict:
    video = str((base / clip["video"]).resolve())
    rep: dict = {"clip": clip.get("name", Path(video).stem), "video": video, "model": model,
                 "quality": quality_tier, "backend": backend_name}
    mon = MemoryMonitor(interval=0.25).start()
    t = time.perf_counter()
    kwargs = {"memory_limit_gb": memory_limit_gb}
    if model:
        kwargs["model_id"] = model
    be = create_backend(backend_name, **kwargs)
    be.load()
    rep["load_s"] = time.perf_counter() - t
    p = Project.create(workdir / f"{rep['clip']}_{quality_tier}.bbg", video, overwrite=True)
    p.backend = backend_name
    p.track.quality = quality_tier
    rep["info"] = p.info.to_dict()
    frame = int(clip.get("frame", 0))
    sel = Selection(p, be)
    t = time.perf_counter()
    sel.encoded(frame)
    rep["encode_s"] = time.perf_counter() - t
    found, phrase_times = {}, []
    for phrase in clip.get("phrases", []):
        t = time.perf_counter()
        found[phrase] = sel.find(frame, phrase)
        phrase_times.append(time.perf_counter() - t)
    sx, sy = p.work_size[0] / p.info.width, p.work_size[1] / p.info.height
    click_times = []
    for c in clip.get("clicks", []):
        x, y = c["xy"]
        f = int(c.get("frame", frame))
        sel.encoded(f)  # click latency excludes the frame's one-off backbone pass
        t = time.perf_counter()
        sel.click(f, x * sx, y * sy, positive=c.get("positive", True))
        click_times.append(time.perf_counter() - t)
    sel.close()
    rep["phrase_s"] = max(phrase_times) if phrase_times else None
    rep["click_s"] = max(click_times) if click_times else None
    rep["objects_selected"] = [o.label for o in p.objects.values()]
    tr = Tracker(p, be).run()
    rep["track"] = tr.to_dict()
    mon.stop()
    rep["memory"] = mon.to_dict()
    rep["backend_memory"] = be.memory_stats()
    rep["objects"] = [o.label for o in p.objects.values()]
    rep["flags"] = len(find_flags(p))
    rep["quality_metrics"] = quality(p, clip, base, frame, found)
    be.unload()
    # Targets.
    crit = {}

    def check(name, value, limit, ok):
        if value is not None:
            crit[name] = {"value": value, "limit": limit, "passed": bool(ok)}

    m = rep["memory"]
    check("peak_footprint_gb", m["peak_footprint_gb"], TARGETS["peak_footprint_gb"],
          m["peak_footprint_gb"] <= TARGETS["peak_footprint_gb"])
    check("swap_growth_gb", m["swap_growth_gb"], TARGETS["swap_growth_gb"],
          m["swap_growth_gb"] <= TARGETS["swap_growth_gb"])
    if p.info.duration >= 120:  # growth only means something on long clips
        check("growth_gb_per_min", m["growth_gb_per_min"], TARGETS["growth_gb_per_min"],
              m["growth_gb_per_min"] <= TARGETS["growth_gb_per_min"])
    check("click_s", rep["click_s"], TARGETS["click_s"],
          rep["click_s"] is not None and rep["click_s"] <= TARGETS["click_s"])
    check("phrase_s", rep["phrase_s"], TARGETS["phrase_s"],
          rep["phrase_s"] is not None and rep["phrase_s"] <= TARGETS["phrase_s"])
    if p.info.height >= 1080 or p.info.width >= 1920:
        key = f"fps_{quality_tier}_1080p"
        check(key, tr.fps, TARGETS[key], tr.fps >= TARGETS[key])
    rep["criteria"] = crit
    rep["passed"] = all(c["passed"] for c in crit.values())
    return rep


def dry_run_manifest(workdir: Path) -> tuple[dict, Path]:
    from bbg.synthetic import make_clip

    make_clip(workdir / "synthetic.mp4", n_frames=60, gt_dir=workdir / "gt")
    manifest = {
        "backend": "synthetic",
        "models": [None],
        "qualities": ["balanced", "fast"],
        "clips": [{"name": "synthetic", "video": "synthetic.mp4", "frame": 0,
                   "phrases": ["ball", "dog"], "clicks": [{"xy": [544, 180]}],
                   "gt_dir": "gt", "phrase_gt": {"ball": ["ball"]}}],
    }
    return manifest, workdir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="?", help="manifest JSON (see spike/manifest.example.json)")
    ap.add_argument("--out", required=True, help="folder for the JSON reports")
    ap.add_argument("--dry-run", action="store_true", help="synthetic clip and backend, no model")
    ap.add_argument("--memory-limit-gb", type=float, default=6.0)
    ap.add_argument("--workdir", help="where projects go (default: a temporary folder)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="bbg-spike-"))
    work.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        manifest, base = dry_run_manifest(work)
    else:
        if not a.manifest:
            ap.error("give a manifest, or --dry-run")
        manifest = json.loads(Path(a.manifest).read_text())
        base = Path(a.manifest).resolve().parent
    backend = manifest.get("backend", "sam3")
    runs = []
    for clip in manifest["clips"]:
        for model in manifest.get("models", [None]):
            for q in manifest.get("qualities", ["balanced"]):
                name = f"{clip.get('name', 'clip')}__{(model or backend).replace('/', '_')}__{q}"
                print(f"== {name}", file=sys.stderr)
                try:
                    rep = run_one(clip, base, backend, model, q, a.memory_limit_gb, work)
                except Exception as e:  # a crashed run is a failed run, with the reason
                    import traceback

                    rep = {"clip": clip.get("name"), "model": model, "quality": q, "passed": False,
                           "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                rep["machine"] = machine()
                (out / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str))
                runs.append({"name": name, "passed": rep["passed"],
                             "failed": [k for k, c in rep.get("criteria", {}).items() if not c["passed"]]
                             or ([rep["error"]] if "error" in rep else [])})
                status = "PASS" if rep["passed"] else "FAIL"
                print(f"   {status} {runs[-1]['failed'] or ''}", file=sys.stderr)
    summary = {"passed": all(r["passed"] for r in runs), "runs": runs, "machine": machine(),
               "targets": TARGETS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SPIKE_COMPLETE", "passed" if summary["passed"] else "FAILED", file=sys.stderr)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
