# BBG — Background Be Gone

On-device video background removal for Apple Silicon Macs with 16 GB of
unified memory. Type what to keep ("dog", "person") or click it, BBG tracks
it through the clip with Meta's SAM 3.1, you fix any frame it flags, and
export with the background removed or replaced. Nothing leaves the Mac.

![BBG main window: a synthetic test clip after tracking](docs/screenshot.png)

*The screenshot uses the synthetic test backend, not SAM 3.1.*

Scope, decisions and plan: [SCOPE.md](SCOPE.md).

## Status

The v1 pipeline and GUI are built. It has **not yet run with the real SAM 3.1
weights**: that is Phase 0, which has to run on a Mac ([spike/](spike/README.md)).

| Part | State | How it was checked |
|---|---|---|
| Video in/out: exact frame index, VFR, rotation, backwards decoding | done | tests, against the FFmpeg CLI |
| Export: ProRes 4444 / WebM with alpha, PNG, matte, H.264/HEVC flattened, audio passthrough | done | tests decode the output and check alpha, timing and audio |
| HEVC with alpha (`hevc_videotoolbox`) | written | **untested**: needs macOS |
| Tracking: keyframes, nearest-keyframe ownership, partial re-tracking, re-detection of typed phrases, flags | done | tests with the synthetic backend (J > 0.95 against ground truth) |
| SAM 3.1 backend (mlx-vlm 0.7.11) | done | `tests/test_sam3_backend.py` (3 tests, passing) runs mlx-vlm's real SAM 3.1 code with random weights in a shallow, narrow config. It checks phrase detection, clicks with refinement, session start, object order on add, that tracker memory stays bounded, and the full tracker loop through the backend. **Mask quality and speed: not measured** (needs the real weights on a Mac) |
| GUI (PySide6) | done | offscreen end-to-end test: select, track, view modes, export |
| Phase 0 spike harness | done | dry run with the synthetic backend |
| macOS packaging (`.app`) | config only | `[tool.briefcase]` in pyproject; not built |

## Install (macOS, Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[sam3,gui]"
bbg path/to/clip.mov          # the GUI
```

The first run asks you to accept the SAM License and downloads the SAM 3.1
weights (`mlx-community/sam3.1-bf16`, about 1.8 GB) from Hugging Face.

### Without the model

```bash
pip install -e ".[gui]"
bbg-cli demo /tmp/bbg-demo                 # synthetic clip: track + export, no model
bbg --backend synthetic /tmp/bbg-demo/demo.mp4   # the GUI on it
```

The synthetic backend segments the synthetic clips by colour ("ball", "box",
"dog"). It exists for tests and for trying the interface; it can't segment
real footage.

## Use

**GUI.** Open a clip, then:
- **Find:** type a short phrase. Every match becomes an object.
- **Click** to start an object or add to the selected one. **⌥-click** subtracts.
- **Drag** a box for a new object. With *Find similar* on, the box finds every look-alike instead.
- **▶ Track.** The timeline shows where each object is, its keyframes (◆) and flags (red).
- Jump between flags with ⚑◀ / ▶⚑. Click to fix a frame, then Track again: only that span is re-tracked.
- **Merge into…** folds a duplicate (say, "ball 2" after the ball re-emerged) into the original. Track again afterwards.
- Untick objects you don't want, choose a background, then **Export**.

**CLI.**

```bash
bbg-cli track clip.mov --phrase person --click 812,430@120 --report track.json
bbg-cli export clip.bbg out.mov --format prores4444
bbg-cli export clip.bbg out.mp4 --format h264 --background blur
bbg-cli export clip.bbg out.mp4 --format hevc --background image:beach.jpg --keep "person 1"
```

Formats: `prores4444`, `hevc_alpha` (macOS), `webm`, `png` (with alpha);
`matte`; `h264`, `hevc` (need a background). Click and box coordinates are
source pixels, `@frame` optional.

## How it works

```
src/bbg/
  video/reader.py    exact frame index from a packet scan; forwards and chunked-backwards decode
  video/writer.py    export formats, source timestamps, audio passthrough
  backends/          the Segmenter interface; SAM 3.1 on MLX; the synthetic test backend
  selection.py       phrase / click / box / find-similar on one frame → objects with keyframes
  tracking.py        planning (who owns each frame) and the tracking loop, re-detection, flags
  masks/             on-disk mask cache; guided upsampling, grow/shrink, feather, temporal smoothing
  export.py          kept objects over a background, full resolution
  project.py         the .bbg project folder
  gui/               PySide6 window; the engine thread that owns the model
  cli.py, metrics.py, memory.py, synthetic.py
spike/               Phase 0 measurement harness
```

**Memory stays flat with clip length.**
- Frames are decoded a few at a time.
- Masks go to disk as soon as they're made: one PNG per object per frame, at most 1024 px on the long side.
- The tracker keeps only the frames its memory can still reach. mlx-vlm 0.7.11 prunes the rest, and `tests/test_sam3_backend.py` checks it.

**Re-tracking is local.**
- Every frame is tracked from its nearest keyframe of that object.
- The project records which keyframe (and which version of its mask) produced each frame.
- After a correction, only frames whose owner changed are re-tracked.

**Edges.**
- The model's mask (1024 px) is upsampled to full resolution with a fast guided filter, so its edge lands on the picture's edge.
- It is then sharpened to a clean edge, and your Grow/Shrink and Feather settings are applied.

## Development

```bash
pip install -e ".[gui,dev]"
QT_QPA_PLATFORM=offscreen pytest                 # fast suite (synthetic backend, GUI offscreen)
pip install "mlx[cpu]" mlx-vlm==0.7.11           # Linux: CPU MLX for the next line
pytest -m slow                                   # SAM 3.1 plumbing, random weights
```

The slow tests build mlx-vlm's SAM 3.1 with random weights in a shallow
configuration: one backbone layer, one detector layer. On Linux, MLX's CPU
build is single-threaded: the three tests take about 35 minutes. On a Mac
they run on Metal.

`mlx-vlm` is pinned: `backends/sam3_mlx.py` calls its SAM 3.1 internals, and
the docstring there lists what it relies on.

## Licences

- **BBG's code:** Apache-2.0 ([LICENSE](LICENSE)).
- **SAM 3.1 weights:** Meta's SAM License, not Apache-2.0. They are downloaded at first run, after a prompt, and never shipped with BBG.
- **mlx-vlm's SAM 3.1 port:** mlx-vlm itself is MIT, but this port is derived from Meta's SAM 3 code.
- **FFmpeg:** PyAV's prebuilt wheels bundle an FFmpeg build that includes GPL encoders (x264, x265). Before distributing a packaged `.app`, either meet the GPL for that bundle or build PyAV against an LGPL FFmpeg.

## Known limitations

- **No real-weight results yet.** Mask quality, speed and peak memory on a 16 GB Mac are unmeasured until Phase 0.
- **Short phrases only.** SAM 3.1 is built for short phrases. For "the man on the left", type "man" and untick the others.
- **Duplicate objects.** Re-detection can create a duplicate when a tracked object re-emerges somewhere its track didn't follow: it appears as e.g. "ball 2". Both are kept by default, so the output is right. Use *Merge into…* to tidy the list.
- **Fast quality is approximate.** It tracks every other frame and averages the skipped frames from their neighbours. Fast motion will lag on those frames.
- **Shadows and reflections aren't included.** They aren't part of the object (SCOPE §13).
