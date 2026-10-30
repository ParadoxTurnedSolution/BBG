# Phase 0 spike

Measures SAM 3.1 on your Macs against the targets in `SCOPE.md` §7–8, so that
the memory budget and speed figures in the scope can be replaced with
measured ones. It must run on a Mac: MLX runs on Metal.

## Run it

On each machine (the M3 Ultra Studio, and a 16 GB Apple Silicon Mac):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[sam3,gui,dev]"
bbg-cli download-model          # shows the SAM License, then downloads ~1.8 GB
python spike/run_spike.py spike/manifest.json --out spike/reports/$(hostname -s)
```

`--memory-limit-gb` (default 6) caps what MLX may allocate, so the Studio
behaves like the 16 GB budget. It doesn't cap the rest of the process, so the
report's `peak_footprint_gb` is the number to believe. Quit other apps on the
16 GB Mac first: the run is meant to see swap growth if there is any.

Check the harness without the model: `python spike/run_spike.py --dry-run --out /tmp/spike`.

## The test set

Put clips in `spike/clips/` (git-ignored) and copy
`manifest.example.json` to `manifest.json`. Per clip:

| Field | Meaning |
|---|---|
| `video` | path, relative to the manifest |
| `frame` | frame to select on (default 0) |
| `phrases` | typed prompts, e.g. `["person"]` |
| `clicks` | `[{"xy": [x, y], "frame": 0, "positive": true}]` in source pixels |
| `gt_dir` | optional ground truth: one folder per object, PNG masks named `000123.png` (frame number) at source resolution; any subset of frames |
| `phrase_gt` | `{"dog": ["dog_1", "dog_2"]}`: which ground-truth objects a phrase should find on `frame` |

The 12 clips in `SCOPE.md` §12 are the target set. For J&F, about 20
hand-drawn masks per clip is enough. Annotate a run of consecutive frames on a
couple of clips too: that enables the dtSSD flicker measure. The no-reference
`warp_error` is computed for every clip regardless.

Top-level manifest keys: `models` (Hugging Face ids, e.g.
`mlx-community/sam3.1-bf16`, plus an 8-bit conversion if you have one) and
`qualities` (`balanced`, `fast`). Every clip runs with every model and quality.

## What comes back

One JSON per run plus `summary.json`; the script exits non-zero if any run
failed a target. Each run reports:

- `memory`: `peak_footprint_gb` (Activity Monitor's figure, which includes
  Metal buffers), `growth_gb_per_min` (after warm-up; only judged on clips of
  2 minutes or more), `swap_growth_gb`
- `load_s`, `encode_s` (one backbone pass), `phrase_s`, `click_s`
- `track`: fps, and where the time went (`decode_s`, `encode_s` for the backbone,
  `track_s`, `detect_s` for re-detection, `write_s`), new and re-attached objects
- `quality_metrics`: J, F and J&F per ground-truth object, phrase recall and
  precision, `dtSSD`, `warp_error`
- `criteria`: each target with its value, limit and pass/fail
- `machine`: chip, memory, macOS and library versions

Quality is reported, not judged: Phase 0 sets that bar.
