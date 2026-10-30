# BBG (Background Be Gone): scope

On-device video background removal for machines with 16 GB of memory, on macOS and Linux. The user types or clicks the objects to keep, the app tracks them through the clip, and exports the result with the background removed or replaced. The project is free and open source, and so is everything it depends on.

**Status:** scoping draft, revised 2026-10-30 after the first round of decisions. Nothing has been built or measured yet. Speed figures quoted below come from papers and vendors, measured on other hardware. Every figure for our own targets is an estimate that the Phase 0 spike (§11) must confirm.

---

## 1. Decisions and assumptions

### Decided (2026-10-30)

| # | Decision | Consequence |
|---|---|---|
| D1 | **macOS and Linux only.** | The stack is cross-platform (§6). The native Swift, Core ML and Apple Vision route is dropped. |
| D2 | **Fully unrestricted FOSS.** | Every shipped component must be OSI-approved, with no field-of-use restrictions, no non-commercial terms and no gated downloads. Read with D5, this excludes SAM 3.1 from the default build (§5.1). |
| D3 | **macOS: the latest release that runs on an M3 Ultra Mac Studio**, i.e. macOS 27, which is Apple Silicon only. | Any Apple Silicon Mac that runs macOS 27 with ≥ 16 GB is in range. The M3 Ultra Studio starts at 96 GB, so it is the reference and development machine, but it cannot show 16 GB problems (§7). |
| D4 | **Clean object edges are enough.** | No matting models. Edges come from upsampling the mask with edge-aware filtering. This removes a whole quality tier and its biggest memory peak. |
| D5 | **Typed prompts in v1** ("dog", "person in red"); the user asked for SAM 3.1 or better. | A text-grounding model is required. The licence conflict with D2 is in §5.1. |

**How I read D2:**
- Permissive licences (Apache-2.0, MIT, BSD) are preferred.
- LGPL is accepted only for libraries that are dynamically linked: FFmpeg and Qt (PySide6).
- No GPL components, so the app itself can be Apache-2.0.
- GPU drivers and runtimes (Apple Metal, NVIDIA's driver and CUDA, AMD ROCm) are part of the user's system, not something we ship.

### Still assumed (confirm or correct)

| # | Assumption |
|---|---|
| A1 | **Offline processing, not live.** Selecting on one frame feels instant; tracking the whole clip runs as a job. |
| A2 | Inputs are phone or camera footage up to 4K, H.264/HEVC/ProRes/AV1, SDR and HDR, up to about 10 minutes long. |
| A3 | **Linux tier 1 is NVIDIA CUDA**: discrete GPUs, and unified-memory boards such as Jetson Orin NX 16 GB. **AMD ROCm is best-effort**: Strix Halo support was still settling in 2026. The CPU is a slow fallback, there for correctness only. Asahi Linux on Apple Silicon has no usable GPU compute for this, so it is CPU only. |
| A4 | "Objects to keep" means things a click or a short phrase can isolate. An object's shadows and reflections are not included (§13). |

## 2. What the user does

```
┌──────────────────────────────────────────────────────────────────┐
│ [Open…] interview.mov          Quality: Fast | Balanced  [Export]│
├────────────────────────────────────────────┬─────────────────────┤
│                                            │ Find: [dog        ] ⏎│
│                                            ├─────────────────────┤
│          viewer: current frame             │ Objects             │
│   click = add   ⌥/Alt-click = subtract     │ ■ person 1  [Keep ✓]│
│   drag = box                               │ ■ dog 1     [Keep ✓]│
│                                            │ ■ dog 2     [Keep ✓]│
│                                            │ ■ chair     [Keep  ]│
│ View: Overlay | Matte | Result             ├─────────────────────┤
│                                            │ Background          │
│                                            │ ◉ Transparent        │
│                                            │ ○ Colour ○ Image ○ Blur│
├────────────────────────────────────────────┴─────────────────────┤
│ ▕████████████▏⚠▕██████████████████▏   person 1                   │
│ ▕██████▏        ▕█████████▏           dog 1                      │
│ [▶ Track]  ██████████░░░░░░  412 / 1800 frames · ~3 min left     │
└──────────────────────────────────────────────────────────────────┘
```

1. **Open** a clip by drag-and-drop. The app shows the first frame and a timeline of thumbnails.
2. **Select** on any frame:
   - **Type** a phrase. Every match becomes an object in the list (dog 1, dog 2…), and the user unticks any they don't want.
   - **Click** adds a point (⌥/Alt-click subtracts one), or **drag a box** for anything the phrase missed.
3. **Track** forwards and backwards from every annotated frame, with a progress bar, a cancel button and a live preview. Typed phrases are re-run every N frames and at scene cuts. New matches (a second dog walking in) join the list, and matches that overlap an existing track are merged into it.
4. **Review.** The timeline flags frames where tracking looks unreliable (low object score, sudden jumps in area).
5. **Fix.** The user clicks on a flagged frame, and the app re-tracks only the affected span.
6. **Edges:** grow/shrink and feather.
7. **Background:** transparent, solid colour, image, blur of the original, or another video.
8. **Export** (§9).

A project file saves the prompts and the mask cache, so the user can quit and come back.

**Design rule:** every listed object is tracked. **Keep** only changes what gets composited, so ticking or unticking an object never requires re-tracking.

## 3. Feature tiers

| Feature | MVP | v1 | Later |
|---|:-:|:-:|:-:|
| Open video, scrub, preview (macOS and Linux/CUDA) | ✓ | | |
| Typed phrase on one frame → objects | ✓ | | |
| Click/box select on one frame | ✓ | | |
| Bidirectional tracking with bounded memory | ✓ | | |
| Several objects with Keep toggles | ✓ | | |
| Transparent / colour background | ✓ | | |
| Export ProRes 4444, PNG sequence, matte video | ✓ | | |
| Re-detecting typed phrases during tracking (new instances, objects re-entering) | | ✓ | |
| Corrections on any frame with partial re-tracking | | ✓ | |
| Confidence flags on the timeline | | ✓ | |
| Save/resume a project | | ✓ | |
| Image / blur / video backgrounds | | ✓ | |
| Flattened AV1/VP9 export, VP9-alpha WebM | | ✓ | |
| Full-resolution 4K edges, HDR passthrough | | ✓ | |
| AMD ROCm support | | ✓ (best-effort) | |
| MLX backend on macOS (if MPS is too slow) | | | ✓ |
| Optional SAM 3.1 plug-in (§5.1) | | | ✓ |
| Batch queue | | | ✓ |

## 4. Processing pipeline

```
decode ──► resize ──┬─► text grounding (phrase → boxes) ── on annotated frames,
(FFmpeg via         │    and every N frames for typed phrases
 PyAV; hw decode)   │               │ boxes
                    │               ▼
                    └─► image encoder ──► mask decoder ◄── clicks / boxes
                        (once per frame)   (cheap)
                                              │
                   memory bank ◄── memory encoder ◄── low-res mask logits (256²)
                 (last N frames +                         │
                  prompted frames)                        ▼
                                          upsample to source res + guided filter
                                                          │
                                          temporal smoothing of alpha
                                                          │
                                   mask cache on disk (per object, per frame)
                                                          │
                          compositor (GPU tensors) ──► encoder (FFmpeg via PyAV)
                                                       + audio passthrough
```

**Streaming with bounded memory is the core engineering task on 16 GB.** The reference SAM 2 video predictor is not built for it:

- `init_state` loads **every frame** of the clip up front as a float32 1024×1024 tensor (`sam2/utils/misc.py`: `torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)`). That is about 12.6 MB per frame, so a 1-minute 30 fps clip needs about 22 GB before tracking starts.
- It keeps every frame's memory features and mask logits for every object, although attention reads only the last few frames (`num_maskmem`) and the prompted frames. That is about 0.8 MB per frame per object, roughly 5–6 GB for a 1-minute clip with 4 objects.
- `offload_video_to_cpu` and `offload_state_to_cpu` save nothing with unified memory. They do help on Linux machines with a discrete GPU.

The production tracker therefore needs its own loop:
- decode into a small ring buffer,
- keep only the frames inside the memory window plus the prompted frames,
- write each frame's mask to disk as soon as it is final,
- keep image embeddings only for the frames being annotated (a small LRU).

The tracking loop in Grounded-SAM-2 (Apache-2.0) is a starting point for re-detecting typed phrases and keeping object IDs stable across re-detections.

## 5. Models

### 5.1 Typed prompts: the licence conflict

**SAM 3.1 does not meet D2.** Its code and weights are under Meta's **SAM License**:
- **Not OSI-approved.** It bars end uses prohibited by trade controls, "including those related to military or warfare purposes". That is a field-of-use restriction, contrary to point 6 of the Open Source Definition.
- It forbids reverse engineering the SAM Materials.
- It requires an indemnity.
- Meta may change the terms, and continued use counts as acceptance.
- The checkpoints are **gated**: a download has to be requested on Hugging Face.
- **Any port of SAM 3 code inherits the same licence.**

On top of that, the official repo needs a CUDA GPU (Python 3.12+, PyTorch 2.7+, CUDA 12.6+). On a Mac it runs only through community MLX ports, whose video tracking is marked experimental.

**What SAM 3.1 would buy:** Meta reports that SAM 3 roughly doubles the accuracy of earlier systems on concept segmentation in images and video, with zero-shot LVIS mask AP of 48.8 against 38.5 for the previous best. A FOSS pipeline will miss more objects and false-match more often, so the user clicks more.

| Option | Licence status | Quality | Recommendation |
|---|---|---|---|
| **A. FOSS text grounding + SAM 2.1** (detector finds boxes, SAM 2.1 tracks) | Fully FOSS | Below SAM 3.1 (gap to be measured in Phase 0) | **Default build** |
| **B. A + an optional SAM 3.1 plug-in**, a separate package and repo under the SAM License, which the user installs and gates themselves; BBG ships no SAM 3 code or weights | BBG stays fully FOSS; a user who opts in accepts Meta's terms | SAM 3.1 for those who opt in | **Recommended**, built after v1 |
| C. Make SAM 3.1 required | Breaks D2 | Best | Only if D2 is relaxed |

### 5.2 Candidates for the FOSS default

| Role | Candidate | Licence | Notes |
|---|---|---|---|
| Tracker | **SAM 2.1** Tiny / Small / Base+ / Large (38.9M / 46M / 80.8M / 224.4M params) | Apache-2.0 | Meta reports 91 / 85 / 64 / 40 fps, measured on an A100. Its example notebooks warn that "Support for MPS devices is preliminary": the model was trained with CUDA and may give different or worse results on MPS. Measure that. |
| Tracker (Fast tier) | **EdgeTAM** | Apache-2.0 | SAM 2 distilled for phones: ~16 fps on an iPhone 15 Pro Max per the paper, and 87.7 J&F on DAVIS 2017. |
| Text grounding | **Grounding DINO**, **MM-Grounding-DINO** (OpenMMLab) | Apache-2.0 | The standard pairing with SAM (Grounded-SAM-2). Use the pure-PyTorch implementation in Hugging Face `transformers`: the original repo's custom CUDA kernel doesn't exist on Mac. |
| Text grounding | **LLMDet** | Apache-2.0 | A Grounding DINO derivative trained with captions from a language model. |
| Text grounding | **OWLv2** | Apache-2.0 | Google's open-vocabulary detector. Can also take an example image as the prompt. |
| Text grounding | **Florence-2** (0.23B / 0.77B) | MIT | Grounds phrases and referring expressions ("the person on the left"). |
| Edges | Guided filter on the upsampled mask | n/a (no model) | Enough for clean edges (D4). |
| *Excluded by D2* | SAM 3 / 3.1 (SAM License), YOLOE / YOLO-World (AGPL/GPL), RMBG-2.0 (CC BY-NC), MatAnyone (NTU S-Lab, non-commercial), RobustVideoMatting (GPL-3.0) | | |

**Phase 0 decides** which grounding model is the default, by testing all four on the same phrase set (§12).

## 6. Technology stack

**Recommendation: Python with PyTorch for inference, Qt for the GUI, FFmpeg for video.** This is the only stack where the reference model code, the macOS GPU (MPS), CUDA and ROCm are all supported, and the research models change too fast to keep re-exporting them to ONNX or Core ML.

| Layer | Choice | Licence |
|---|---|---|
| Language | Python 3.12+ | PSF |
| Inference | PyTorch (MPS on macOS, CUDA/ROCm on Linux, CPU fallback); `transformers` for the grounding models | BSD-3 / Apache-2.0 |
| Later: macOS speed-up | MLX (Metal on macOS, CUDA on Linux), if MPS proves too slow | MIT |
| GUI | Qt 6 via PySide6 | LGPL-3.0 (dynamically linked) |
| Video in/out | FFmpeg via PyAV: hardware decode with VideoToolbox (macOS), NVDEC/VAAPI (Linux) | LGPL-2.1+ / BSD-3 |
| Image ops | NumPy, OpenCV (guided filter, morphology) | BSD-3 / Apache-2.0 |
| Packaging | macOS: `.app` via Briefcase or PyInstaller (signing and notarising need an Apple Developer ID). Linux: `pipx`/PyPI, and Flatpak for CPU and ROCm builds | BSD-3 / GPL-2.0 with bootloader exception |

**FFmpeg must be an LGPL build:** no `--enable-gpl` or `--enable-nonfree`, and therefore no x264 or x265. PyAV's prebuilt wheels bundle FFmpeg; **check their configure flags** and build our own if they include GPL parts.

**Rejected:**
- **Rust/C++ with ONNX Runtime:** faster and smaller, but exporting SAM 2's memory modules and the grounding models to ONNX, and keeping three execution providers working, is weeks of work per model update.
- **Swift with Core ML:** doesn't run on Linux.

## 7. Memory budget (16 GB machine)

The operating system and other open apps can take 5–6 GB. **Target app peak ≤ 6 GB, hard ceiling 8 GB, no swap growth over a 10-minute clip.**

The development machine (M3 Ultra, ≥ 96 GB) won't show memory pressure, so the budget is enforced in the app:
- MPS: `PYTORCH_MPS_HIGH_WATERMARK_RATIO`
- CUDA: `torch.cuda.set_per_process_memory_fraction`
- MLX: `mx.set_memory_limit`

Peak footprint is logged on every run. **The final QA runs on a real 16 GB Mac** (any M-series base model) and on an 8–16 GB NVIDIA Linux machine.

| Item | Estimate |
|---|---|
| SAM 2.1 Base+ weights, fp16 | ~0.16 GB (Large ~0.45 GB) |
| Grounding model weights | ~0.3–1.5 GB depending on the model; loaded at prompt time and for re-detection |
| Encoder and detector activations at ~1024² | 0.5–2 GB, **to measure** |
| Decoded frame ring buffer (16 frames) | 4K BGRA 8-bit is 33 MB a frame, about 0.5 GB in total |
| Tracker state (memory window + prompted frames) | Tens of MB per object, bounded |
| Mask cache | On disk, not in RAM |
| Preview and compositing | ~0.3 GB |
| *Optional SAM 3.1 plug-in* | ~1.75 GB of bf16 weights (~0.9 GB at 8-bit) plus activations |

The FOSS default should land around 2–4 GB. Dropping matting (D4) removed the largest peak from the first draft.

## 8. Performance targets (to validate, not yet known)

Targets, on a 16 GB Apple Silicon Mac under macOS 27:

| Measure | Target |
|---|---|
| Typed phrase to objects on the current frame | < 2 s |
| Click to mask | < 300 ms once the frame has been encoded |
| Tracking, Balanced (SAM 2.1 S/B+), 1080p | ≥ 5 fps, so a 1-minute 30 fps clip takes ≤ 6 min |
| Tracking, Fast (EdgeTAM), 1080p | ≥ 15 fps |
| 4K | Tracked at model resolution, edges refined at full resolution; ≥ 2 fps |

The Linux/CUDA targets will be set from Phase 0 numbers.

Published speeds come from an A100/H100 or an iPhone and do not carry over. EdgeTAM's paper shows that SAM 2 Tiny converted naively to Core ML ran at about 1 fps on a phone, so the runtime matters as much as the model. Fanless MacBook Airs throttle on long jobs, so the app needs pause/resume and checkpointing.

## 9. Export

Every format below can be written with FOSS encoders.

| Format | Encoder (licence) | Use |
|---|---|---|
| ProRes 4444 with alpha (`.mov`) | FFmpeg `prores_ks` (LGPL), or `prores_videotoolbox` on macOS | DaVinci Resolve, Kdenlive, Blender, Premiere, Final Cut |
| PNG / TIFF sequence with alpha | FFmpeg (LGPL) | Any tool |
| Matte-only video | FFV1 (LGPL, lossless) or ProRes 422 | Track matte in any editor |
| WebM VP9 with alpha | libvpx (BSD-3) | Web |
| Flattened AV1 / VP9 | SVT-AV1 or libaom, libvpx (BSD) | Sharing; royalty-free codecs |
| Flattened H.264/HEVC | Hardware only: VideoToolbox (macOS), NVENC/VAAPI (Linux) | Compatibility. **No x264/x265** (GPL). |
| HEVC with alpha | `hevc_videotoolbox`, macOS only | Apple apps; optional |

For all formats:
- pass the audio through,
- keep the frame rate (including variable frame rate), rotation metadata and colour metadata.

For HDR, the models see an SDR conversion, and the output keeps the original pixels and transfer function.

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Tracker memory grows with clip length (§4) | High | Own tracking loop with a bounded window, eviction and a disk cache. Soak test on 10-minute clips. |
| FOSS text grounding falls well short of SAM 3.1 | Med–High | Measure the gap in Phase 0. Re-detection, quick click fixes and the opt-in SAM 3.1 plug-in. |
| MPS accuracy or speed differs from CUDA, or ops fall back to the CPU | Med–High | Run every benchmark on both. Use the MLX backend as the escape route on macOS. |
| ROCm instability on AMD APUs | Med | Best-effort tier; CI on CPU and CUDA only at first. |
| A GPL or non-free component slips in (FFmpeg flags, transitive dependencies) | High for D2 | A licence audit in CI (`pip-licenses`, FFmpeg `-buildconf` check) that fails the build. |
| Edge flicker over time | Med | Smooth alpha over time; measure flicker in QA. |
| Drift, occlusion, objects re-entering | Med | Periodic re-detection for typed phrases, confidence flags, cheap corrections. |
| Packaging Python and PyTorch (bundle size, notarisation) | Med | A per-platform installer that downloads models with checksums; size measured in Phase 0. |
| Thermal throttling | Med | Pause/resume, checkpoints, adaptive time estimates. |
| Expecting shadows and reflections to be kept | Expectation | State it in the UI and docs (§13). |

## 11. Phases and effort

| Phase | Weeks | Delivers | Exit criterion |
|---|---|---|---|
| **0. Spike** | 2 | On the M3 Ultra (capped at 6 GB), a 16 GB Mac and a Linux CUDA box: SAM 2.1 T/S/B+ and EdgeTAM tracking; Grounding DINO / MM-GDINO / LLMDet / OWLv2 / Florence-2 grounding; SAM 3.1 on CUDA as the quality reference. Peak memory, fps, J&F, flicker, phrase recall. The PyAV FFmpeg licence checked. | Default models chosen from **measured** numbers; the gap to SAM 3.1 known; go/no-go on the SAM 3.1 plug-in and on MLX. |
| **1. MVP** | 4–6 | Open; typed phrase plus click/box selection; bidirectional tracking with bounded memory; feathered masks; transparent/colour background; ProRes 4444 / PNG / matte export. macOS and Linux/CUDA. | A 10-minute 1080p clip finishes inside the memory budget on both platforms. |
| **2. Usable tool** | 3–4 | Re-detection with stable IDs, corrections with partial re-tracking, confidence timeline, project save, image/blur/video backgrounds, VP9/AV1 export. | A non-expert gets an acceptable result on the test set without help. |
| **3. Quality and platforms** | 2–3 | Full-resolution 4K edges, temporal smoothing, HDR handling, ROCm best-effort, performance work (MLX if needed). | Flicker and edge metrics meet the bar set in Phase 0. |
| **4. Ship** | 2–3 | macOS `.app` and Linux packages, model downloader showing each model's licence and checksum, the CI licence audit, docs. | Release candidate. |
| *5. SAM 3.1 plug-in (optional)* | *2–4* | *A separate SAM-License repo: CUDA first, then macOS through MLX.* | *Opt-in install works without changing BBG's licence.* |

**Total: about 13–18 weeks for one experienced engineer**, plus 2–4 weeks for the plug-in. Dropping matting saved about 2 weeks; supporting Linux adds about 2–3. This is a rough estimate until Phase 0 reports.

**Skills:** Python/PyTorch ML deployment, Qt, FFmpeg and video basics (codecs, colour, timing), and Linux GPU stacks.

**Hardware:**
- the M3 Ultra Mac Studio for development,
- a real 16 GB Apple Silicon Mac for QA,
- an NVIDIA Linux box with 8–16 GB (a Jetson Orin NX 16 GB if unified memory on Linux matters).

## 12. Testing and acceptance

**Test clips** (about 12, owned or licensed):
- talking head
- full-body motion
- loose hair
- a pet
- two or more people crossing
- an object held in a hand
- an occlusion
- fast motion with blur
- low light
- 4K HDR
- variable frame rate
- a 10-minute clip

**Phrase set:** about 30 phrases over those clips, from plain classes ("dog") to descriptions ("the man in the blue shirt").

**Measures:**
- J&F against hand-drawn masks on about 20 frames per clip,
- phrase recall and precision per grounding model,
- flicker (dtSSD on the alpha),
- fps,
- peak memory and swap,
- the number of corrections until the user is satisfied.

**MVP acceptance:**
- the whole test set completes on macOS and on Linux/CUDA with no out-of-memory failure or swap growth,
- exports open in DaVinci Resolve, Kdenlive and Blender,
- audio stays in sync,
- the licence audit passes.

## 13. Out of scope (v1)

- Live camera or virtual-webcam use.
- Hair-level matting and semi-transparency (D4).
- Keeping an object's shadows, reflections or smoke (the "omnimatte" research area).
- Removing an object and filling the hole (inpainting).
- Windows; cloud processing; training models.

## 14. Decisions still needed

1. **SAM 3.1 vs D2:** option B (FOSS default plus an opt-in SAM 3.1 plug-in, recommended), option A (FOSS only), or option C (relax D2)? See §5.1.
2. **Linux hardware:** is NVIDIA CUDA as tier 1 right? Does AMD (ROCm) or Jetson matter?
3. **Licences:** is Apache-2.0 right for BBG itself? Are LGPL libraries (FFmpeg, Qt) acceptable under "fully unrestricted"?
4. **Limits:** maximum resolution and length (assumed 4K and 10 minutes).

## Sources

- SAM 2 repo (sizes, A100 speeds, Apache-2.0, MPS caveat in `notebooks/video_predictor_example.ipynb`): https://github.com/facebookresearch/sam2
- SAM 3 / 3.1 repo (CUDA requirement, SAM License, 3.1 release): https://github.com/facebookresearch/sam3 · licence: https://github.com/facebookresearch/sam3/blob/main/LICENSE
- SAM 3 paper (2× concept-segmentation accuracy, LVIS 48.8 vs 38.5): https://arxiv.org/abs/2511.16719
- Open Source Definition: https://opensource.org/osd
- EdgeTAM: https://arxiv.org/abs/2501.07256 · https://github.com/facebookresearch/EdgeTAM
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO · Grounded-SAM-2: https://github.com/IDEA-Research/Grounded-SAM-2
- MM-Grounding-DINO (MMDetection): https://github.com/open-mmlab/mmdetection
- LLMDet: https://github.com/iSEE-Laboratory/LLMDet
- OWLv2 (Scenic): https://github.com/google-research/scenic
- Florence-2: https://huggingface.co/microsoft/Florence-2-large
- MLX (Metal and CUDA backends): https://github.com/ml-explore/mlx
- SAM 2.1 MLX port: https://github.com/eisneim/sam2.1_mlx · SAM 3 MLX builds: https://huggingface.co/mlx-community/sam3-8bit
- PyAV: https://github.com/PyAV-Org/PyAV
- macOS 27 compatibility (Apple Silicon only): https://www.macrumors.com/2026/04/18/macos-27-compatibility-change/ · https://appleinsider.com/articles/26/06/08/macos-27-compatibility-list-focuses-entirely-on-apple-silicon
- ROCm on Strix Halo status: https://community.frame.work/t/linux-rocm-january-2026-stable-configurations-update/79876
- YOLOE licence (AGPL-3.0): https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr
- RobustVideoMatting (GPL-3.0): https://github.com/PeterL1n/RobustVideoMatting
