# BBG (Background Be Gone): scope

On-device video background removal for machines with 16 GB of unified memory. The user clicks the objects to keep, the app tracks them through the clip, and exports the result with the background removed or replaced.

**Status:** scoping draft, 2026-10-30. Nothing has been built or measured yet. Speed figures quoted below come from papers and vendors, measured on other hardware. Every figure for our own targets is an estimate that the Phase 0 spike (§11) must confirm.

---

## 1. Assumptions (confirm or correct)

| # | Assumption | Why it matters |
|---|---|---|
| A1 | "16 GB unified memory" means an **Apple Silicon Mac** (M1–M5, including the 16 GB base MacBook Air) running macOS. | It decides the whole stack (§6). Windows-on-Arm and Intel Lunar Lake laptops also share memory between CPU and GPU. Supporting them is a different build. |
| A2 | **Offline processing, not live.** Selecting on one frame feels instant. Tracking the whole clip runs as a job and may be slower than real time. | Live webcam use needs a different model tier and architecture. |
| A3 | Inputs are phone or camera footage up to 4K, H.264/HEVC/ProRes, SDR and iPhone HDR (HLG), up to about 10 minutes long. | Memory budget, colour handling, test matrix. |
| A4 | It will be **distributed** (free or paid), not just used privately. | It rules out non-commercial model weights (§5). |
| A5 | "Features/objects to keep" means things a click can isolate: a person, a pet, a held object, or part of one (by subtracting). | Shadows, reflections and smoke that belong to an object are not segmented by these models (§13). |

## 2. What the user does

```
┌──────────────────────────────────────────────────────────────────┐
│ [Open…] interview.mov    Quality: Fast | Balanced | Best  [Export]│
├────────────────────────────────────────────┬─────────────────────┤
│                                            │ Objects             │
│                                            │ ■ Person 1   [Keep ✓]│
│          viewer: current frame             │ ■ Dog        [Keep ✓]│
│   click = add   ⌥-click = subtract         │ ■ Chair      [Keep  ]│
│   drag = box                               │ [Auto-detect] [+ New]│
│                                            ├─────────────────────┤
│                                            │ Background          │
│ View: Overlay | Matte | Result             │ ◉ Transparent        │
│                                            │ ○ Colour ○ Image ○ Blur│
├────────────────────────────────────────────┴─────────────────────┤
│ ▕████████████▏⚠▕██████████████████▏   Person 1                   │
│ ▕██████▏        ▕█████████▏           Dog                        │
│ [▶ Track]  ██████████░░░░░░  412 / 1800 frames · ~3 min left     │
└──────────────────────────────────────────────────────────────────┘
```

1. **Open** a clip by drag-and-drop. The app shows the first frame and a timeline of thumbnails.
2. **Select** on any frame:
   - **Auto-detect** outlines the likely subjects. The user ticks the ones to keep.
   - **Click** adds a point (⌥-click subtracts one), or **drag a box**.
   - *Optional, depends on model choice:* type a word ("dog") to select every match.
3. **Track** forwards and backwards from every frame the user annotated, with a progress bar, a cancel button and a live preview.
4. **Review.** The timeline shows where each object is present and flags frames where tracking looks unreliable (low object score, sudden jumps in area). The user jumps between flags.
5. **Fix.** The user clicks on a flagged frame, and the app re-tracks only the affected span.
6. **Refine edges** with the quality tier, grow/shrink and feather controls.
7. **Background:** transparent, solid colour, image, blur of the original, or another video.
8. **Export** (§9).

A project file saves the prompts and the mask cache, so the user can quit and come back.

**Design rule:** every selected object is tracked. **Keep** only changes what gets composited, so ticking or unticking an object never requires re-tracking.

## 3. Feature tiers

| Feature | MVP | v1 | Later |
|---|:-:|:-:|:-:|
| Open video, scrub, preview | ✓ | | |
| Click/box select on one frame, track forwards and backwards | ✓ | | |
| Several objects with Keep toggles | ✓ | | |
| Transparent / colour background | ✓ | | |
| Export ProRes 4444, PNG sequence, flattened MP4 | ✓ | | |
| Corrections on any frame with partial re-tracking | | ✓ | |
| Auto-detect proposals | | ✓ | |
| Confidence flags on the timeline | | ✓ | |
| Save/resume a project | | ✓ | |
| Image / blur / video backgrounds, HEVC-with-alpha export | | ✓ | |
| "Best" edge tier (matting for hair and motion blur) | | ✓ | |
| 4K at full-resolution edges, HDR passthrough | | ✓ | |
| Text prompts ("all the dogs") | | | ✓ |
| Batch queue, iPad build, WebM-alpha export | | | ✓ |

## 4. Processing pipeline

```
decode ──► resize to model input ──► image encoder ──► mask decoder ◄── user prompts
(VideoToolbox)   (1024², ImageNet norm)  (expensive,        (cheap, ms)     (points/box)
                                          once per frame)        │
                                                                 ▼
                      memory bank ◄── memory encoder ◄── low-res mask logits (256²)
                    (last N frames +                            │
                     prompted frames)                           ▼
                                               upsample to source res + edge refine
                                               (guided filter, or matting in the
                                                boundary band only)
                                                                 │
                                               temporal smoothing of alpha
                                                                 │
                                         mask cache on disk (per object, per frame)
                                                                 │
                         compositor (Metal / Core Image) ──► encoder (AVAssetWriter)
                                                              + audio passthrough
```

**Streaming with bounded memory is the core engineering task on 16 GB.** The reference SAM 2 video predictor is not built for it:

- `init_state` loads **every frame** of the clip up front as a float32 1024×1024 tensor (`sam2/utils/misc.py`: `torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)`). That is about 12.6 MB per frame, so a 1-minute 30 fps clip needs about 22 GB before tracking starts.
- It keeps every frame's memory features and mask logits for every object, although attention reads only the last few frames (`num_maskmem`) and the prompted frames. That is about 0.8 MB per frame per object (64×64×64 memory features in bf16 plus 256² logits), roughly 5–6 GB for a 1-minute clip with 4 objects.
- `offload_video_to_cpu` and `offload_state_to_cpu` move this data from GPU memory to CPU memory. With unified memory those are the same RAM, so the options save nothing.

The production tracker therefore needs its own loop:
- decode into a small ring buffer,
- keep only the frames inside the memory window plus the prompted frames,
- write each frame's mask to disk as soon as it is final,
- cache image embeddings only for the frames the user is annotating (a small LRU), so that clicking stays instant.

## 5. Model options

### Tracking and segmentation (core)

| Candidate | Size | Licence | Notes |
|---|---|---|---|
| **SAM 2.1** Tiny / Small / Base+ / Large | 38.9M / 46M / 80.8M / 224.4M params | Apache 2.0 | Point/box prompts and video memory. Meta reports 91 / 85 / 64 / 40 fps, but on an A100, not on Apple hardware. Apple publishes fp16 Core ML conversions on Hugging Face (`apple/coreml-sam2.1-*`); they are **image-only**, with no memory modules converted. A community MLX port that tracks video exists (`eisneim/sam2.1_mlx`), but it is early-stage and was tested only on 60-frame clips. |
| **EdgeTAM** (Meta, CVPR 2025) | SAM 2 distilled for phones | Apache 2.0 | About 16 fps on an iPhone 15 Pro Max per the paper, versus about 1 fps for SAM 2 Tiny converted naively to Core ML. 87.7 J&F on DAVIS 2017. Candidate for the **Fast** tier. |
| **SAM 3 / SAM 3.1** (Meta, Nov 2025 / Mar 2026) | ~0.86–0.87B params | **SAM License** (custom Meta licence) | Adds text prompts ("every dog") and detect-then-track, which can find an object again after it leaves the frame. 3.1's Object Multiplex tracks objects jointly; Meta reports about 2× throughput at moderate object counts, measured on an H100. MLX ports exist (8-bit, 5-bit), but their video tracking is marked experimental. The licence is royalty-free and allows redistribution with the licence attached. It also bans trade-controlled and military end uses, carries an indemnity, and lets Meta change the terms. **Needs legal review before shipping.** |
| **Apple Vision** (built in, nothing to bundle) | n/a | OS API | `VNGeneratePersonSegmentationRequest` segments people per frame. `VNGenerateForegroundInstanceMaskRequest` (macOS 14+) segments salient subjects in still images. `GenerateIterativeSegmentationRequest` (WWDC26 session 237) is tap-to-segment with include/exclude points: still images only, the model downloads on first use, and its minimum OS is not yet confirmed. **None of them track across frames.** Use them for Auto-detect proposals on the annotated frame and as a fast people-only mode. |

### Edge refinement and matting ("Best" tier)

| Candidate | Licence | Notes |
|---|---|---|
| Guided filter on the upsampled mask | n/a, no model | Cheap and handles most edges. The **Balanced** default. |
| **ViTMatte** (trimap from the tracked mask, boundary band only) | MIT code | Check the weights: trained on research matting datasets whose terms may not allow commercial use. |
| **BiRefNet** (general, HR, matting and lite variants up to about 2K) | MIT code | Heavy: the authors report 3.45 GB of GPU memory at 1024² fp16 on an RTX 4090. Works per frame, with nothing keeping edges consistent over time. |
| **BEN2** | MIT code | Alternative per-frame refiner, to evaluate. |
| *Avoid unless licensed:* RMBG-2.0 | CC BY-NC 4.0 | Non-commercial. BRIA sells a commercial licence. |
| *Avoid unless licensed:* MatAnyone, MatAnyone 2 | NTU S-Lab 1.0 | Non-commercial. Otherwise the best fit (video matting from a first-frame mask). |
| *Avoid:* RobustVideoMatting | GPL-3.0 | Copyleft; incompatible with a closed app. |

**Proposed for v1** (spike to confirm):
- **Tracker:** SAM 2.1 Small or Base+ for Balanced, EdgeTAM for Fast.
- **Edges:** guided filter by default; matting in the boundary band for Best, licence permitting.
- **Auto-detect:** Apple Vision proposals.
- **SAM 3.1:** evaluated in the spike for text prompts and re-detection. Adopt it only if the licence is acceptable and it fits the memory budget.

## 6. Technology stack

**Recommendation: a native macOS app.**
- **UI:** Swift and SwiftUI, with an AppKit/MTKView viewer.
- **Video:** AVFoundation and VideoToolbox for hardware decode and encode.
- **Compositing:** Metal and Core Image.
- **Inference:** Core ML (Neural Engine or GPU), converted with `coremltools`. MLX Swift is the alternative where Core ML conversion stalls.

| Option | For | Against |
|---|---|---|
| **Native Swift + Core ML** (recommended) | Tight memory control, hardware codecs, the Neural Engine (frees the GPU and runs cooler on fanless Macs), native ProRes/HEVC-alpha export, a small notarised `.app`. | The SAM 2 memory encoder and memory attention must be converted and validated ourselves (Apple's conversion covers only image segmentation). It is Mac-only. |
| Native Swift + MLX Swift | Ports are simpler than Core ML conversion, community SAM 2.1/3 ports exist, quantisation is easy. | GPU only, no Neural Engine. The ports are young. |
| Python (PyTorch MPS) + Qt/PySide | Fastest to prototype, since the reference code runs as is. | A bundle of 1–2 GB, less predictable MPS memory, some ops fall back to the CPU, awkward to sign and ship. **Use it for the spike only.** |
| C++/Rust core + ONNX Runtime (CoreML / DirectML / QNN providers) + Qt or Tauri | Cross-platform (Windows on Arm, Lunar Lake). | Roughly 1.5× the effort, weaker Apple performance, and three model paths to validate. Only if A1 is wrong. |

## 7. Memory budget (16 GB machine)

macOS, a browser and an editor can easily take 5–6 GB. **Target app peak ≤ 6 GB, hard ceiling 8 GB, no swap growth over a 10-minute clip.** The GPU is capped below physical RAM; read `MTLDevice.recommendedMaxWorkingSetSize` on the real target machine rather than assuming a value.

| Item | Estimate |
|---|---|
| SAM 2.1 Base+ weights, fp16 | ~0.16 GB (Large ~0.45 GB) |
| SAM 3.1 weights | ~1.75 GB at bf16, ~0.9 GB at 8-bit |
| Encoder activations at 1024² | 0.5–1.5 GB, **to measure** |
| Decoded frame ring buffer (16 frames) | 4K BGRA 8-bit is 33 MB a frame, about 0.5 GB in total |
| Tracker state (memory window + prompted frames) | Tens of MB per object, bounded |
| Mask cache | On disk, not in RAM |
| Preview and compositing textures | ~0.3 GB |
| Matting model + activations (Best tier) | **Several GB.** Run it as a separate pass after tracking, on the boundary band only, with the tracker unloaded. |

Balanced should land around 2–4 GB. Best is where 16 GB gets tight, which is why tracking and refinement run as separate passes and never at the same time.

## 8. Performance targets (to validate, not yet known)

Targets, on the slowest supported machine (M1 or M2 MacBook Air, 16 GB):

| Measure | Target |
|---|---|
| Click to mask on an annotated frame | < 300 ms after that frame's encoder has run once (target < 1 s) |
| Tracking, Balanced, 1080p | ≥ 5 fps, so a 1-minute 30 fps clip takes ≤ 6 min |
| Tracking, Fast, 1080p | ≥ 15 fps |
| Best tier, 4K | ≥ 1 fps; acceptable as an overnight job |

Published speed numbers come from an A100/H100 or an iPhone and do not carry over. EdgeTAM's paper shows that SAM 2 Tiny converted naively to Core ML ran at about 1 fps on a phone, so how a model is converted matters as much as which model is picked. **Fanless MacBook Airs throttle** on long jobs, so the app needs pause/resume, checkpointing and time estimates that adapt.

## 9. Export

| Format | Use | Notes |
|---|---|---|
| ProRes 4444 with alpha (`.mov`) | Final Cut, Premiere, Resolve | Large files. Hardware encode only on chips with a ProRes engine; software encoding elsewhere. |
| HEVC with alpha (`.mov`) | Apple apps, Safari, Keynote | Small files; poorly supported outside Apple. |
| PNG / TIFF sequence with alpha | Any tool | Universal; many files. |
| Matte-only video (greyscale) | Track matte in any editor | Cheap and very compatible. |
| Flattened MP4 (H.264/HEVC) | Sharing directly | With a colour, image, blur or video background. |
| WebM VP9 with alpha | Web | Needs a bundled libvpx (BSD). Later. |

For all formats:
- pass the audio through,
- keep the frame rate (including variable frame rate), rotation metadata and colour space.

**HDR needs a decision.** The proposed default: models see an SDR conversion, and the output keeps the original pixels and transfer function.

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Tracker memory grows with clip length (§4) | High | Own tracking loop with a bounded window, eviction and a disk cache. Soak test on 10-minute clips. |
| Converting SAM 2's memory attention (RoPE) and memory encoder to Core ML; Neural Engine op coverage | High | Start from Apple's image-only conversion and export fixed shapes. Fall back to the GPU, or to MLX, per module. |
| Hair, fur, motion blur, semi-transparency | Med–High | Best tier with matting. Quality tiers set expectations in the UI. |
| Edge flicker over time | Med | Smooth alpha over time; refine consistently from frame to frame; flicker metric in QA. |
| Drift, occlusion, objects leaving and re-entering | Med | Confidence flags and cheap corrections. SAM 3.1's re-detection as an option. |
| Model licences | High if shipped | Only Apache/MIT by default. Legal review for the SAM License; no non-commercial weights. |
| Thermal throttling | Med | Pause/resume, checkpoints, adaptive time estimates. |
| HDR, VFR, rotation, odd codecs | Med | A test matrix of real phone footage. |
| Expecting shadows and reflections to be kept | Expectation | State it in the UI and docs; out of scope (§13). |

## 11. Phases and effort

| Phase | Weeks | Delivers | Exit criterion |
|---|---|---|---|
| **0. Spike** | 1–2 | Runs of SAM 2.1 T/S/B+, EdgeTAM and SAM 3.1 (PyTorch MPS and MLX) on a 16 GB M-series Mac over the test set (§12); peak memory, fps, quality and flicker. Apple Vision proposals tried. An attempted Core ML conversion of SAM 2.1's memory modules. | Models and stack chosen from **measured** numbers; go/no-go on 4K and on the Best tier. |
| **1. MVP** | 4–6 | Open, click/box select, bidirectional tracking with bounded memory, feathered masks, transparent/colour background, ProRes 4444 / PNG sequence / MP4 export. | A 10-minute 1080p clip finishes inside the memory budget. |
| **2. Usable tool** | 3–4 | Object list with Keep toggles, Auto-detect, corrections with partial re-tracking, confidence timeline, project save, image/blur backgrounds, HEVC alpha. | A non-expert gets an acceptable result on the test set without help. |
| **3. Quality** | 3–5 | Matting tier, temporal smoothing, full-resolution 4K edges, HDR handling. | Flicker and edge metrics meet the bar set in Phase 0. |
| **4. Ship** | 2–4 | Performance and thermal tuning, model download or bundling, signing and notarisation (and App Store sandboxing, if chosen), opt-in crash reporting, accessibility, docs. | Release candidate. |

**Total: about 13–21 weeks for one experienced engineer**, or about 2.5–4 months for two: one on app and video, one on ML deployment. This is a rough estimate until Phase 0 reports.

**Skills:** Swift/SwiftUI, AVFoundation, Metal; ML deployment (PyTorch, `coremltools`, quantisation); video basics (codecs, colour, frame timing).

**Hardware:** develop and test on the slowest target, a 16 GB M1 or M2 MacBook Air, not only on a large-memory desktop.

## 12. Testing and acceptance

**Test set** (about 12 clips, owned or licensed):
- talking head
- full-body motion
- curly or loose hair
- a pet
- two or more people crossing
- an object held in a hand
- an occlusion
- fast motion with blur
- low light
- 4K iPhone HDR
- variable frame rate
- a 10-minute clip

**Measures:**
- J&F against hand-drawn masks on about 20 frames per clip,
- flicker over time (dtSSD on the alpha),
- fps,
- peak memory and swap use,
- the number of corrections, and time, until the user is satisfied.

**MVP acceptance:**
- the whole test set completes with no out-of-memory failure or swap growth,
- each export opens correctly in Final Cut Pro and DaVinci Resolve,
- audio stays in sync.

## 13. Out of scope (v1)

- Live camera or virtual-webcam use.
- Keeping an object's **shadows, reflections or smoke**. This is the "omnimatte" research area, too heavy for 16 GB today.
- **Removing an object and filling the hole** (inpainting). That is the opposite job: here we keep the selection and drop everything else.
- Cloud processing. Everything stays on the device.
- Training or fine-tuning models.

## 14. Decisions needed

1. **Platforms:** macOS only, or also iPad or Windows? (Decides §6.)
2. **Distribution:** paid or free, App Store or direct download? Is Meta's SAM License acceptable? (Decides whether SAM 3.x and text prompts are possible.)
3. **Minimum hardware and OS:** M1 and macOS 14? Or newer, to rely on newer Vision APIs?
4. **Quality bar for v1:** clean object edges, or hair-level matting from the start?
5. **Limits:** maximum resolution and length (4K? 10 minutes?).
6. **Text prompts:** wanted in v1? (Only SAM 3.x offers them.)

## Sources

- SAM 2 repo (sizes, A100 speeds, Apache 2.0): https://github.com/facebookresearch/sam2
- SAM 2 frame preloading: `sam2/utils/misc.py` and `sam2/sam2_video_predictor.py` in that repo
- SAM License (SAM 3): https://github.com/facebookresearch/sam3/blob/main/LICENSE
- SAM 3.1 release coverage: https://gigazine.net/gsc_news/en/20260331-meta-releasing-sam-3-1
- EdgeTAM paper: https://arxiv.org/abs/2501.07256 · repo (Apache 2.0): https://github.com/facebookresearch/EdgeTAM
- Apple Core ML SAM 2.1 (image-only): https://huggingface.co/apple/coreml-sam2.1-tiny
- SAM 2.1 MLX port: https://github.com/eisneim/sam2.1_mlx
- SAM 3 MLX builds: https://huggingface.co/mlx-community/sam3-8bit
- WWDC26 session 237 (tap-to-segment): https://developer.apple.com/videos/play/wwdc2026/237/
- BiRefNet (MIT): https://github.com/ZhengPeng7/BiRefNet
- ViTMatte (MIT): https://github.com/hustvl/ViTMatte
- BEN2 (MIT): https://github.com/PramaLLC/BEN2
- RMBG-2.0 (CC BY-NC 4.0): https://github.com/Bria-AI/RMBG-2.0
- MatAnyone (NTU S-Lab 1.0): https://pq-yang.github.io/projects/MatAnyone
- RobustVideoMatting (GPL-3.0): https://github.com/PeterL1n/RobustVideoMatting
