# BBG (Background Be Gone): scope

On-device video background removal for Apple Silicon Macs with 16 GB of unified memory. The user types or clicks the objects to keep, the app tracks them through the clip, and exports the result with the background removed or replaced.

**Status:** scope agreed 2026-10-30. The v1 pipeline, GUI and spike harness are built (see README.md for what is tested and how), but nothing has been measured with the real SAM 3.1 weights yet. The next step is the Phase 0 spike (§14). Speed figures quoted below come from papers, model cards and vendors, measured on other hardware. Every figure for our own targets is an estimate that the Phase 0 spike (§11) must confirm.

---

## 1. Decisions and assumptions

### Decided (2026-10-30)

| # | Decision | Consequence |
|---|---|---|
| D1 | **macOS only.** | Apple frameworks (MLX, AVFoundation/VideoToolbox, Metal) are available. No cross-platform layer is needed. |
| D2 | **The licence can be relaxed as needed.** It started as "fully unrestricted FOSS". | **SAM 3.1** (Meta's SAM License) becomes the core model. Its terms are in §5.2. |
| D3 | **macOS 27**, the latest release for an M3 Ultra Mac Studio. | Any Apple Silicon Mac on macOS 27 with ≥ 16 GB is in range. The M3 Ultra Studio starts at 96 GB, so it is the development machine, but it cannot show 16 GB problems (§7). |
| D4 | **Clean object edges are enough.** | No matting models. Edges come from upsampling the mask with edge-aware filtering. |
| D5 | **Typed prompts in v1, using SAM 3.1 or better.** | SAM 3.1 detects, segments and tracks from a short phrase. No separate detector is needed. |
| D6 | **Stack for v1: Python + MLX + Qt (PySide6) + PyAV**, the recommendation in §6. | A native SwiftUI rewrite stays a later, optional project. |
| D7 | **BBG's own code is open source under Apache-2.0** (`LICENSE`). | Only the SAM 3.1 weights, and code derived from SAM 3, carry Meta's terms (§5.2). |
| D8 | **Limits: up to 4K and up to 10 minutes per clip.** Inputs are phone or camera footage, H.264/HEVC/ProRes, SDR and HDR. | They set the memory budget (§7), the soak test (§11) and the test set (§12). |

### Assumed

| # | Assumption |
|---|---|
| A1 | **Offline processing, not live.** Selecting on one frame feels instant; tracking the whole clip runs as a job. |
| A2 | "Objects to keep" means things a phrase or a click can isolate. An object's shadows and reflections are not included (§13). |

## 2. What the user does

```
┌──────────────────────────────────────────────────────────────────┐
│ [Open…] interview.mov          Quality: Fast | Balanced  [Export]│
├────────────────────────────────────────────┬─────────────────────┤
│                                            │ Find: [dog        ] ⏎│
│                                            ├─────────────────────┤
│          viewer: current frame             │ Objects             │
│   click = add   ⌥-click = subtract         │ ■ person 1  [Keep ✓]│
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
   - **Type** a short phrase ("dog", "red car"). Every match becomes an object in the list, and the user unticks any they don't want.
   - **Click** adds a point (⌥-click subtracts one), or **drag a box** for anything the phrase missed.
   - Drawing a box around one example and choosing *Find similar* adds every look-alike.
3. **Track** forwards and backwards with a progress bar, a cancel button and a live preview. SAM 3.1 keeps detecting the typed phrases while it tracks, so a second dog walking in joins the list as dog 3.
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
| Open video, scrub, preview | ✓ | | |
| Typed phrase → objects; click/box select | ✓ | | |
| Bidirectional tracking with bounded memory | ✓ | | |
| Several objects with Keep toggles | ✓ | | |
| Transparent / colour background | ✓ | | |
| Export ProRes 4444, HEVC with alpha, PNG sequence, matte video | ✓ | | |
| New instances of typed phrases picked up during tracking | ✓ | | |
| Corrections on any frame with partial re-tracking | | ✓ | |
| Confidence flags on the timeline | | ✓ | |
| Save/resume a project | | ✓ | |
| *Find similar* (an example box as the prompt) | | ✓ | |
| Image / blur / video backgrounds; flattened H.264/HEVC export | | ✓ | |
| Full-resolution 4K edges, HDR passthrough | | ✓ | |
| Batch queue | | | ✓ |
| Native SwiftUI rewrite (§6) | | | ✓ |

## 4. Processing pipeline

```
decode ──► resize to ~1008² ──► shared backbone (once per frame)
(VideoToolbox                          │
 via PyAV)              ┌──────────────┴──────────────┐
                        ▼                             ▼
                   detector                       tracker
          (typed phrase / example box     (memory bank; Object Multiplex
           → every matching instance)      tracks up to 16 objects per pass)
                        └──────────┬──────────────────┘
                                   │  ◄── clicks / boxes (corrections)
                                   ▼
                     per-object low-res masks
                                   │
                     upsample to source res + guided filter
                                   │
                     temporal smoothing of alpha
                                   │
                     mask cache on disk (per object, per frame)
                                   │
           compositor (Metal) ──► encoder (VideoToolbox) + audio passthrough
```

**Streaming with bounded memory is the core engineering task on 16 GB.** SAM 3's tracker is inherited from SAM 2, and the reference SAM 2 video predictor is not built for streaming:

- `init_state` loads **every frame** of the clip up front as a float32 1024×1024 tensor (`sam2/utils/misc.py`: `torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)`). That is about 12.6 MB per frame, so a 1-minute 30 fps clip needs about 22 GB before tracking starts.
- It keeps every frame's memory features and mask logits for every object, although attention reads only the last few frames and the prompted frames. That is about 0.8 MB per frame per object.
- `offload_*_to_cpu` saves nothing with unified memory.

**The MLX port does not keep everything:** mlx-vlm 0.7.11's `track_step` prunes non-conditioning outputs older than the memory window, and BBG's tests check that a session's stored outputs stay bounded. Frame loading is BBG's own (the port's tracker never sees the clip as a whole). Phase 0 still measures the whole process's footprint over a 10-minute clip. The production loop:
- decode into a small ring buffer,
- keep only the frames inside the memory window plus the prompted frames,
- write each frame's mask to disk as soon as it is final,
- keep backbone features only for the frames being annotated (a small LRU).

## 5. Models

### 5.1 Choice

| Role | Model | Notes |
|---|---|---|
| **Detect, segment and track (core)** | **SAM 3.1** (Meta, 2026-03-27; ~873M params) | A detector and a tracker share one backbone, prompted by short phrases, example boxes, points or boxes. Meta reports SAM 3 roughly doubles the accuracy of earlier systems at concept segmentation, with zero-shot LVIS mask AP of 48.8 against 38.5. 3.1 adds **Object Multiplex**, which tracks objects jointly rather than one by one. |
| Runtime | **MLX** via `mlx-vlm`'s SAM 3.1 implementation (`mlx-community/sam3.1-bf16`) | The model card reports the tracker at **~25 fps sustained on an M3 Max**, against ~5 fps unoptimised. The multiplex mask decoder handles 16 objects at once, for 2.4–4× faster tracking. Resolution and object count are not stated, and I haven't verified the figures. A separate port (`sam3-mlx`) marks multiplex video tracking as experimental and incomplete. |
| Runtime fallback | PyTorch (MPS) via `transformers` `Sam3VideoModel` | Pure PyTorch, so it may run on MPS. Untested: Meta's own repo requires CUDA. |
| Fast tier (if needed) | SAM 3.1 at a lower input resolution, or SAM 2.1 Small (MLX port, Apache-2.0) for click-only tracking | Phase 0 decides whether a second model is worth carrying. |
| Edges | Guided filter on the upsampled mask | Enough for clean edges (D4). |
| Watch | **EfficientSAM3** (distilled SAM 3 students for devices, arXiv 2511.15833) | Results and code were "coming soon" at last check. It could become the Fast tier. |

**What text prompts can and can't do:** SAM 3 is built for **short noun phrases** ("dog", "yellow school bus"). Longer descriptions ("the man on the left holding a cup") are what Meta pairs with a separate language model ("SAM 3 Agent"). For those, the app's answer is to type "man", then untick or ⌥-click the wrong ones.

### 5.2 Licence obligations (SAM License)

The licence is royalty-free and allows use, modification and redistribution. In return:
- **ship a copy of the licence** with the weights, and with any SAM 3-derived code,
- **acknowledge SAM** in any publication,
- users must not use it for trade-controlled end uses, including military or warfare,
- no reverse engineering of the SAM Materials,
- indemnify Meta,
- Meta may change the terms.

**In practice:** the app downloads the weights on first run and shows the licence before the download. Meta's official checkpoints are **gated** on Hugging Face. Phase 0 decides between asking for the user's own Hugging Face token and hosting our own converted copy, which the licence allows if it is shipped with the licence attached. The MLX port is derived from SAM 3's code, so it falls under the same terms. BBG's own code stays Apache-2.0 (D7). The licence and attribution travel with the downloaded weights.

## 6. Technology stack

**Chosen for v1 (D6): Python + MLX + Qt (PySide6) + FFmpeg (PyAV), packaged as a signed `.app`.** The only working Apple Silicon implementation of SAM 3.1 is Python on MLX (`mlx-vlm`), so building on it directly is the fastest and lowest-risk route. Because inference is MLX, the bundle needs no PyTorch.

| Layer | Choice |
|---|---|
| Inference | MLX + `mlx-vlm` (pinned version; fixes sent upstream) |
| GUI | Qt 6 via PySide6 (trim unused Qt modules) |
| Video in/out | PyAV (FFmpeg): VideoToolbox hardware decode; `prores_videotoolbox` / `hevc_videotoolbox` encode |
| Image ops | NumPy, OpenCV (guided filter, morphology), Metal through MLX for compositing |
| Packaging | Briefcase or PyInstaller → `.app`, signed and notarised with an Apple Developer ID; weights downloaded on first run |

| Option | For | Against |
|---|---|---|
| **Python + MLX + PySide6** (chosen) | Reuses the existing SAM 3.1 port as is; one process, one language; quickest to a working tool. | Less native look and feel; a Python bundle of several hundred MB; every bundled dylib must be signed for notarisation. |
| SwiftUI + bundled Python/MLX helper process | Native UI and AVFoundation, with the model still in Python. | Two languages and IPC (frames and masks over shared memory). More moving parts than either pure option. |
| SwiftUI + **MLX Swift** + AVFoundation | Best end state: native, small, no Python. | SAM 3.1 has to be ported to Swift and checked for numerical parity: several weeks, and it must be redone whenever the Python implementation improves. **Recommended as a later rewrite, not v1.** |
| Swift + Core ML | Could use the Neural Engine. | Converting an 873M-parameter detector plus tracker, with a varying number of objects, is the hardest route. A community Core ML conversion exists but is unverified. |

## 7. Memory budget (16 GB machine)

macOS and other open apps can take 5–6 GB. **Target app peak ≤ 6 GB, hard ceiling 8 GB, no swap growth over a 10-minute clip.**

The development machine (M3 Ultra, ≥ 96 GB) won't show memory pressure, so the app caps itself with `mx.set_memory_limit` and logs its peak footprint on every run. **The final QA runs on a real 16 GB Mac** (any base M-series).

| Item | Estimate |
|---|---|
| SAM 3.1 weights | ~1.75 GB at bf16; ~0.9 GB at 8-bit (quantisation saves memory but should not be assumed to speed things up) |
| Backbone, detector and tracker activations at ~1008² | **To measure.** The largest unknown. |
| Decoded frame ring buffer (16 frames) | 4K BGRA 8-bit is 33 MB a frame, about 0.5 GB in total |
| Tracker state (memory window + prompted frames) | Bounded; to measure with Object Multiplex |
| Mask cache | On disk, not in RAM |
| Python, Qt, preview and compositing | ~0.5 GB |

The weights fit comfortably. Whether the whole app stays under 6 GB depends on activations and tracker state, which is **the first thing Phase 0 measures**. If it doesn't fit: use 8-bit weights, a smaller input resolution for the Fast tier, and fewer objects per multiplex pass.

## 8. Performance targets (to validate, not yet known)

Targets, on a 16 GB base M-series Mac:

| Measure | Target |
|---|---|
| Typed phrase to objects on the current frame | < 1 s |
| Click to mask | < 300 ms once the frame has been encoded |
| Tracking, Balanced, 1080p | ≥ 5 fps, so a 1-minute 30 fps clip takes ≤ 6 min |
| Tracking, Fast, 1080p | ≥ 12 fps |
| 4K | Tracked at model resolution, edges refined at full resolution; ≥ 2 fps |

The only Apple Silicon figure available is the model card's ~25 fps on an **M3 Max**, which has 30–40 GPU cores. Base M3/M4 chips have 8–10. If tracking is GPU-bound, expect roughly a quarter to a third of that speed, which is close to the Balanced target; it has to be measured. The M3 Ultra (60–80 GPU cores) will be much faster than any target machine. Fanless MacBook Airs throttle on long jobs, so the app needs pause/resume and checkpointing.

## 9. Export

| Format | Encoder | Use |
|---|---|---|
| ProRes 4444 with alpha (`.mov`) | VideoToolbox (hardware on chips with a ProRes engine), otherwise `prores_ks` | Final Cut, Premiere, DaVinci Resolve |
| HEVC with alpha (`.mov`) | VideoToolbox | Small files; Final Cut, Safari, Keynote, Motion |
| PNG / TIFF sequence with alpha | FFmpeg | Any tool |
| Matte-only video (greyscale) | ProRes 422 or HEVC | Track matte in any editor |
| Flattened H.264 / HEVC | VideoToolbox | Sharing, with a colour, image, blur or video background |
| WebM VP9 with alpha | libvpx | Web; later |

For all formats:
- pass the audio through,
- keep the frame rate (including variable frame rate), rotation metadata and colour metadata.

For HDR, the model sees an SDR conversion, and the output keeps the original pixels and transfer function.

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| The SAM 3.1 MLX port is young; multiplex video is marked experimental in one port | High | Phase 0 tests the `mlx-vlm` version on the full test set. Pin versions, send fixes upstream, keep `transformers` on MPS as the fallback. |
| Tracker memory grows with clip length (§4) | High | Own tracking loop with a bounded window, eviction and a disk cache. Soak test on 10-minute clips. |
| A base chip is too slow for Balanced | Med | 8-bit weights, a lower input resolution, a Fast tier, honest time estimates. |
| Phrases match the wrong things or miss some | Med | Unticking and ⌥-click fixes; *Find similar*; clicks for anything a phrase can't express. |
| SAM License: terms can change, licence must be shipped, gated weights | Low–Med | First-run licence screen; pin the checkpoint version and keep its licence text alongside it. |
| Edge flicker over time | Med | Smooth alpha over time; measure flicker in QA. |
| Python packaging and notarisation | Med | Briefcase build in CI from the start; measure bundle size in Phase 0. |
| Thermal throttling | Med | Pause/resume, checkpoints, adaptive time estimates. |
| Expecting shadows and reflections to be kept | Expectation | State it in the UI and docs (§13). |

## 11. Phases and effort

| Phase | Weeks | Delivers | Exit criterion |
|---|---|---|---|
| **0. Spike** | 1–2 | On the M3 Ultra (capped at 6 GB) and a 16 GB Mac: SAM 3.1 through `mlx-vlm` (bf16 and 8-bit), with `transformers` on MPS as the fallback. Peak memory, whether memory grows over a 10-minute clip, fps at 1080p and 4K, J&F, phrase recall, flicker. Weight distribution route decided. | Runtime and precision chosen from **measured** numbers; go/no-go on the 16 GB target; Fast tier decided. |
| **1. MVP** | 4–5 | Open; typed phrase plus click/box selection; bidirectional tracking with bounded memory; feathered masks; transparent/colour background; ProRes 4444 / HEVC-alpha / PNG / matte export. | A 10-minute 1080p clip finishes inside the memory budget on a 16 GB Mac. |
| **2. Usable tool** | 3–4 | Corrections with partial re-tracking, confidence timeline, project save, *Find similar*, image/blur/video backgrounds, flattened export. | A non-expert gets an acceptable result on the test set without help. |
| **3. Quality** | 1–2 | Full-resolution 4K edges, temporal smoothing, HDR handling. | Flicker and edge metrics meet the bar set in Phase 0. |
| **4. Ship** | 2–3 | Signed, notarised `.app`; first-run model download with the licence screen and checksums; docs. | Release candidate. |

**Total: about 11–16 weeks for one experienced engineer.** This is less than the previous draft, because there is no Linux, no separate detector and no matting. It is a rough estimate until Phase 0 reports.

A later native rewrite (SwiftUI + MLX Swift) is a separate project of roughly two months, worth doing only for App Store distribution or a fully native feel.

**Skills:** Python and MLX (model internals well enough to change the tracking loop), Qt, FFmpeg/VideoToolbox and video basics (codecs, colour, timing), and macOS signing and notarisation.

**Hardware:** the M3 Ultra Mac Studio for development, and a real 16 GB Apple Silicon Mac (ideally a fanless Air) for QA.

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

**Phrase set:** about 30 short phrases over those clips.

**Measures:**
- J&F against hand-drawn masks on about 20 frames per clip,
- phrase recall and precision,
- flicker (dtSSD on the alpha),
- fps,
- peak memory and swap,
- the number of corrections until the user is satisfied.

**MVP acceptance:**
- the whole test set completes on a 16 GB Mac with no out-of-memory failure or swap growth,
- exports open in Final Cut Pro and DaVinci Resolve,
- audio stays in sync.

## 13. Out of scope (v1)

- Live camera or virtual-webcam use.
- Hair-level matting and semi-transparency (D4).
- Keeping an object's shadows, reflections or smoke (the "omnimatte" research area).
- Removing an object and filling the hole (inpainting).
- Long descriptive prompts that need a language model (§5.1).
- Linux, Windows, iPad; cloud processing; training models.

## 14. Next step: Phase 0 spike

No scope decisions remain open. Phase 0 answers the technical unknowns, and its exit criterion (§11) decides go/no-go on the 16 GB target.

**What it needs:**
- **The M3 Ultra Mac Studio and a 16 GB Apple Silicon Mac.** MLX runs on Metal, so the spike can't run in a Linux container.
- **The test clips and phrase set** (§12).
- **A Hugging Face account** with access to Meta's gated SAM 3.1 checkpoint, or the `mlx-community` conversion.

**Order:**
1. Memory first: peak footprint, and whether memory grows over a 10-minute clip.
2. Then speed at 1080p and 4K, bf16 and 8-bit.
3. Then quality: J&F, phrase recall, flicker.

The spike writes one JSON report per run. The numbers that come back replace the estimates in §7 and §8.

## Sources

- SAM 3 / 3.1 repo (release notes, CUDA requirement, licence): https://github.com/facebookresearch/sam3 · https://github.com/facebookresearch/sam3/blob/main/LICENSE
- SAM 3 paper: https://arxiv.org/abs/2511.16719
- SAM 3.1 MLX build and M3 Max figures: https://huggingface.co/mlx-community/sam3.1-bf16
- `sam3-mlx` port (multiplex video experimental): https://pypi.org/project/sam3-mlx/
- `transformers` SAM 3 video: https://huggingface.co/docs/transformers/model_doc/sam3_video
- SAM 2 repo (frame preloading in `sam2/utils/misc.py`): https://github.com/facebookresearch/sam2
- SAM 2.1 MLX port: https://github.com/eisneim/sam2.1_mlx
- EfficientSAM3: https://arxiv.org/abs/2511.15833
- MLX: https://github.com/ml-explore/mlx
- PyAV: https://github.com/PyAV-Org/PyAV
- macOS 27 compatibility (Apple Silicon only): https://www.macrumors.com/2026/04/18/macos-27-compatibility-change/
