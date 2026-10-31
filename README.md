# BBG — Background Be Gone

BBG removes the background from a video on your Mac. Type what you want to
keep ("dog", "person"), or click on it. BBG follows it through the whole
clip. You fix any moment it got wrong, and save the result with the
background gone or replaced. Everything happens on your Mac; nothing is
uploaded.

![BBG's window](docs/screenshot.png)

> **Early version.** BBG hasn't been tried on a Mac yet: it was built and
> tested on another computer. If anything below doesn't go as described,
> write down what the screen says (or take a screenshot).

---

## Contents

1. [What you need](#what-you-need)
2. [Install BBG](#install-bbg-once-about-15-minutes): four steps, about 15 minutes, once
3. [Start BBG](#start-bbg)
4. [Remove a background](#remove-a-background-step-by-step): step by step
5. [Update BBG](#update-bbg) · [Uninstall](#uninstall-bbg)
6. [If something goes wrong](#if-something-goes-wrong)
7. [Technical details](#technical-details) (for developers)

---

## What you need

Check these first: Apple menu (top-left corner of the screen) → **About This Mac**.

| You need | Where to look in About This Mac | |
|---|---|---|
| An **Apple chip**: M1, M2, M3, M4, M5 or later | **Chip** says "Apple M…" | If it says "Intel", BBG can't run on this Mac |
| **macOS 14** (Sonoma) or newer | **macOS** line | Older? Update it: System Settings → General → Software Update |
| **16 GB** of memory or more | **Memory** | Less works, but slowly |

You also need:
- **4 GB of free disk space.**
- **An internet connection** while installing. After that, BBG works offline.
- **Your GitHub sign-in.** BBG's page is private.

---

## Install BBG (once, about 15 minutes)

### Step 1 — Install Python

BBG is written in Python, so the Mac needs a recent copy of it.

1. Go to **[python.org/downloads](https://www.python.org/downloads/)**.
2. Click the big yellow **Download Python 3.…** button.
3. Open the file it downloads (it's in your **Downloads** folder, called something like `python-3.14.0-macos11.pkg`).
4. Click **Continue**, **Continue**, **Agree**, **Install**. Type your Mac password if it asks, then **Close**.

A Finder window called "Python 3.…" may open at the end. You can close it.

> Already have Python? The installer in step 3 checks it and tells you if it's not new enough.

### Step 2 — Download BBG

1. Go to **[github.com/ParadoxTurnedSolution/BBG](https://github.com/ParadoxTurnedSolution/BBG)** and sign in if GitHub asks.
2. Click the green **<> Code** button, then **Download ZIP**.
3. Look in your **Downloads** folder:
   - If there is a folder called **`BBG-main`**: good, go to step 3.
   - If there is a file called **`BBG-main.zip`**: double-click it, and the `BBG-main` folder appears next to it.

Don't open or move anything inside the folder.

### Step 3 — Run the installer

1. Open **Terminal**: press **⌘ Space**, type **Terminal**, press **Return**.
   A window with a line of text and a blinking cursor opens.
2. Copy this line. (On GitHub, hover over the box and click the copy icon on its right.)

   ```bash
   bash ~/Downloads/BBG-main/install.sh
   ```

3. Click in the Terminal window, paste with **⌘ V**, and press **Return**.

The installer checks your Mac, moves BBG into your home folder (as a folder
called **BBG**), and installs what BBG needs. Text scrolls past for 5 to 10
minutes; that's normal. It shows its progress as **1/4** to **4/4**.

**At 4/4 it asks about the model.** BBG uses Meta's **SAM 3.1** model to
recognise objects (about 1.8 GB, downloaded once). It shows Meta's licence
terms. The model may be used freely, but not for military or weapons
purposes. To accept, type **y** and press **Return**, then wait for the
download.

**It's finished when it says `Done: BBG is installed.`** A Finder window
opens showing your BBG folder. You can close Terminal.

If it stops with a red **✗** line, read the message under it: it says what
to do. The most common one is "Python 3.11 or newer isn't installed" (do
step 1, then paste the same line again).

### Step 4 — Make BBG easy to find (optional)

In the **BBG** folder that opened, find **`BBG.command`**. Drag it into the
right-hand end of the **Dock** (next to the Trash). From then on, one click
there starts BBG.

---

## Start BBG

Double-click **`BBG.command`**. It's in the **BBG** folder in your home folder:
in Finder, **Go → Home → BBG**.

- **Two windows open:** a Terminal window, then BBG itself. **Leave the Terminal window open** while you use BBG. It shows what BBG is doing, and any error messages. Close it after you quit BBG.
- **The first start takes a little longer** while the model loads.
- **If you skipped the model download** during install, BBG asks now. Click **Yes**.

---

## Remove a background, step by step

1. **Open your video.** Click **Open video…** at the top left and choose the clip.
   - BBG makes a folder next to the video, named after it (e.g. `holiday.bbg`), where it keeps its work.
   - If you open the same video later, BBG asks whether to continue where you left off.

2. **Tell BBG what to keep.** Either:
   - **Type it.** In the **Find** box on the right, type a short word or two ("dog", "person", "red car") and press **Return**. Everything that matches gets a coloured outline and appears in the **Objects** list.
   - **Click it.** Click on the thing in the picture. Click again on parts it missed. **⌥ Option-click** on parts it shouldn't have included.
   - **Draw round it.** Press and drag a box around it.

   To start a *second* object by clicking, press **+ New** first. Otherwise your click adds to the object that's selected in the list.

3. **Press ▶ Track** (top). BBG follows every object through the whole clip. A bar at the bottom shows the progress and the time left. You can keep watching; **Cancel** stops it.

4. **Check the result.** Use the controls under the picture:

   | Control | What it does |
   |---|---|
   | **◀ ▶** (or the ← → keys) | Step one frame back or forward |
   | **▶ Play** (or Space) | Play the clip |
   | **⚑◀ ▶⚑** | Jump to the moments BBG marked as doubtful (red marks on the timeline) |
   | The **timeline** | One coloured bar per object, showing where it is in the clip. Click anywhere on it to jump there |

5. **Fix any mistakes.** Go to a frame where an outline is wrong. Select the object in the list, then click on the parts it missed (⌥ Option-click for parts it shouldn't have). Press **▶ Track** again. Only the part of the clip near your fix is redone.
   - **Undo fix** removes your correction on the current frame.
   - If an object shows up twice in the list (e.g. "dog 1" and "dog 2" for the same dog), select one, click **Merge into…** and choose the other. Then Track again.

6. **Choose what stays.** The tick box next to each object decides whether it stays in the final video. Untick anything you want removed with the background.

7. **Choose the background** (right-hand side). The **Result** view shows how it will look.

   | Background | Result |
   |---|---|
   | **Transparent** | No background at all, for putting the clip over something else in an editing app |
   | **Colour** | A plain colour (click **Choose…**). Green is the default |
   | **Image** | A picture of your choice (click **Choose…**) |
   | **Blur** | The original background, blurred |
   | **Video** | Another video (click **Choose…**) |

   **Edges:** *Grow / shrink* moves the outline outwards or inwards. *Feather* softens it. *Smoothing* calms flickering edges.

8. **Save the result.** Click **Export…**, choose a format, and click **OK**. The new video is saved next to the original unless you choose somewhere else.

   | If you want to… | Choose |
   |---|---|
   | Edit it further in Final Cut, Premiere, DaVinci Resolve or iMovie | **ProRes 4444 with alpha** (big files, keeps transparency) |
   | Use it in Apple apps (Keynote, Final Cut, Motion) with smaller files | **HEVC with alpha** |
   | Post or send it, with the new background | **H.264** (plays everywhere) or **HEVC** (smaller). Pick a background other than *Transparent* first |
   | Put it on a website | **VP9 with alpha (.webm)** |
   | Get every frame as a picture | **PNG sequence** |
   | Get the black-and-white cut-out shape only | **Matte only** |

BBG saves its work as you go. The `.bbg` folder next to your video can be
deleted once you're happy with the export; it can be large.

---

## Update BBG

Updating is the same as installing:

1. Download BBG again ([step 2](#step-2--download-bbg)), so that a fresh `BBG-main` folder is in Downloads.
2. In Terminal, paste the same line as before and press **Return**:

   ```bash
   bash ~/Downloads/BBG-main/install.sh
   ```

3. When it says **Done**, there is a folder called **BBG-old-** followed by a
   date in your home folder: that's the previous version. Once the new one
   works, drag it to the Trash.

The model isn't downloaded again, and your projects (the `.bbg` folders next
to your videos) are untouched.

If you put BBG.command in the Dock, drag that Dock icon out (it may still
point at the old copy) and drag in the new **BBG.command** from the BBG folder.

## Uninstall BBG

1. Drag the **BBG** folder (in your home folder) to the Trash, and any **BBG-old-…** folders. If BBG.command is in the Dock, drag it out.
2. To free the model's 1.8 GB too, paste this into Terminal and press **Return**:

   ```bash
   rm -rf ~/.cache/huggingface/hub/models--mlx-community--sam3.1-bf16
   ```

3. Delete any `.bbg` folders next to your videos.

---

## If something goes wrong

| What happens | What to do |
|---|---|
| Terminal says **`No such file or directory`** | The `BBG-main` folder isn't in Downloads. Do [step 2](#step-2--download-bbg) again, and double-click the ZIP if there is one. If your browser saved it as `BBG-main 2`, delete the old `BBG-main` folder and rename the new one |
| **"There is already a folder called BBG in your home folder, and it isn't BBG"** | You have a folder of your own called BBG. Rename it (click its name in Finder, press Return, type a new name), then paste the install line again |
| **"Python 3.11 or newer isn't installed"** | Do [step 1](#step-1--install-python), then paste the install line again |
| **"This Mac doesn't have an Apple chip"** | BBG can't run on Intel Macs |
| **"Terminal is running under Rosetta"** | Follow the instructions it prints, then try again |
| **"Something went wrong"** while installing | Usually the internet connection. Paste the same line again |
| The model download says **401**, **403**, or asks you to **log in** | Create a free account at [huggingface.co](https://huggingface.co), open the [model's page](https://huggingface.co/mlx-community/sam3.1-bf16) and accept its terms if asked, then make a token (your picture → Settings → Access Tokens → Create, type "Read"). Paste `cd ~/BBG && .venv/bin/hf auth login` into Terminal, paste the token when asked, then `.venv/bin/bbg-cli download-model` |
| **CERTIFICATE_VERIFY_FAILED** in Terminal | In Finder, open Applications → Python 3.… and double-click **Install Certificates.command**. Then try again |
| Double-clicking BBG.command says it **"can't be opened"** or is from an **"unidentified developer"** | Open System Settings → Privacy & Security, scroll down, click **Open Anyway** next to BBG.command. Or paste `bash ~/BBG/BBG.command` into Terminal |
| BBG.command says **"BBG isn't installed yet"** | Paste `bash ~/BBG/install.sh` into Terminal |
| BBG.command says the folder **"was moved or renamed"** | Type **y** and press Return: it repairs itself in a few minutes |
| BBG is **very slow**, or the Mac is struggling | Quit other apps (especially web browsers with many tabs). Choose **Fast** next to *Quality* before pressing Track |
| An outline is **wrong in a few places** | Fix it on the worst frame and Track again ([step 5](#remove-a-background-step-by-step)) |
| **Shadows or reflections** aren't kept | BBG keeps the object itself, not its shadow |
| Anything else | The Terminal window that opened with BBG shows the error. Copy it, or take a screenshot |

---

## Technical details

Everything below is for developers.

### Status

The v1 pipeline and GUI are built. It has **not yet run with the real SAM 3.1
weights**: that is Phase 0 of [SCOPE.md](SCOPE.md), which has to run on a
Mac ([spike/](spike/README.md)). The screenshot above uses the synthetic test
backend, not SAM 3.1.

| Part | State | How it was checked |
|---|---|---|
| Video in/out: exact frame index, VFR, rotation, backwards decoding | done | tests, against the FFmpeg CLI |
| Export: ProRes 4444 / WebM with alpha, PNG, matte, H.264/HEVC flattened, audio passthrough | done | tests decode the output and check alpha, timing and audio |
| HEVC with alpha (`hevc_videotoolbox`) | written | **untested**: needs macOS |
| Tracking: keyframes, nearest-keyframe ownership, partial re-tracking, re-detection of typed phrases, flags, merge | done | tests with the synthetic backend (J > 0.95 against ground truth) |
| SAM 3.1 backend (mlx-vlm 0.7.11) | done | `tests/test_sam3_backend.py` (3 tests, passing) runs mlx-vlm's real SAM 3.1 code with random weights in a shallow, narrow config. It checks phrase detection, clicks with refinement, session start, object order on add, that tracker memory stays bounded, and the full tracker loop through the backend. **Mask quality and speed: not measured** |
| GUI (PySide6) | done | offscreen end-to-end test: select, track, view modes, export |
| `install.sh`, `BBG.command` | done | shellcheck. Run on Linux with the Mac checks and model download skipped, in the flows the README describes: install from Downloads, start, start after renaming the folder (self-repair), update with the same command, a foreign `~/BBG` left alone. **Not yet run on a Mac** |
| Phase 0 spike harness | done | dry run with the synthetic backend |
| macOS `.app` | config only | `[tool.briefcase]` in pyproject; not built |

### Developer install

```bash
git clone https://github.com/ParadoxTurnedSolution/BBG.git && cd BBG
```

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[sam3,gui,dev]"
```

Every dependency installs from prebuilt arm64 wheels on Python 3.11 to 3.14
(nothing to compile). MLX needs macOS 14 or later.

### Without the model

```bash
bbg-cli demo /tmp/bbg-demo
```

```bash
bbg --backend synthetic /tmp/bbg-demo/demo.mp4
```

The synthetic backend segments the generated test clips by colour ("ball",
"box", "dog"). It exists for tests and for trying the interface; it can't
segment real footage.

### Command line

```bash
bbg-cli track clip.mov --phrase person --click 812,430@120 --report track.json
bbg-cli export clip.bbg out.mov --format prores4444
bbg-cli export clip.bbg out.mp4 --format h264 --background blur
bbg-cli export clip.bbg out.mp4 --format hevc --background image:beach.jpg --keep "person 1"
```

Formats: `prores4444`, `hevc_alpha` (macOS), `webm`, `png` (with alpha);
`matte`; `h264`, `hevc` (need a background). Click and box coordinates are
source pixels, `@frame` optional.

### How it works

```
src/bbg/
  video/reader.py    exact frame index from a packet scan; forwards and chunked-backwards decode
  video/writer.py    export formats, source timestamps, audio passthrough
  backends/          the Segmenter interface; SAM 3.1 on MLX; the synthetic test backend
  selection.py       phrase / click / box / find-similar on one frame → objects with keyframes
  tracking.py        planning (who owns each frame) and the tracking loop, re-detection, flags
  masks/             on-disk mask cache; guided upsampling, grow/shrink, feather, temporal smoothing
  export.py          kept objects over a background, full resolution
  project.py         the .bbg project folder, merging objects
  gui/               PySide6 window; the engine thread that owns the model
  cli.py, metrics.py, memory.py, synthetic.py
spike/               Phase 0 measurement harness
install.sh, BBG.command   the end-user installer and launcher
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
- It is then sharpened to a clean edge, and grow/shrink and feather are applied.

### Tests

```bash
QT_QPA_PLATFORM=offscreen pytest
```

```bash
pytest -m slow
```

The first is the fast suite: synthetic backend, GUI offscreen. The second
runs mlx-vlm's real SAM 3.1 code with random weights; on Linux, install
`mlx[cpu]` first. MLX's Linux CPU build is single-threaded, so the slow
tests take about 35 minutes there; on a Mac they run on Metal.

`mlx-vlm` is pinned: `backends/sam3_mlx.py` calls its SAM 3.1 internals, and
the docstring there lists what it relies on.

### Licences

- **BBG's code:** Apache-2.0 ([LICENSE](LICENSE)).
- **SAM 3.1 weights:** Meta's SAM License, not Apache-2.0. They are downloaded at first run, after a prompt, and never shipped with BBG.
- **mlx-vlm's SAM 3.1 port:** mlx-vlm itself is MIT, but this port is derived from Meta's SAM 3 code.
- **FFmpeg:** PyAV's prebuilt wheels bundle an FFmpeg build that includes GPL encoders (x264, x265). Before distributing a packaged `.app`, either meet the GPL for that bundle or build PyAV against an LGPL FFmpeg.

### Known limitations

- **No real-weight results yet.** Mask quality, speed and peak memory on a 16 GB Mac are unmeasured until Phase 0.
- **Short phrases only.** SAM 3.1 is built for short phrases. For "the man on the left", type "man" and untick the others.
- **Duplicate objects.** Re-detection can create a duplicate when a tracked object re-emerges somewhere its track didn't follow: it appears as e.g. "ball 2". Both are kept by default, so the output is right. Use *Merge into…* to tidy the list.
- **Fast quality is approximate.** It tracks every other frame and averages the skipped frames from their neighbours. Fast motion will lag on those frames.
- **Shadows and reflections aren't included.** They aren't part of the object (SCOPE §13).
