# AOI Comparative Benchmark

Automated optical inspection of 4K PCB images, built to answer one question with
evidence rather than assertion:

> **Can Python replace C++ in a production 4K AOI pipeline, and if not, where
> exactly does the time go?**

Two inference paths run the same SAHI-tiled YOLOv8 workload on the same pixels,
under the same thresholds, and are compared with interval estimates, a
significance test, an effect size and a stage-by-stage attribution of the gap.

```bash
pip install -r requirements.txt
python main.py verify     # assert the invariants the comparison rests on
python main.py            # launch the GUI
python main.py bench      # headless study -> workbook + publication figures
```

Nothing above needs weights, a GPU or a camera. A synthetic 4K board with
planted defects stands in, and it carries ground truth, so precision and recall
are computable on any machine.

---

## 1. What the study actually measures

### Stage-resolved timing

Both paths report the same eight stages per frame, so the headline ratio can be
decomposed instead of merely quoted:

| Stage | What it covers | Why it is separate |
|---|---|---|
| `slice` | computing the tile grid | should be ~0 on both; a canary |
| `marshal` | moving pixels across the language boundary | **the study's core question**: `np.ascontiguousarray` on one side, a free `cv::Mat` ROI view on the other |
| `preprocess` | letterbox + blob/normalise | identical code on both sides |
| `inference` | forward pass, device-synchronised | the part that is not about language |
| `decode` | raw tensor → candidate boxes | vectorised on both sides, so it measures the runtime, not the loop style |
| `nms` | per-tile suppression | |
| `merge` | cross-tile suppression | |
| `dispatch` | **residual**: frame total minus every named stage | framework overhead that cannot hide inside a neighbouring stage |

A result where 37% of the gap sits in `marshal` supports a very different
conclusion from one where it sits in `inference` — and the report says which.

### Statistics, not a single number

- 95% confidence interval on every mean (Student's *t*, not a normal
  approximation — sample sizes here are 30 or fewer per trial)
- **Welch's** unequal-variance *t*-test: the interpreted path's tail is fatter,
  and pooling variances would understate the standard error
- **Cliff's δ** effect size, so "significant" is never mistaken for "large"
- **Bootstrap CI on the speedup ratio itself** (4000 resamples) — the ratio of
  two means has no closed-form standard error, so "1.65× faster" would otherwise
  be quoted with no uncertainty at all
- CV, p95 and p99, because an inspection line cares about the tail
- **Between-trial CV**, which flags a machine that was not thermally settled

There is no SciPy dependency. The *t* tail is evaluated through a continued
fraction for the regularised incomplete beta function, and `python main.py
verify` checks it against SciPy reference values to 8 decimal places.

### Detection parity — the guard that makes the speed claim legal

A path that finds fewer defects is trivially faster. Every study IoU-matches the
two paths' outputs and reports the agreement (Jaccard over matched boxes). Below
90%, the GUI and the report both say the speed claim is unsafe.

### Curves, not points

Optional sweeps repeat the measurement across tile geometries and input
resolutions. A gap that is fixed overhead flattens as the tile count rises; one
that is per-dispatch does not. A single operating point cannot tell them apart.

---

## 2. The test paths

| Key | Path | Runtime |
|---|---|---|
| `python` | **Path A** | Ultralytics `predict()` on `.pt` — idiomatic Python, framework included |
| `python-lean` | **Path A′** | Same PyTorch weights, but pre/post-processing is the *same code* the C++ path uses |
| `cpp` | **Path B** | Compiled C++17, ONNX Runtime + CUDA EP, via pybind11 |
| `sim` | reference | No weights. Draws from the synthetic board's ground truth with a controllable miss and false-alarm rate. Never a measurement, and labelled as such everywhere |

**Path A′ is an ablation, not a third test path.** It exists because Path A vs
Path B confounds two variables — PyTorch vs ONNX Runtime, and Python vs C++.
The A→A′ gap is framework overhead; the A′→B gap is what remains for runtime and
language. Without it the headline ratio cannot be attributed, and the report says
so in the caveat box on the Method tab.

---

## 3. Interface

Three tabs over one controller.

**Inspect** — a 4K canvas with wheel zoom and drag pan. Overlays are drawn in
widget space over the raw frame, not baked into pixels, so box strokes stay one
pixel wide at any magnification and layers toggle without re-running inference:
detections, labels, **the SAHI tile grid** (the one genuinely interesting part of
this pipeline, previously invisible), and ground truth. A defect gallery shows a
cropped patch per detection, worst confidence first; clicking one drives the
canvas to it. Live latency trend beneath.

**Benchmark** — study design (which paths, trials, frames, sweeps), a progress
bar, and on completion: the verdict with its CI and *p*-value, the dominant
stage, the parity check, a stage-decomposition bar, a latency histogram and the
statistics table. Exports the workbook and the figures.

**Method** — environment and versions, the controls applied to keep the
comparison valid, the stage attribution table, detection quality vs ground truth,
and the runtime-vs-language caveat.

Themes are one QSS template plus a token table, so the two palettes cannot drift.

---

## 4. Layout

```
main.py                  entry point + every CLI subcommand
aoi/
  config.py              RunConfig (one object both paths read) + Windows DLL/CUDA bootstrap
  pipeline.py            SAHI geometry, letterbox, NMS, IoU matching, PR/AP, parity
  synth.py               4K PCB renderer + ground-truth defect injection
  engines.py             sim / python / python-lean / cpp behind one interface
  analysis.py            statistics + study runner + parameter sweeps
  report.py              multi-sheet workbook + 300 dpi figures (PNG and vector PDF)
  workers.py             camera thread, inference thread, controller
  ui/widgets.py          board canvas, QPainter charts, KPI cards, defect gallery
  ui/window.py           the three tabs
cpp_backend/
  aoi_engine.hpp/.cpp    the engine, no Python
  bindings.cpp           pybind11 module (aoi_engine_cpp)
  bench_cli.cpp          standalone benchmark, no interpreter in the process
  CMakeLists.txt
ui/theme.qss             one stylesheet, @token@ substituted per theme
datasets/weights/        yolov8_pcb_4k.pt / .onnx / .manifest.json
reports/                 CSV, workbooks and figures land here
```

The C++ side compiles once as an object library and links into both targets.
`aoi_bench` exists so the question "does sharing an address space with the
interpreter contaminate the C++ numbers?" can be answered by measurement rather
than assumed away.

---

## 5. Commands

| Command | Effect |
|---|---|
| `python main.py` | GUI, simulation engine |
| `python main.py --engine cpp --theme light` | GUI with Path B preselected |
| `python main.py --tile "3x3 (true 25%)"` | override the SAHI grid |
| `python main.py bench` | headless study on every available path |
| `python main.py bench --trials 5 --frames 50 --sweep-all` | full study with both sweeps |
| `python main.py bench --no-cuda` | CPU baseline |
| `python main.py verify` | assert the parity and statistics invariants |
| `python main.py doctor` | environment, available paths, SAHI grid table, cameras |
| `python main.py doctor --max-index 10` | widen the camera probe |
| `python main.py board out.png --count 20 --golden` | synthetic boards + YOLO label files |
| `python main.py export-onnx` | `.pt` → `.onnx` with a SHA-256 manifest |

`board` writes normalised YOLO labels alongside each image, so the same
generator that feeds the benchmark can produce a training set with exactly the
defect appearance the demo shows.

---

## 6. Building Test Path B

```bash
cd cpp_backend
cmake -B build -S . -A x64 ^
      -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib" ^
      -DONNXRUNTIME_ROOT="C:/onnxruntime-gpu" ^
      -DPython_EXECUTABLE="C:/path/to/python.exe" ^
      -DUSE_NVML=ON
cmake --build build --config Release

python main.py doctor
```

`doctor` must report `cpp_providers` containing `CUDAExecutionProvider`.

Import it through `bootstrap()`, never bare — a plain
`python -c "import aoi_engine_cpp"` fails with `ModuleNotFoundError` because the
extension lives in `aoi/`, not the project root, and even once found it cannot
resolve `onnxruntime_providers_cuda.dll` until the CUDA directories are
registered. Python 3.8+ ignores `PATH` for a native extension's dependencies:

```powershell
python -c "from aoi.config import bootstrap; bootstrap(); import aoi_engine_cpp as m; print(m.__version__, m.available_providers())"
```

The extension must be built against the **same interpreter** that runs the GUI,
or the import fails with an ABI error. CMake drops it into `aoi/` together with
the OpenCV and ONNX Runtime DLLs it needs.

CMake warns at configure time if `onnxruntime_providers_cuda.dll` is absent —
the CPU-only ORT package links and runs exactly like the GPU one and then never
touches the GPU, which would turn the whole study into a GPU-vs-CPU measurement
reported as a language one.

Export the ONNX from the **same training run** as the `.pt`:

```bash
python main.py export-onnx     # writes a .manifest.json with both SHA-256 hashes
python main.py verify          # re-checks those hashes against the files on disk
```

Use `--opset 12` unless you have a reason not to; OpenCV DNN frequently fails to
parse opset 17+.

---

## 7. What keeps the comparison honest

These are deliberate and easy to break by accident. `python main.py verify`
checks the first three by execution, not by inspection.

| Guard | Consequence of losing it |
|---|---|
| `axis_origins` identical in both languages | the paths process different pixels |
| `letterbox` identical | a plain resize squashes a 16:9 tile and changes which fine-pitch defects survive |
| Weight manifest hashes match the files on disk | weight drift is reported as a language effect |
| One `RunConfig` read by every engine | a threshold silently differs between paths |
| Warmup **and** settling frames discarded on both | the first passes pay for cuDNN algorithm search and workspace allocation |
| `torch.cuda.synchronize()` per tile / `Session::Run` synchronising | otherwise you time the kernel *launch*, not the work |
| Annotation outside every timer | otherwise you compare two drawing routines |
| CPU normalised by logical core count on both sides | 1200% on one path, 100% on the other |
| Detection parity ≥ 0.9 | a path that finds less is trivially faster |
| `dispatch` reported as an explicit residual | framework overhead hides inside whichever stage brackets it |

---

## 8. The 4K tile-count trap

With 1920×1080 tiles on a 3840×2160 frame, **any non-zero overlap forces a 3×3
grid and the real overlap lands at 50%, not the 20% requested.** 3840 is exactly
2×1920, so two columns give 0% overlap and three is the next possible value.

The cost is direct: **nine inferences per frame instead of four.** The GUI shows
requested-vs-effective overlap on the Inspect tab and turns the chip amber when
they diverge; `python main.py doctor` prints the same table for every preset.

Tile sizes in `TILE_PRESETS` are solved for the grid they claim —
`t = W / (n − (n−1)·r)` — rather than guessed at round numbers, which is how the
1920×1080 case arose in the first place.

---

## 9. Threading model — do not reintroduce a `while` loop

Both workers are `QObject`s moved onto a `QThread` with `thread.started`
connected to `run()`. Because the worker lives on that thread the call is
**direct**: `run()` executes *before* `QThread.exec()` starts the event loop.

A blocking `while self._running:` there breaks two things at once:

- **queued cross-thread slots are never delivered** — `enqueue`, `set_engine`,
  `request_study` all silently no-op, so no inference runs;
- **`thread.quit()` is discarded**, because no event loop exists to receive it.
  `run()` returns, `exec()` starts a loop nobody will quit, `wait()` times out,
  `terminate()` fires, and the process has to be killed from Task Manager.

So `run()` arms a self-rearming `singleShot` chain (camera) or returns
immediately (AI). `Controller.stop()` writes the plain `_running` flags first —
visible without an event loop, so a long study aborts promptly — then `quit()`,
then a bounded `wait()`, then cleanup once the threads have joined. If a hang
ever recurs: `python main.py --debug-shutdown`, and a `faulthandler` watchdog
dumps every thread's stack after 12 s and names the blocking call.

**Windows COM:** `cv2.VideoCapture` opens the camera through DirectShow or Media
Foundation, both COM-based, and Qt does not call `CoInitializeEx` on worker
threads. Some drivers' filter graphs then crash with `0x8001010D` — a native
access violation no `except:` can catch. `CameraWorker.run()` initialises STA
before touching `VideoCapture`; `stop()` matches it.

---

## 10. Graceful degradation

| Missing | Behaviour |
|---|---|
| Camera | synthetic 4K board with ground truth |
| `.pt` / `ultralytics` | Paths A and A′ greyed out **with the reason** |
| `.pyd` / `.onnx` | Path B greyed out with the reason |
| `matplotlib` | workbook still written; figures skipped with a note |
| `openpyxl` | exports degrade to CSV; the button relabels itself |
| `pynvml` | GPU columns read `n/a`, never `0.0` |
| `psutil` | CPU and RSS columns read `n/a` |
| NVML in the C++ build | the `-1` sentinel is filled from pynvml, so both paths read one instrument |

Nothing in that list stops the application from starting.

---

## 11. Caveat to carry into the write-up

Path A executes `.pt` weights through PyTorch; Path B executes `.onnx` through
ONNX Runtime. **That is two runtimes as well as two languages.** Report Path A′
alongside them, or the headline ratio cannot be attributed to language at all.

`gpu_percent` comes from `nvmlDeviceGetUtilizationRates`, which averages over the
driver's own sampling window (order of a second), not over one frame. Quote it as
a run-level figure; a single row is not a per-frame measurement. `gpu_mem_mb` is
instantaneous and whole-device, so it includes the desktop and anything else
resident.
