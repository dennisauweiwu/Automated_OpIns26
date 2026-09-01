# Op-Ins 2601 — AOI Comparative Benchmark

PyQt5 inspection GUI for 4K PCB automated optical inspection, built to compare a
**Python inference path** against a **C++ inference path** on identical work.

The GUI runs immediately with no model, no GPU and no camera — a simulation
engine and a synthetic 4K feed stand in — so interface work is never blocked on
the rig.

```
python main.py
```

---

## 1. Quick start

```bash
pip install -r requirements.txt
python tools/selfcheck.py      # shows exactly what is wired up and what is missing
python main.py                 # launches with the simulation engine
```

Useful flags:

| Command | Effect |
|---|---|
| `python main.py` | Simulation engine, synthetic 4K feed |
| `python main.py --engine python` | Test Path A (needs `ultralytics` + `.pt`) |
| `python main.py --engine cpp` | Test Path B (needs the built `.pyd` + `.onnx`) |
| `python main.py --theme light` | Light theme |
| `python main.py --camera 1` | Camera index override |
| `python main.py --no-cuda` | Force CPU on both paths |

Backends that cannot load are greyed out in the dropdown **with the reason**
rather than crashing when selected.

---

## 2. Layout

```
opins_aoi/
├── main.py                       entry point, CLI, stylesheet loading
├── requirements.txt
├── aoi_gui/
│   ├── config.py                 EngineConfig — single source of truth
│   ├── engines.py                ★ backend abstraction (sim / python / cpp)
│   ├── AIWorker.py               inference thread, A/B sweep, CSV logging
│   ├── CameraWorker.py           capture thread + synthetic fallback
│   ├── Controller.py             owns threads and hardware lifecycle
│   ├── Hardware.py               signal tower (serial) + MQTT alerts
│   ├── Components.py             frameless TitleBar
│   ├── Report.py                 .xlsx / .csv report export
│   ├── MainWindow.py             GUI, benchmark panel, result slots
│   └── yolo_backend_cpp.pyd      ← appears here after the CMake build
├── cpp_backend/
│   ├── yolo_backend_cpp.cpp      C++17 engine (also builds a standalone exe)
│   └── CMakeLists.txt
├── ui/
│   ├── style.qss                 dark theme
│   └── style_light.qss           light theme
├── tools/
│   ├── selfcheck.py              environment diagnostic
│   ├── run_benchmark.py          headless A/B sweep → performance matrix
│   └── export_onnx.py            .pt → .onnx with a hash manifest
├── datasets/weights/             put yolov8_pcb_4k.pt / .onnx here
└── reports/                      CSV output lands here
```

`engines.py` is the integration point. Every backend exposes the same call:

```python
result = engine.run_frame(frame_bgr, frame_id)   # -> FrameResult
```

so `AIWorker` swaps backends at runtime and nothing above it changes.

---

## 3. Data flow

```
CameraWorker (thread 1)          AIWorker (thread 2)              MainWindow (GUI thread)
─────────────────────            ───────────────────              ───────────────────────
capture / synthesise 4K
      │ frame_ready ────────────► enqueue_frame (drops stale)
                                        │
                                  engine.run_frame()
                                    ├─ SAHI slice into tiles
                                    ├─ letterbox + inference
                                    └─ cross-tile NMS
                                        │
                                  annotate (outside the timer)
                                        │ detection_ready ───────► update_results_slot
                                                                     ├─ PASS/FAIL indicator
                                                                     ├─ defect + latency labels
                                                                     ├─ FPS / CPU / GPU / RAM
                                                                     └─ letterboxed video pane
```

For the C++ engine, `run_frame` hands the numpy buffer to pybind11, which wraps
it as a `cv::Mat` **without copying** and releases the GIL for the duration, so
the GUI keeps repainting during inference.

---

## 4. Building Test Path B

```bash
cd cpp_backend
cmake -B build -S . -A x64 ^
      -DOpenCV_DIR="C:/opencv/build" ^
      -DPython_EXECUTABLE="C:/path/to/venv/Scripts/python.exe" ^
      -DUSE_NVML=ON
cmake --build build --config Release

python -c "import yolo_backend_cpp; print('ok')"
```

The `.pyd` **must** be built against the same interpreter that runs the GUI, or
the import fails with an ABI error. CMake drops it into `aoi_gui/` automatically.

Two targets are produced from the one source file:

- `yolo_backend_cpp.pyd` — imported by the GUI
- `aoi_cpp_bench.exe` — standalone benchmark, no Python in the loop

Export the ONNX from the *same* training run as the `.pt`:

```bash
python tools/export_onnx.py        # writes a .manifest.json with both hashes
```

Use `--opset 12`; OpenCV DNN frequently fails to parse opset 17+.

---

## 5. Running the comparison

Press **Run A/B Benchmark** in the sidebar. The same captured frame is pushed
through each available backend 30 times, and you get mean / p50 / p95 latency,
FPS, a pre/infer/post breakdown, CPU, GPU and RSS.

Per-frame data is appended to `reports/benchmark_gui.csv`.

For the numbers that go in the write-up, prefer the **headless runner** — Qt
repaints compete with inference for CPU, which inflates latency by a few percent:

```bash
python tools/run_benchmark.py --frames 50                # all available backends
python tools/run_benchmark.py --engines python cpp --image board.png
python tools/run_benchmark.py --no-cuda                  # CPU baseline
```

It prints the matrix to the console and writes `reports/benchmark_<stamp>.xlsx`
with three sheets: session summary, the comparison matrix, and per-frame raw
data. Derived cells (yield, ratios, speedup) are **live Excel formulas**, so the
workbook still recalculates if a count is edited during review.

### What keeps the comparison honest

These are deliberate and easy to break by accident:

| Guard | Why it matters |
|---|---|
| Identical tile origins (`_axis_origins` ↔ `axis_origins`) | The paths must process the same pixels. Change one, change both. |
| Identical letterboxing | A plain 640×640 resize squashes a 16:9 tile and changes which defects are found. |
| One `EngineConfig` for both | Different thresholds silently change detection counts. |
| Warmup discarded on both | The first CUDA pass pays for kernel autotuning and cuDNN workspace allocation. |
| `torch.cuda.synchronize()` per tile | CUDA is async — without it you time the kernel *launch*, not the work. |
| Annotation outside the timer | Otherwise you compare two drawing routines, not two engines. |
| CPU% normalised by core count | 100% means all cores busy on both paths, not 1200% on one. |
| Explicit CUDA probe in C++ | An OpenCV build without CUDA silently falls back to CPU and corrupts the result. |

### A caveat for the write-up

The Python path runs `.pt` weights through PyTorch; the C++ path runs `.onnx`
through OpenCV DNN. That is **two runtimes, not just two languages** — some of
the gap is ONNX-vs-PyTorch kernel differences rather than interpreter overhead.
If that distinction gets challenged, run the Python path through
`onnxruntime-gpu` on the same `.onnx` to isolate the language variable.

---

## 6. Known behaviour: tile count at 4K

With 1920×1080 tiles on a 3840×2160 frame, **any non-zero overlap forces a 3×3
grid** and the real overlap lands at 50%, not the 20% requested. 3840 is exactly
2×1920, so two columns give 0% overlap and three is the next possible value.

The cost is direct: **9 inferences per frame instead of 4.** `tools/selfcheck.py`
prints the actual grid and flags this.

If 9 tiles/frame is too slow, either accept 2×2 with `overlap_ratio = 0.0`
(defects straddling the seam may be missed), or use a tile size that divides the
frame less evenly — e.g. 1536×864 gives 3×3 at a true 25% overlap with 36% fewer
pixels per tile.

Whatever you choose, the setting lives in `aoi_gui/config.py` and both engines
read it, so the paths stay aligned.

---

## 7. Threading model (do not reintroduce a while-loop)

Both workers are `QObject`s moved onto a `QThread`, with `thread.started`
connected to `run()`. Because the worker lives on that thread, that call is
**direct** — `run()` executes *before* `QThread.exec()` starts the event loop.

So `run()` must arm a timer (camera) or return immediately (AI) and let the
event loop take over. A blocking `while self._running:` loop there breaks two
things at once:

- **queued cross-thread slots are never delivered** — `enqueue_frame`,
  `set_engine`, `request_benchmark` all silently no-op, so no inference runs;
- **`thread.quit()` is discarded**, because no event loop exists to receive it.
  `run()` then returns, `exec()` starts a loop nobody will ever quit, `wait()`
  times out, `terminate()` fires, and the process has to be killed from Task
  Manager.

Shutdown order in `Controller.stop()` is deliberate: plain flag writes first
(visible without an event loop, so a long benchmark sweep aborts), then
`quit()`, then a bounded `wait()`, then worker cleanup once the threads have
joined, and hardware release in a `finally` so a stuck thread can never leave a
non-daemon MQTT network thread holding the interpreter open.

---

## 8. Graceful degradation

| Missing | Behaviour |
|---|---|
| Camera | Synthetic 4K PCB test pattern |
| `.pt` / `ultralytics` | Python backend greyed out with the reason |
| `.pyd` / `.onnx` | C++ backend greyed out with the reason |
| `openpyxl` | Report export degrades to `.csv`, button relabels itself |
| `pynvml` | GPU columns show `NVML off` |
| `psutil` | CPU/RAM columns show `n/a` |
| `pyserial` | Tower disabled, status message, inspection continues |
| `paho-mqtt` | Alerts disabled, status message, inspection continues |

Nothing in that list stops the GUI from starting.
