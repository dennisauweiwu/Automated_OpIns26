# aoi_gui/AIWorker.py
"""
Inference worker. Owns the active InferenceEngine and runs it off the GUI thread.

Signals out:
    detection_ready(np.ndarray, dict)  -> MainWindow.update_results_slot
    status_signal(str)                 -> status bar
    engine_ready(str, str)             -> (label, backend description)
    benchmark_done(dict)               -> aggregate A/B results

ARCHITECTURE NOTE - why there is no while-loop here.
`thread.started` is connected to `run()`. Because this worker was moved onto
that thread, the connection is DIRECT, so `run()` executes before
QThread.exec() ever starts the event loop. A blocking `while self._running:`
loop in `run()` therefore starves the event loop permanently, and two things
break:
  * queued cross-thread slots (enqueue_frame, set_engine, request_benchmark)
    are never delivered, so no inference ever happens;
  * `thread.quit()` is discarded because no event loop exists to receive it,
    so shutdown hangs and the process has to be killed.
Instead `run()` returns immediately and work is driven by slot invocations
plus QTimer.singleShot, keeping the event loop live between frames.
"""

from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from aoi_gui.config import EngineConfig
from aoi_gui.engines import (
    Detection, FrameResult, InferenceEngine,
    available_engines, create_engine, unavailable_reason,
)

# BGR colour per class index
_PALETTE = [
    (0, 255, 0), (0, 165, 255), (255, 0, 255),
    (0, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 255),
]


class AIWorker(QObject):
    detection_ready = pyqtSignal(np.ndarray, dict)
    status_signal = pyqtSignal(str)
    engine_ready = pyqtSignal(str, str)
    benchmark_done = pyqtSignal(dict)

    def __init__(self, cfg: EngineConfig, default_engine: str = "sim", parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.engine: Optional[InferenceEngine] = None

        self._default_engine = default_engine
        self._running = False
        self._continuous = False
        self._processing = False
        self._pending_frame: Optional[np.ndarray] = None
        self._frame_id = 0

        self._csv_path = Path(cfg.report_dir) / "benchmark_gui.csv"
        self._csv_file = None
        self._csv_writer = None

    # ---------------- lifecycle ----------------

    @pyqtSlot()
    def run(self):
        """Initialise and RETURN so the thread's event loop can start."""
        self._running = True
        self._open_csv()
        if self._default_engine:
            self.set_engine(self._default_engine)

    @pyqtSlot()
    def stop(self):
        """Runs on the worker thread. Releases the engine and closes the CSV."""
        self._running = False
        self._pending_frame = None
        try:
            if self.engine is not None:
                self.engine.close()
        except Exception:
            pass
        self.engine = None
        self._close_csv()

    def cleanup(self):
        """Called from the owning thread AFTER the worker thread has joined, so
        releasing the engine (and its CUDA context) cannot race inference."""
        self._running = False
        self._pending_frame = None
        try:
            if self.engine is not None:
                self.engine.close()
        except Exception:
            pass
        self.engine = None
        self._close_csv()

    # ---------------- engine ----------------

    @pyqtSlot(str)
    def set_engine(self, name: str):
        """Queued from the GUI, so the engine (and any CUDA context) is built on
        THIS thread, never on the GUI thread."""
        self.status_signal.emit(f"Loading '{name}' engine (warming up)...")
        try:
            if self.engine is not None:
                self.engine.close()
            self.engine = create_engine(name, self.cfg)
            self.engine_ready.emit(self.engine.label, self.engine.backend_desc)
            self.status_signal.emit(f"{self.engine.label} ready - {self.engine.backend_desc}")
        except Exception as e:
            self.engine = None
            self.engine_ready.emit("Engine failed", str(e))
            self.status_signal.emit(f"Engine load failed: {e}")
            traceback.print_exc()

    # ---------------- intake ----------------

    @pyqtSlot(np.ndarray)
    def enqueue_frame(self, frame: np.ndarray):
        """From CameraWorker. Keeps only the newest frame, so a slow backend
        shows up as low FPS rather than unbounded lag."""
        if not (self._running and self._continuous):
            return
        self._pending_frame = frame
        self._schedule()

    @pyqtSlot(np.ndarray)
    def analyze_once(self, frame: np.ndarray):
        """Single-shot capture-and-analyse, bypassing the continuous gate."""
        if not self._running:
            return
        self._pending_frame = frame
        self._schedule()

    @pyqtSlot(bool)
    def set_continuous(self, enabled: bool):
        self._continuous = enabled
        self.status_signal.emit(
            "Continuous inspection ON" if enabled else "Continuous inspection OFF")

    def _schedule(self):
        # singleShot(0) returns to the event loop first, so queued slots that
        # arrived meanwhile (stop, set_engine) still get a chance to run.
        if not self._processing:
            QTimer.singleShot(0, self._process)

    @pyqtSlot()
    def _process(self):
        if not self._running or self._processing:
            return
        frame = self._pending_frame
        self._pending_frame = None
        if frame is None or self.engine is None:
            return

        self._processing = True
        try:
            result = self.engine.run_frame(frame, self._frame_id)
            self._frame_id += 1
            self._write_csv(result)

            # Annotation sits OUTSIDE the timed region and is identical for
            # every backend, so drawing cost never lands in one engine's column.
            annotated = self._annotate(frame, result.detections)
            stats = result.to_stats_dict(self.cfg.class_names)
            stats["backend_desc"] = self.engine.backend_desc
            self.detection_ready.emit(annotated, stats)
        except Exception as e:
            self.status_signal.emit(f"Inference error: {e}")
            traceback.print_exc()
        finally:
            self._processing = False

        if self._pending_frame is not None and self._running:
            self._schedule()

    # ---------------- A/B sweep ----------------

    @pyqtSlot(np.ndarray, int)
    def request_benchmark(self, frame: np.ndarray, frames: int = 30):
        """The sweep carries its own frame; sharing the live feed would race the
        normal inference path."""
        try:
            self._run_ab_benchmark(np.ascontiguousarray(frame), frames)
        except Exception as e:
            self.status_signal.emit(f"Benchmark failed: {e}")
            traceback.print_exc()
            self.benchmark_done.emit({"error": str(e)})

    def _run_ab_benchmark(self, frame: np.ndarray, frames: int):
        if frame is None or frame.size == 0:
            self.status_signal.emit("Benchmark aborted: no frame available.")
            self.benchmark_done.emit({"error": "no frame available"})
            return

        avail = available_engines(self.cfg)
        summary = {}
        previous = self.engine.name if self.engine else self._default_engine

        for name in ("python", "cpp"):
            if not avail.get(name):
                summary[name] = {"error": unavailable_reason(name, self.cfg)}
                continue

            self.status_signal.emit(f"Benchmarking {name} over {frames} frames...")
            try:
                eng = create_engine(name, self.cfg)
            except Exception as e:
                summary[name] = {"error": str(e)}
                continue

            runs: List[FrameResult] = []
            for i in range(frames):
                # Plain attribute read - visible even though the event loop is
                # busy inside this sweep, so closing the app aborts promptly.
                if not self._running:
                    break
                runs.append(eng.run_frame(frame, i))
                if i % 5 == 0:
                    self.status_signal.emit(f"{name}: frame {i + 1}/{frames}")

            if not runs:
                eng.close()
                continue

            def avg(key, _runs=runs):
                vals = [r.metrics.get(key, 0.0) for r in _runs]
                return sum(vals) / len(vals)

            lat = sorted(r.metrics.get("total_ms", 0.0) for r in runs)
            summary[name] = {
                "label": eng.label,
                "backend_desc": eng.backend_desc,
                "frames": len(runs),
                "tiles": runs[0].metrics.get("tiles", 0),
                "mean_ms": avg("total_ms"),
                "p50_ms": lat[len(lat) // 2],
                "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
                "fps": avg("fps"),
                "preprocess_ms": avg("preprocess_ms"),
                "inference_ms": avg("inference_ms"),
                "postprocess_ms": avg("postprocess_ms"),
                "cpu_percent": avg("cpu_percent"),
                "gpu_percent": avg("gpu_percent"),
                "rss_mb": avg("rss_mb"),
                "peak_rss_mb": max(r.metrics.get("peak_rss_mb", 0.0) for r in runs),
                "detections": avg("detections"),
            }
            for r in runs:
                self._write_csv(r)
            eng.close()

        py, cpp = summary.get("python", {}), summary.get("cpp", {})
        if "error" not in py and "error" not in cpp and py and cpp and cpp.get("mean_ms"):
            summary["speedup"] = py["mean_ms"] / cpp["mean_ms"]
            summary["memory_delta_mb"] = py["rss_mb"] - cpp["rss_mb"]

        summary["csv"] = str(self._csv_path)
        self.status_signal.emit("Benchmark complete.")
        self.benchmark_done.emit(summary)

        if self._running:
            self.set_engine(previous)

    # ---------------- helpers ----------------

    def _annotate(self, frame: np.ndarray, dets: List[Detection]) -> np.ndarray:
        out = frame.copy()
        names = self.cfg.class_names
        for d in dets:
            colour = _PALETTE[d.class_id % len(_PALETTE)]
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
            label = names[d.class_id] if 0 <= d.class_id < len(names) else f"cls{d.class_id}"
            cv2.putText(out, f"{label} {d.confidence:.2f}", (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        return out

    _FIELDS = ["timestamp", "engine", "frame_id", "tiles", "slicing_ms",
               "preprocess_ms", "inference_ms", "postprocess_ms", "merge_ms",
               "total_ms", "fps", "cpu_percent", "gpu_percent", "gpu_mem_mb",
               "rss_mb", "peak_rss_mb", "detections"]

    def _open_csv(self):
        try:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._FIELDS)
            self._csv_writer.writeheader()
        except Exception:
            self._csv_writer = None

    def _write_csv(self, result: FrameResult):
        if not self._csv_writer:
            return
        row = {"timestamp": round(time.time(), 3)}
        row.update({k: v for k, v in result.metrics.items() if k in self._FIELDS})
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception:
            pass

    def _close_csv(self):
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None
