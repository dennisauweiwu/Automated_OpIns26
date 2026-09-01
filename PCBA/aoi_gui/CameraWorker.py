# aoi_gui/CameraWorker.py
"""
Frame acquisition. Runs on its own QThread and never touches the GUI directly.

Timer-driven, NOT a blocking while-loop. This matters: `thread.started` is
connected to `run()`, and because the worker lives on that thread the call is
direct, so `run()` executes *before* QThread.exec() starts the event loop. A
blocking loop here would keep the event loop from ever running, which would
(a) silently discard `thread.quit()` and hang shutdown, and (b) stop every
queued cross-thread slot from ever being delivered. So `run()` kicks off a
self-rearming singleShot chain and returns immediately.

Falls back to a synthetic 4K test pattern when no camera is attached, so the
whole pipeline can be exercised on a laptop before the rig is available.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from aoi_gui.config import EngineConfig


def make_test_pattern(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Synthetic PCB-ish frame: green substrate, copper traces, pads.
    Deliberately cheap - it must not dominate timing when used as a source."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 90, 35), np.uint8)          # solder-mask green

    for _ in range(120):                                       # traces
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        length = int(rng.integers(120, 900))
        thick = int(rng.integers(3, 9))
        if rng.random() < 0.5:
            cv2.line(img, (x1, y1), (min(w - 1, x1 + length), y1), (30, 140, 190), thick)
        else:
            cv2.line(img, (x1, y1), (x1, min(h - 1, y1 + length)), (30, 140, 190), thick)

    for _ in range(200):                                       # pads / vias
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (cx, cy), int(rng.integers(6, 16)), (60, 170, 210), -1)

    noise = rng.integers(-8, 9, (h, w, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)

    def __init__(self, cfg: EngineConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._running = False
        self._cap = None
        self._synthetic = False
        self._pattern = None
        self._latest = None
        self._interval_ms = 33

    # ---------------- lifecycle ----------------

    @pyqtSlot()
    def run(self):
        """Start the capture chain and RETURN, letting the event loop start."""
        self._running = True
        self._interval_ms = max(1, int(1000 / max(1, self.cfg.target_fps)))
        self._open_source()
        QTimer.singleShot(0, self._tick)

    @pyqtSlot()
    def _tick(self):
        """Self-rearming via singleShot rather than a persistent QTimer.

        A long-lived QTimer object belongs to this thread, so when the thread
        dies and something else drops the last reference, Qt warns
        'Timers cannot be stopped from another thread' and may crash. A
        singleShot chain leaves no object to destroy: when _running goes False
        the chain simply stops rearming.
        """
        if not self._running:
            return

        started = time.monotonic()

        if self._synthetic:
            frame = self._pattern
        else:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.status_signal.emit("Camera read failed - switching to synthetic feed.")
                self._start_synthetic()
                self._rearm(started)
                return
            if (frame.shape[1] != self.cfg.frame_width or
                    frame.shape[0] != self.cfg.frame_height):
                frame = cv2.resize(frame,
                                   (self.cfg.frame_width, self.cfg.frame_height),
                                   interpolation=cv2.INTER_AREA)

        self._latest = frame
        self.frame_ready.emit(frame)

        self._rearm(started)

    def _rearm(self, started: float):
        if not self._running:
            return
        # Subtract the work already done so the cadence tracks target_fps
        # instead of drifting by the capture duration each cycle.
        elapsed_ms = (time.monotonic() - started) * 1000.0
        QTimer.singleShot(max(0, int(self._interval_ms - elapsed_ms)), self._tick)

    @pyqtSlot()
    def stop(self):
        """Stops the capture chain from rearming and releases the device."""
        self._running = False
        self.release()

    def cleanup(self):
        """Called from the owning thread AFTER the worker thread has joined.
        Touches no Qt objects - only the OpenCV handle - so nothing is destroyed
        from the wrong thread."""
        self._running = False
        self.release()

    def release(self):
        """Safe to call from any thread once the worker thread has finished."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def latest_frame(self):
        return self._latest

    # ---------------- source selection ----------------

    def _open_source(self):
        if getattr(self.cfg, "force_synthetic", False):
            self.status_signal.emit("Camera disabled (--no-camera) - synthetic feed.")
            self._start_synthetic()
            return
        try:
            cap = cv2.VideoCapture(self.cfg.camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._cap = cap
                self._synthetic = False
                self.status_signal.emit(
                    f"Camera {self.cfg.camera_index} open at {aw}x{ah}")
                return
            cap.release()
        except Exception as e:
            self.status_signal.emit(f"Camera error: {e}")

        self._start_synthetic()

    def _start_synthetic(self):
        self.release()
        self._synthetic = True
        if self._pattern is None:
            self._pattern = make_test_pattern(self.cfg.frame_width, self.cfg.frame_height)
        self.status_signal.emit(
            f"No camera - synthetic {self.cfg.frame_width}x{self.cfg.frame_height} feed active")
