# aoi_gui/Controller.py
"""
Owns the worker threads and hardware interfaces. MainWindow talks only to this
object, never to a QThread directly.

Both workers are plain QObjects moved onto QThreads (rather than QThread
subclasses) so their slots execute on the worker thread. Neither worker blocks
its thread's event loop - see the note in AIWorker for why that matters for
both frame delivery and clean shutdown.
"""

from __future__ import annotations

import os
import time

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from aoi_gui.AIWorker import AIWorker
from aoi_gui.CameraWorker import CameraWorker
from aoi_gui.Hardware import MQTTPublisher, TowerController
from aoi_gui.config import EngineConfig


class Controller(QObject):
    status_signal = pyqtSignal(str)

    def __init__(self, cfg: EngineConfig | None = None,
                 default_engine: str = "sim", parent=None):
        super().__init__(parent)
        self.cfg = cfg or EngineConfig()
        self._stopped = False

        # --- Threads ---------------------------------------------------------
        self.cam_thread = QThread()
        self.cam_thread.setObjectName("CameraThread")
        self.ai_thread = QThread()
        self.ai_thread.setObjectName("AIThread")

        # --- Workers ---------------------------------------------------------
        self.cam_worker = CameraWorker(self.cfg)
        self.ai_worker = AIWorker(self.cfg, default_engine=default_engine)

        self.cam_worker.moveToThread(self.cam_thread)
        self.ai_worker.moveToThread(self.ai_thread)

        self.cam_thread.started.connect(self.cam_worker.run)
        self.ai_thread.started.connect(self.ai_worker.run)

        # Camera -> AI. Queued automatically because they live on different
        # threads, so the camera never blocks waiting for inference.
        self.cam_worker.frame_ready.connect(self.ai_worker.enqueue_frame)

        # --- Hardware --------------------------------------------------------
        self.tower_controller = TowerController(self.cfg)
        self.mqtt_publisher = MQTTPublisher(self.cfg)

    def start(self):
        self.tower_controller.connect()
        self.mqtt_publisher.connect()
        self.cam_thread.start()
        self.ai_thread.start()
        self.status_signal.emit("Controller started.")

    def latest_frame(self):
        return self.cam_worker.latest_frame()

    def stop(self):
        """Idempotent. Ordered so that a hang in any one step still releases the
        hardware handles - a live paho network thread is non-daemon and will
        keep the interpreter alive after the Qt loop exits."""
        if self._stopped:
            return
        self._stopped = True

        verbose = os.environ.get("AOI_DEBUG_SHUTDOWN") == "1"
        t0 = time.monotonic()

        def step(msg):
            if verbose:
                print(f"[shutdown +{time.monotonic() - t0:6.2f}s] {msg}", flush=True)

        step("begin")

        try:
            # 1. Plain attribute writes. Visible immediately without needing the
            #    worker's event loop, so a long benchmark sweep aborts promptly
            #    and the capture timer stops firing on its next tick.
            self.ai_worker._running = False
            self.cam_worker._running = False
            step("flags cleared")

            # 2. Exit both event loops. This works only because neither worker
            #    blocks its loop - see the note in AIWorker.
            for thread in (self.ai_thread, self.cam_thread):
                thread.quit()
            step("quit() posted to both event loops")

            # 3. Join, bounded. Deliberately NOT BlockingQueuedConnection: if a
            #    worker were mid-CUDA-load that would freeze the GUI for the
            #    whole load with no timeout available.
            for thread in (self.ai_thread, self.cam_thread):
                name = thread.objectName() or "thread"
                step(f"joining {name}...")
                if not thread.wait(5000):
                    step(f"{name} DID NOT EXIT in 5s - terminating "
                         "(this is the hang; see the stack dump above)")
                    thread.terminate()
                    thread.wait(1000)
                step(f"{name} joined")

            # 4. Threads are gone, so touching worker state is now safe from
            #    this thread.
            self.cam_worker.cleanup()
            self.ai_worker.cleanup()
            step("worker cleanup done")
        finally:
            # Always runs, even if a thread had to be terminated above. A live
            # paho network thread is non-daemon and would keep the interpreter
            # alive after the Qt loop exits - this is what forces a Task
            # Manager kill if it is skipped.
            try:
                self.tower_controller.close()
            except Exception:
                pass
            try:
                self.mqtt_publisher.close()
            except Exception:
                pass
            step("hardware released - stop() complete")
