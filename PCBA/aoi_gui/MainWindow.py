# aoi_gui/MainWindow.py

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import (
    QEasingCurve, QPropertyAnimation, Qt, pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QComboBox, QFormLayout, QGraphicsOpacityEffect,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMenuBar, QMessageBox,
    QPushButton, QSizePolicy, QSlider, QStatusBar, QVBoxLayout, QWidget,
)

from aoi_gui.Components import TitleBar
from aoi_gui.config import EngineConfig
from aoi_gui.Controller import Controller
from aoi_gui.engines import available_engines, unavailable_reason
from aoi_gui.Report import export_report, xlsx_available


class MainWindow(QMainWindow):
    # GUI -> AIWorker (queued across threads)
    engine_change_requested = pyqtSignal(str)
    benchmark_requested = pyqtSignal(np.ndarray, int)
    continuous_toggled = pyqtSignal(bool)
    analyze_once_requested = pyqtSignal(np.ndarray)

    def __init__(self, cfg: EngineConfig | None = None, default_engine: str = "sim"):
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self._default_engine = default_engine
        self.current_theme = "dark"
        self._last_stats: dict = {}
        self._last_summary: dict = {}
        self._history: list[dict] = []
        self._boards_total = 0
        self._boards_passed = 0
        self._continuous = False

        self.init_ui()
        self.init_threads()

    # =========================================================================
    # UI construction
    # =========================================================================

    def init_ui(self):
        self.setWindowTitle("Op-Ins 2601 - AOI Comparative Benchmark")
        self.setGeometry(100, 100, 1400, 900)

        try:
            self._load_stylesheet()
        except Exception:
            pass

        self.setWindowFlag(Qt.FramelessWindowHint)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout(main_widget)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self.title_bar = TitleBar(self)
        vbox.addWidget(self.title_bar)

        menubar = QMenuBar(self)
        menubar.setObjectName("MenuBar")
        file_menu = menubar.addMenu("&Menu")

        theme_action = QAction("Toggle &Theme", self)
        theme_action.setShortcut("Ctrl+T")
        theme_action.triggered.connect(self.toggle_theme)
        file_menu.addAction(theme_action)

        export_action = QAction("&Export Report", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_report)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        vbox.addWidget(menubar)

        main_layout = QHBoxLayout()
        vbox.addLayout(main_layout)

        # --- Video display ---------------------------------------------------
        self.video_label = QLabel("Waiting for Camera Feed...")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(400, 300)
        # No setScaledContents(True): frames are letterboxed explicitly so the
        # 16:9 board is never stretched in the operator's view.
        main_layout.addWidget(self.video_label, 3)

        # --- Sidebar ---------------------------------------------------------
        sidebar_widget = QWidget()
        sidebar_widget.setMinimumWidth(340)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        main_layout.addWidget(sidebar_widget, 1)

        sidebar_layout.addWidget(self._create_stats_group())
        sidebar_layout.addWidget(self._create_benchmark_group())
        sidebar_layout.addWidget(self._create_control_group())
        sidebar_layout.addWidget(self._create_action_buttons())
        sidebar_layout.addStretch(1)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Initialising...")

    def _create_stats_group(self):
        stats_group = QGroupBox("Inspection Report")
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.status_indicator = QLabel("LOADING...")
        self.status_indicator.setObjectName("StatusIndicator")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setMinimumHeight(50)
        self.status_indicator.setProperty("status", "loading")
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)
        self._init_status_animation()
        layout.addWidget(self.status_indicator)

        form = QFormLayout()
        form.setContentsMargins(0, 6, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.defects_label = QLabel("0")
        self.type_label = QLabel("N/A")
        self.yield_label = QLabel("100.0%")
        self.latency_label = QLabel("0 ms")

        form.addRow("Total Defects:", self.defects_label)
        form.addRow("Top Defect Type:", self.type_label)
        form.addRow("Inspection Yield:", self.yield_label)
        form.addRow("Inference Latency:", self.latency_label)

        layout.addLayout(form)
        stats_group.setLayout(layout)
        return stats_group

    def _create_benchmark_group(self):
        """Backend selector plus the live performance matrix."""
        group = QGroupBox("Benchmark Engine")
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.engine_combo = QComboBox()
        self.engine_combo.setObjectName("engineCombo")
        avail = available_engines(self.cfg)
        for key, text in (("sim", "Simulation (no model)"),
                          ("python", "Test Path A - Python"),
                          ("cpp", "Test Path B - C++")):
            self.engine_combo.addItem(text, key)

        # Grey out what cannot load, with the reason, instead of crashing later
        for row in range(self.engine_combo.count()):
            key = self.engine_combo.itemData(row)
            if not avail.get(key, False):
                self.engine_combo.model().item(row).setEnabled(False)
                self.engine_combo.setItemText(
                    row, f"{self.engine_combo.itemText(row)}  ({unavailable_reason(key, self.cfg)})")

        idx = self.engine_combo.findData(self._default_engine)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)

        selector = QFormLayout()
        selector.setHorizontalSpacing(12)
        selector.setVerticalSpacing(8)
        selector.addRow("Active Backend:", self.engine_combo)
        layout.addLayout(selector)

        self.backend_desc_label = QLabel("not loaded")
        self.backend_desc_label.setObjectName("BackendDesc")
        self.backend_desc_label.setWordWrap(True)
        layout.addWidget(self.backend_desc_label)

        metrics = QFormLayout()
        metrics.setContentsMargins(0, 4, 0, 0)
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(6)

        self.fps_label = QLabel("0.0")
        self.tiles_label = QLabel("0")
        self.stage_label = QLabel("- / - / -")
        self.cpu_label = QLabel("0.0 %")
        self.gpu_label = QLabel("0.0 %")
        self.ram_label = QLabel("0 MB")
        for w in (self.fps_label, self.tiles_label, self.stage_label,
                  self.cpu_label, self.gpu_label, self.ram_label):
            w.setObjectName("MetricValue")

        metrics.addRow("Effective FPS:", self.fps_label)
        metrics.addRow("SAHI Tiles/Frame:", self.tiles_label)
        metrics.addRow("Pre/Infer/Post:", self.stage_label)
        metrics.addRow("CPU Load:", self.cpu_label)
        metrics.addRow("GPU Load:", self.gpu_label)
        metrics.addRow("Memory (RSS):", self.ram_label)
        layout.addLayout(metrics)

        self.benchmark_button = QPushButton("Run A/B Benchmark (30 frames)")
        self.benchmark_button.setObjectName("benchmarkButton")
        self.benchmark_button.setMinimumHeight(40)
        self.benchmark_button.clicked.connect(self._on_run_benchmark)
        layout.addWidget(self.benchmark_button)

        group.setLayout(layout)
        return group

    def _create_control_group(self):
        control_group = QGroupBox("Hardware/Light Controls")
        layout = QFormLayout()
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(128)
        layout.addRow("Brightness:", self.brightness_slider)

        control_group.setLayout(layout)
        return control_group

    def _create_action_buttons(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)

        self.continuous_button = QPushButton("Start Continuous Inspection")
        self.continuous_button.setObjectName("continuousButton")
        self.continuous_button.setCheckable(True)
        self.continuous_button.setMinimumHeight(44)
        self.continuous_button.toggled.connect(self._on_continuous_toggled)
        v.addWidget(self.continuous_button)

        self.capture_button = QPushButton("Capture and Analyze")
        self.capture_button.setObjectName("captureButton")
        self.capture_button.setMinimumHeight(44)
        self.capture_button.clicked.connect(self._on_capture_and_analyze)
        v.addWidget(self.capture_button)

        ext = ".xlsx" if xlsx_available() else ".csv"
        self.report_button = QPushButton(f"Export Report ({ext})")
        self.report_button.setObjectName("reportButton")
        self.report_button.setMinimumHeight(44)
        self.report_button.clicked.connect(self._on_export_report)
        v.addWidget(self.report_button)

        return w

    # =========================================================================
    # Threads and wiring
    # =========================================================================

    def init_threads(self):
        self.controller = Controller(self.cfg, default_engine=self._default_engine)

        for src in (self.controller.cam_worker.status_signal,
                    self.controller.ai_worker.status_signal,
                    self.controller.tower_controller.status_signal,
                    self.controller.mqtt_publisher.status_signal,
                    self.controller.status_signal):
            try:
                src.connect(self._show_status)
            except Exception:
                pass

        self.controller.ai_worker.detection_ready.connect(self.update_results_slot)
        self.controller.ai_worker.engine_ready.connect(self.on_engine_ready)
        self.controller.ai_worker.benchmark_done.connect(self.on_benchmark_done)

        # GUI -> worker. Cross-thread signals are queued, so set_engine runs on
        # the AI thread and the CUDA context is never built on the GUI thread.
        self.engine_change_requested.connect(self.controller.ai_worker.set_engine)
        self.benchmark_requested.connect(self.controller.ai_worker.request_benchmark)
        self.continuous_toggled.connect(self.controller.ai_worker.set_continuous)
        self.analyze_once_requested.connect(self.controller.ai_worker.analyze_once)

        self.controller.start()

    @pyqtSlot(str)
    def _show_status(self, msg: str):
        self.statusBar().showMessage(msg, 6000)

    # =========================================================================
    # Actions
    # =========================================================================

    def _on_engine_changed(self, index: int):
        key = self.engine_combo.itemData(index)
        if not key:
            return
        self.backend_desc_label.setText("loading engine...")
        self.engine_change_requested.emit(key)

    def _on_continuous_toggled(self, checked: bool):
        self._continuous = checked
        self.continuous_button.setText(
            "Stop Continuous Inspection" if checked else "Start Continuous Inspection")
        self.continuous_toggled.emit(checked)

    def _on_capture_and_analyze(self):
        frame = self.controller.latest_frame()
        if frame is None:
            self.statusBar().showMessage("No frame available yet.", 3000)
            return
        self.statusBar().showMessage("Analysing captured frame...", 3000)
        self.analyze_once_requested.emit(frame)

    def _on_run_benchmark(self):
        frame = self.controller.latest_frame()
        if frame is None:
            self.statusBar().showMessage("No frame available for benchmarking.", 3000)
            return
        self.benchmark_button.setEnabled(False)
        self.engine_combo.setEnabled(False)
        self.statusBar().showMessage("A/B benchmark running - please wait.", 0)
        # The frame travels with the request; seeding the shared queue instead
        # would let the normal inference loop consume it first.
        self.benchmark_requested.emit(frame, 30)

    def _on_export_report(self):
        if not self._history:
            self.statusBar().showMessage("Nothing to export yet - run an inspection first.", 4000)
            return
        try:
            path = export_report(self.cfg.report_dir, self._last_stats,
                                 self._history, self._last_summary)
            self.statusBar().showMessage(f"Report written to {path}", 8000)
        except Exception as e:
            self.statusBar().showMessage(f"Export failed: {e}", 8000)

    # =========================================================================
    # Slots from the worker
    # =========================================================================

    @pyqtSlot(np.ndarray, dict)
    def update_results_slot(self, annotated_img: np.ndarray, stats_dict: dict):
        self._last_stats = stats_dict

        total_defects = stats_dict.get("total_defects", 0)
        is_pass = total_defects == 0

        # Rolling session tally - this is what drives Inspection Yield.
        self._boards_total += 1
        if is_pass:
            self._boards_passed += 1
        record = {k: stats_dict.get(k) for k in (
            "engine", "total_defects", "top_defect_type", "latency_ms", "fps",
            "tiles", "preprocess_ms", "inference_ms", "postprocess_ms",
            "cpu_percent", "gpu_percent", "rss_mb")}
        record["timestamp"] = round(time.time(), 3)
        self._history.append(record)
        # Cap the in-memory history so a long shift cannot grow without bound;
        # per-frame data is already streamed to CSV by the worker.
        if len(self._history) > 5000:
            del self._history[:1000]

        self.defects_label.setText(str(total_defects))
        self.type_label.setText(stats_dict.get("top_defect_type", "N/A"))
        self.latency_label.setText(f"{stats_dict.get('latency_ms', 0):.2f} ms")

        yield_pct = (self._boards_passed / self._boards_total * 100.0
                     if self._boards_total else 100.0)
        self.yield_label.setText(
            f"{yield_pct:.1f}%  ({self._boards_passed}/{self._boards_total})")

        # Benchmark readouts
        self.fps_label.setText(f"{stats_dict.get('fps', 0.0):.1f}")
        self.tiles_label.setText(str(stats_dict.get("tiles", 0)))
        self.stage_label.setText(
            f"{stats_dict.get('preprocess_ms', 0.0):.1f} / "
            f"{stats_dict.get('inference_ms', 0.0):.1f} / "
            f"{stats_dict.get('postprocess_ms', 0.0):.1f} ms")
        cpu = stats_dict.get("cpu_percent", -1.0)
        gpu = stats_dict.get("gpu_percent", -1.0)
        self.cpu_label.setText(f"{cpu:.1f} %" if cpu >= 0 else "n/a")
        self.gpu_label.setText(f"{gpu:.1f} %" if gpu >= 0 else "NVML off")
        self.ram_label.setText(f"{stats_dict.get('rss_mb', 0.0):.0f} MB")

        self._animate_status_change("PASS" if is_pass else "FAIL",
                                    "pass" if is_pass else "fail")

        try:
            self.controller.tower_controller.set_status(is_pass)
        except Exception:
            pass
        if not is_pass:
            try:
                self.controller.mqtt_publisher.publish_alert(stats_dict)
            except Exception:
                pass

        target = self.video_label.size()
        q_img = self._convert_cv_to_qimage(annotated_img, target.width(), target.height())
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    @pyqtSlot(str, str)
    def on_engine_ready(self, label: str, backend_desc: str):
        self.backend_desc_label.setText(backend_desc)
        self.statusBar().showMessage(f"{label} - {backend_desc}", 6000)

    @pyqtSlot(dict)
    def on_benchmark_done(self, summary: dict):
        self.benchmark_button.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self._last_summary = summary

        # Per-engine failures are nested under their engine key, so a top-level
        # 'error' means the sweep itself could not start.
        if "error" in summary:
            QMessageBox.warning(self, "Benchmark", f"Benchmark failed: {summary['error']}")
            return

        lines = []
        for key in ("python", "cpp"):
            s = summary.get(key, {})
            if not s:
                continue
            if "error" in s:
                lines.append(f"<b>{key}</b>: unavailable ({s['error']})")
                continue
            lines.append(
                f"<b>{s['label']}</b><br>"
                f"&nbsp;&nbsp;{s['backend_desc']}<br>"
                f"&nbsp;&nbsp;mean {s['mean_ms']:.2f} ms "
                f"(p50 {s['p50_ms']:.2f} / p95 {s['p95_ms']:.2f})<br>"
                f"&nbsp;&nbsp;{s['fps']:.2f} FPS over {s['tiles']} tiles<br>"
                f"&nbsp;&nbsp;pre {s['preprocess_ms']:.2f} / infer {s['inference_ms']:.2f} "
                f"/ post {s['postprocess_ms']:.2f} ms<br>"
                f"&nbsp;&nbsp;CPU {s['cpu_percent']:.1f}% · GPU {s['gpu_percent']:.1f}% "
                f"· RSS {s['rss_mb']:.0f} MB")

        if "speedup" in summary:
            lines.append(f"<b>C++ speedup: {summary['speedup']:.2f}x</b><br>"
                         f"Memory delta: {summary['memory_delta_mb']:.0f} MB")
        if "csv" in summary:
            lines.append(f"<br>Per-frame data: {summary['csv']}")

        box = QMessageBox(self)
        box.setWindowTitle("A/B Benchmark Results")
        box.setTextFormat(Qt.RichText)
        box.setText("<br>".join(lines) if lines else "No backends available to benchmark.")
        box.exec_()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _convert_cv_to_qimage(self, cv_img: np.ndarray,
                              target_w: int = None, target_h: int = None) -> QImage:
        """BGR ndarray -> QImage, letterboxed into the exact target size.

        .copy() is essential: QImage does not take ownership of the numpy buffer,
        so without it the pixmap can reference freed memory and the video pane
        renders garbage or crashes.
        """
        if target_w is None or target_h is None:
            if cv_img is None:
                return QImage(640, 480, QImage.Format_RGB888)
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

        target_w = max(1, target_w)
        target_h = max(1, target_h)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if cv_img is not None:
            src_h, src_w = cv_img.shape[:2]
            scale = min(target_w / src_w, target_h / src_h)
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))
            resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            canvas[y:y + new_h, x:x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, target_w, target_h, 3 * target_w,
                      QImage.Format_RGB888).copy()

    def _init_status_animation(self):
        self.status_opacity = QGraphicsOpacityEffect(self.status_indicator)
        self.status_indicator.setGraphicsEffect(self.status_opacity)

        self._fade_out = QPropertyAnimation(self.status_opacity, b"opacity")
        self._fade_out.setDuration(300)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InOutQuad)

        self._fade_in = QPropertyAnimation(self.status_opacity, b"opacity")
        self._fade_in.setDuration(300)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.InOutQuad)

        self._fade_out.finished.connect(self._on_fade_out_finished)
        self._pending_status = None

    def _animate_status_change(self, text: str, prop: str):
        if (self.status_indicator.text() == text
                and self.status_indicator.property("status") == prop):
            return
        # A single persistent connection with a pending payload, rather than
        # connecting a fresh closure each call - otherwise rapid PASS/FAIL
        # alternation stacks duplicate handlers on the animation.
        self._pending_status = (text, prop)
        self._fade_out.stop()
        self._fade_in.stop()
        self._fade_out.start()

    def _on_fade_out_finished(self):
        if not self._pending_status:
            return
        text, prop = self._pending_status
        self._pending_status = None
        self.status_indicator.setText(text)
        self.status_indicator.setProperty("status", prop)
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)
        self._fade_in.start()

    def _load_stylesheet(self):
        app = QApplication.instance()
        if app is None or app.styleSheet().strip():
            return
        qss = Path(__file__).resolve().parents[1] / "ui" / "style.qss"
        if qss.exists():
            app.setStyleSheet(qss.read_text(encoding="utf-8"))

    def toggle_theme(self):
        app = QApplication.instance()
        if app is None:
            return
        base = Path(__file__).resolve().parents[1] / "ui"
        target = base / ("style_light.qss" if self.current_theme == "dark" else "style.qss")
        if not target.exists():
            return
        app.setStyleSheet(target.read_text(encoding="utf-8"))
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        try:
            self.title_bar.themeBtn.setText("\u263d" if self.current_theme == "light" else "\u2600")
        except Exception:
            pass
        # Re-polish so the [status] property selector re-evaluates under the new sheet
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)

    def closeEvent(self, event):
        # A watchdog turns a hang into a diagnosis. If shutdown has not finished
        # in 12s, faulthandler dumps EVERY thread's stack to stderr and kills the
        # process, so the blocking call is named instead of needing Task Manager.
        import faulthandler
        import os
        import sys
        try:
            faulthandler.dump_traceback_later(12, exit=True)
        except Exception:
            pass

        # NOTE: no QApplication.processEvents() here. Pumping the event loop
        # while the window is closing can re-enter closeEvent and deadlock.
        try:
            self.statusBar().showMessage("Shutting down...", 0)
        except Exception:
            pass

        try:
            self.controller.stop()
        except Exception:
            import traceback
            traceback.print_exc()

        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass

        event.accept()
        try:
            QApplication.instance().quit()
        except Exception:
            pass

        # Last-resort guarantee. OpenCV, CUDA and some MQTT builds can leave
        # non-daemon or native threads alive that keep the interpreter from
        # exiting even after the Qt loop returns. Everything of ours is already
        # closed by this point, so a hard exit is safe and beats a zombie.
        if os.environ.get("AOI_NO_HARD_EXIT") != "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
