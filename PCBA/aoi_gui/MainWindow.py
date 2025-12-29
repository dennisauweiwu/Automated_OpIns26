# aoi_gui/MainWindow.py

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStatusBar,
    QGroupBox, QFormLayout, QPushButton, QSlider, QSizePolicy, QGraphicsOpacityEffect,
    QMenuBar, QAction
)
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap
import cv2
from pathlib import Path

    
from aoi_gui.Components import TitleBar
from aoi_gui.Controller import Controller
from PyQt5.QtWidgets import QApplication


class MainWindow(QMainWindow):
    # Signal to pass the frame from the CameraWorker to the AIWorker
    trigger_ai_signal = pyqtSignal(np.ndarray) 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Op-Ins AOI System")
        # Track current theme: 'dark' (default) or 'light'
        self.current_theme = 'dark'
        self.init_ui()
        self.init_threads() 

    def init_ui(self):
        # Setting Window Size and Title
        self.setWindowTitle("Op-Ins 2601 - GUI Development Mode")
        self.setGeometry(100, 100, 1200, 800)

        # Load centralized stylesheet (if available)
        try:
            self._load_stylesheet()
        except Exception:
            pass

        # Use frameless window and insert a custom title bar so it matches the theme
        self.setWindowFlag(Qt.FramelessWindowHint)

        # --- Main Layout (QHBoxLayout) with TitleBar (frameless window) ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout(main_widget)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Custom TitleBar
        self.title_bar = TitleBar(self)
        vbox.addWidget(self.title_bar)

        # Menu bar (Menu -> Exit)
        menubar = QMenuBar(self)
        menubar.setObjectName('MenuBar')
        file_menu = menubar.addMenu('&Menu')
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        vbox.addWidget(menubar)

        # Main content layout
        main_layout = QHBoxLayout()
        vbox.addLayout(main_layout) 

        # 1. Video Display Area (Large Left Side)
        self.video_label = QLabel("Waiting for Camera Feed...")
        self.video_label.setObjectName('VideoLabel')
        self.video_label.setAlignment(Qt.AlignCenter)
        # Responsive Design Configuration
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Use QSizePolicy here
        self.video_label.setMinimumSize(400, 300)
        # Do not call setScaledContents(True) — we scale and letterbox frames explicitly to preserve aspect ratio

        main_layout.addWidget(self.video_label, 3)


        # 2. Control & Status Sidebar (Right Side)
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        main_layout.addWidget(sidebar_widget, 1) 

        # --- Sidebar Components ---
        sidebar_layout.addWidget(self._create_stats_group())
        sidebar_layout.addWidget(self._create_control_group())
        sidebar_layout.addWidget(self._create_action_buttons())

        # Spacer
        sidebar_layout.addStretch(1)

        
        self.setStatusBar(QStatusBar(self))

    def _on_capture_and_analyze(self):
        """Placeholder: trigger a single capture and analysis cycle (mock)."""
        # Show quick feedback in the status bar; integration with real capture can replace this
        self.statusBar().showMessage("Capture & Analyze triggered (mock)", 3000)

    def _create_stats_group(self):
        stats_group = QGroupBox("Inspection Report")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # 1. BIG Dynamic Status Indicator
        self.status_indicator = QLabel("LOADING...")
        self.status_indicator.setObjectName("StatusIndicator") 
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setMinimumHeight(50)
        # Use a dynamic property for status (loading | pass | fail) and rely on stylesheet rules
        self.status_indicator.setProperty('status', 'loading')
        # Ensure the style is applied immediately
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)
        # Initialize fade animations for status transitions
        self._init_status_animation()
        main_layout.addWidget(self.status_indicator)   

        # 2. Add a QFormLayout specifically for the detailed metrics (Rows)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 6, 0, 0)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(8)
        
        self.defects_label = QLabel("0")
        self.type_label = QLabel("N/A")
        self.yield_label = QLabel("100.0%")
        self.latency_label = QLabel("0 ms")

        form_layout.addRow("Total Defects:", self.defects_label)
        form_layout.addRow("Top Defect Type:", self.type_label)
        form_layout.addRow("Inspection Yield:", self.yield_label)
        form_layout.addRow("Inference Latency:", self.latency_label)
    
        main_layout.addLayout(form_layout)
        
        stats_group.setLayout(main_layout)
        return stats_group

    def _create_control_group(self):
        control_group = QGroupBox("Hardware/Light Controls")
        layout = QFormLayout()
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        
        # Dummy Slider for Brightness
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(128)
        layout.addRow("Brightness:", self.brightness_slider)
        
        control_group.setLayout(layout)
        return control_group

    def _create_action_buttons(self):
        """Factory: Create a compact widget that contains the report and capture buttons."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)

        self.report_button = QPushButton("Export Report (.xlsx)")
        self.report_button.setObjectName("reportButton")
        self.report_button.setMinimumHeight(44)
        # Make the report button act like the Capture button for now (placeholder behavior)
        self.report_button.setEnabled(True)
        self.report_button.clicked.connect(self._on_capture_and_analyze)
        v.addWidget(self.report_button)

        self.capture_button = QPushButton("Capture and Analyze")
        self.capture_button.setObjectName("captureButton")
        self.capture_button.setMinimumHeight(44)
        self.capture_button.clicked.connect(self._on_capture_and_analyze)
        v.addWidget(self.capture_button)

        return w

    def init_threads(self):
        """Create and start the Controller which owns threads and hardware interfaces.
        Keep only GUI-related signal connections here (status & result callbacks).
        """
        # Instantiate controller and start
        self.controller = Controller(camera_index=0, tower_port='COM3')

        # Connect worker status signals to the main window status bar
        try:
            self.controller.cam_worker.status_signal.connect(self.statusBar().showMessage)
        except Exception:
            pass
        try:
            self.controller.ai_worker.status_signal.connect(self.statusBar().showMessage)
        except Exception:
            pass
        try:
            self.controller.tower_controller.status_signal.connect(self.statusBar().showMessage)
        except Exception:
            pass

        # Connect AI detection results to GUI
        self.controller.ai_worker.detection_ready.connect(self.update_results_slot)

        # Start controller-managed threads
        self.controller.start()
        
    @pyqtSlot(np.ndarray, dict)
    def update_results_slot(self, annotated_img: np.ndarray, stats_dict: dict):
        """Receives final annotated image and statistics from the AI Worker and updates the GUI."""
        
        # 1. Main Logic: Determine Pass/Fail Status
        total_defects = stats_dict.get('total_defects', 0)
        is_pcba_pass = total_defects == 0
        
        # 2. Update Statistics Labels
        self.defects_label.setText(str(total_defects))
        self.type_label.setText(stats_dict.get('top_defect_type', 'N/A'))
        self.latency_label.setText(f"{stats_dict.get('latency_ms', 0):.2f} ms") 

        # 3. Dynamic Status Indicator (animated)
        if is_pcba_pass:
            self._animate_status_change("PASS", "pass")
        else:
            # Show a simple FAIL status in the UI; keep defect counts in the detailed labels/report
            self._animate_status_change("FAIL", "fail")

        # (style applied during animation) 

        # 4. Hardware and Communication (delegated to controller)
        try:
            self.controller.tower_controller.set_status(is_pcba_pass)
        except Exception:
            pass
        if not is_pcba_pass:
            try:
                self.controller.mqtt_publisher.publish_alert(stats_dict)
            except Exception:
                pass

        # 5. Display Image (Responsive Video Display)
        target_size = self.video_label.size()
        q_img = self._convert_cv_to_qimage(annotated_img, target_size.width(), target_size.height())
        pixmap = QPixmap.fromImage(q_img)
        # The QImage returned is already letterboxed to target size — set directly
        self.video_label.setPixmap(pixmap)


    # Helper function to convert OpenCV image to a PyQt displayable image
    def _convert_cv_to_qimage(self, cv_img: np.ndarray, target_w: int = None, target_h: int = None) -> QImage:
        """Convert a BGR OpenCV image to a QImage.
        If target_w/target_h are provided, letterbox the frame into that exact size while preserving aspect ratio.
        """
        # If target size not provided, fall back to original behavior
        if target_w is None or target_h is None:
            if cv_img is None:
                return QImage(640, 480, QImage.Format_RGB888)  # Return black image if empty
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Create a black background of the exact target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if cv_img is None:
            rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * target_w
            return QImage(rgb_image.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)

        src_h, src_w = cv_img.shape[:2]
        # Compute scale to fit while preserving aspect ratio
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        # Resize the frame and blit onto the center of the canvas
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized

        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * target_w
        return QImage(rgb_image.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)

    # --- Status label animations (fade out/in) ---
    def _init_status_animation(self):
        """Prepare fade in/out animations for the status indicator."""
        self.status_opacity = QGraphicsOpacityEffect(self.status_indicator)
        self.status_indicator.setGraphicsEffect(self.status_opacity)

        self._fade_out = QPropertyAnimation(self.status_opacity, b"opacity")
        self._fade_out.setDuration(400)  # doubled duration for a smoother, longer transition
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InOutQuad)

        self._fade_in = QPropertyAnimation(self.status_opacity, b"opacity")
        self._fade_in.setDuration(400)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.InOutQuad)

    def _load_stylesheet(self):
        """Load `ui/style.qss` if present and **only apply it** when the application has no stylesheet.

        This avoids duplicating rules when `main.py` already loaded the theme at startup.
        """
        app = QApplication.instance()
        if app is None:
            return

        # If the application already has a stylesheet, we assume `main.py` loaded it and skip.
        if app.styleSheet().strip():
            return

        qss_path = Path(__file__).resolve().parents[1] / 'ui' / 'style.qss'
        if qss_path.exists():
            try:
                with open(qss_path, 'r', encoding='utf-8') as fh:
                    app.setStyleSheet(fh.read())
            except Exception:
                pass

    def toggle_theme(self):
        """Switch between dark and light theme by loading the appropriate QSS file."""
        app = QApplication.instance()
        if app is None:
            return

        base = Path(__file__).resolve().parents[1] / 'ui'
        dark_path = base / 'style.qss'
        light_path = base / 'style_light.qss'

        try:
            if self.current_theme == 'dark':
                if light_path.exists():
                    with open(light_path, 'r', encoding='utf-8') as fh:
                        app.setStyleSheet(fh.read())
                    self.current_theme = 'light'
                    try:
                        self.title_bar.themeBtn.setText('🌙')
                    except Exception:
                        pass
            else:
                if dark_path.exists():
                    with open(dark_path, 'r', encoding='utf-8') as fh:
                        app.setStyleSheet(fh.read())
                    self.current_theme = 'dark'
                    try:
                        self.title_bar.themeBtn.setText('☀')
                    except Exception:
                        pass
        except Exception:
            pass

    def _animate_status_change(self, text: str, prop: str):
        """Animate fade out, update status text/property, then fade in."""
        # Skip if nothing to change
        if self.status_indicator.text() == text and self.status_indicator.property('status') == prop:
            return

        # Disconnect previous finished callback if any
        try:
            self._fade_out.finished.disconnect(self._on_fade_out_finished)
        except Exception:
            pass

        def _on_fade_out_finished():
            self.status_indicator.setText(text)
            self.status_indicator.setProperty('status', prop)
            # Re-apply stylesheet and start fade-in
            self.status_indicator.style().unpolish(self.status_indicator)
            self.status_indicator.style().polish(self.status_indicator)
            self._fade_in.start()
            # Disconnect local slot
            try:
                self._fade_out.finished.disconnect(_on_fade_out_finished)
            except Exception:
                pass

        # Attach and start fade out
        self._fade_out.finished.connect(_on_fade_out_finished)
        self._fade_out.start()

    def closeEvent(self, event):
        """Ensure all threads and connections stop cleanly when the main window is closed."""
        try:
            self.controller.stop()
        except Exception:
            pass
        event.accept()
