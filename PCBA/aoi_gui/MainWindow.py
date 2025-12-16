# aoi_gui/MainWindow.py

import sys
import numpy as np
# Import ALL necessary PyQt modules
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStatusBar,
    QGroupBox, QFormLayout, QPushButton, QSlider, QSizePolicy # <--- ADD QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2 
    
# Import the new worker classes
from aoi_gui.WorkerThreads import CameraWorker, AIWorker 
from aoi_core.HardwareControl import TowerLightController
from aoi_core.DataPublisher import MQTTPublisher 

class MainWindow(QMainWindow):
    # Signal to pass the frame from the CameraWorker to the AIWorker
    trigger_ai_signal = pyqtSignal(np.ndarray) 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VD25 AOI System")
        self.init_ui()
        self.init_threads() 

    def init_ui(self):
        # Setting Window Size and Title
        self.setWindowTitle("PCBA AOI System - GUI Development Mode")
        self.setGeometry(100, 100, 1200, 800) 

        # --- Main Layout (QHBoxLayout) ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 1. Video Display Area (Large Left Side)
        self.video_label = QLabel("Waiting for Camera Feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; color: #EEE; border: 1px solid gray;")
        
        # Responsive Design Configuration
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Use QSizePolicy here
        self.video_label.setMinimumSize(400, 300) 
        self.video_label.setScaledContents(True)

        main_layout.addWidget(self.video_label, 3) 

        # 2. Control & Status Sidebar (Right Side)
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        main_layout.addWidget(sidebar_widget, 1) 

        # --- Sidebar Components ---
        sidebar_layout.addWidget(self._create_stats_group())
        sidebar_layout.addWidget(self._create_control_group())

        # C. Reporting & Status Buttons
        self.report_button = QPushButton("Capture & Export Report (.xlsx)")
        self.report_button.setStyleSheet("background-color: #4CAF50; color: white; height: 40px;")
        sidebar_layout.addWidget(self.report_button)
        
        # Spacer
        sidebar_layout.addStretch(1) 
        
        self.setStatusBar(QStatusBar(self))

    def _create_stats_group(self):
        stats_group = QGroupBox("Inspection Statistics")
        main_layout = QVBoxLayout() 

        # 1. BIG Dynamic Status Indicator
        self.status_indicator = QLabel("LOADING...")
        self.status_indicator.setObjectName("StatusIndicator") 
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setMinimumHeight(50)
        self.status_indicator.setStyleSheet("background-color: gray; border-radius: 5px;") 
        main_layout.addWidget(self.status_indicator) 

        # 2. Add a QFormLayout specifically for the detailed metrics (Rows)
        form_layout = QFormLayout()
        
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
        
        # Dummy Slider for Brightness
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(128)
        layout.addRow("Brightness:", self.brightness_slider)
        
        control_group.setLayout(layout)
        return control_group

    def init_threads(self):
        # 1. --- Camera Thread Setup ---
        self.cam_thread = QThread()
        self.cam_worker = CameraWorker(camera_index=0) 
        self.cam_worker.moveToThread(self.cam_thread)

        # 2. --- AI Thread Setup ---
        self.ai_thread = QThread()
        self.ai_worker = AIWorker() 
        self.ai_worker.moveToThread(self.ai_thread)
        
        # 3. --- Hardware Controller Setup ---
        self.tower_controller = TowerLightController(port='COM3') 
        self.mqtt_publisher = MQTTPublisher()

        # 4. --- Signal Connections ---
        self.cam_worker.status_signal.connect(self.statusBar().showMessage)
        self.ai_worker.status_signal.connect(self.statusBar().showMessage)
        self.tower_controller.status_signal.connect(self.statusBar().showMessage)

        self.cam_worker.frame_ready.connect(self.ai_worker.process_frame)
        self.ai_worker.detection_ready.connect(self.update_results_slot)

        # C. Thread Control
        self.cam_thread.started.connect(self.cam_worker.run)
        self.ai_thread.started.connect(lambda: self.statusBar().showMessage("AI Thread Started."))
        
        # 5. --- Start Threads ---
        self.cam_thread.start()
        self.ai_thread.start()
        
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

        # 3. Dynamic Status Indicator (Professional UI Update)
        if is_pcba_pass:
            self.status_indicator.setText("PASS")
            self.status_indicator.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;") 
        else:
            self.status_indicator.setText(f"FAIL - {total_defects} Defects")
            self.status_indicator.setStyleSheet("background-color: #C0392B; color: white; border-radius: 5px;") 

        # 4. Hardware and Communication
        self.tower_controller.set_status(is_pcba_pass) 
        if not is_pcba_pass:
            self.mqtt_publisher.publish_alert(stats_dict) 

        # 5. Display Image (Responsive Video Display)
        q_img = self._convert_cv_to_qimage(annotated_img)
        pixmap = QPixmap.fromImage(q_img)
        
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

    # Helper function to convert OpenCV image to a PyQt displayable image
    def _convert_cv_to_qimage(self, cv_img: np.ndarray) -> QImage:
        # Check if frame is valid (not None) before processing
        if cv_img is None:
            return QImage(640, 480, QImage.Format_RGB888) # Return black image if empty
            
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def closeEvent(self, event):
        """Ensure all threads and connections stop cleanly when the main window is closed."""
        # Clean up worker threads
        self.cam_worker.stop()
        self.cam_thread.quit()
        self.cam_thread.wait()
        self.ai_thread.quit()
        self.ai_thread.wait()
        
        # Clean up connections
        self.tower_controller.close()
        self.mqtt_publisher.close()
        event.accept()