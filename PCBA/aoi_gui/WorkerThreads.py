# aoi_gui/WorkerThreads.py

import cv2
import time
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

# --------------------
# 1. Camera Worker (Dummy)
# --------------------

class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray) 
    status_signal = pyqtSignal(str)      

    def __init__(self, camera_index=0):
        super().__init__()
        self._is_running = True
        self.frame_counter = 0

    @pyqtSlot()
    def run(self):
        """The loop that executes inside the dedicated thread (DUMMY MODE)."""
        self.status_signal.emit("DUMMY Camera feed started (Simulating 15 FPS).")
        
        while self._is_running:
            # Generate a Simple Dummy Frame 
            frame = np.zeros((480, 640, 3), dtype=np.uint8) 
            
            # Alternate colors and text to prove the feed is 'live'
            is_defect = (self.frame_counter // 60) % 2 == 1
            color = (0, 0, 255) if is_defect else (0, 255, 0)
            
            cv2.rectangle(frame, (100, 100), (540, 380), color, -1)
            cv2.putText(frame, f"Frame: {self.frame_counter} (Defect Sim: {is_defect})", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
            
            self.frame_ready.emit(frame)
            
            self.frame_counter += 1
            time.sleep(1/5) # Simulate 15 FPS 

    def stop(self):
        self._is_running = False


# --------------------
# 2. AI Worker (Dummy)
# --------------------

class AIWorker(QObject):
    detection_ready = pyqtSignal(np.ndarray, dict)
    status_signal = pyqtSignal(str)

    def __init__(self, model_path='data/models/yolov8n.pt'):
        super().__init__()
        self.status_signal.emit("AI Worker Mock Ready. Awaiting frames...")
        self.defect_cycle = 0

    @pyqtSlot(np.ndarray)
    def process_frame(self, frame):
        """Slot to receive frames and simulate processing."""
        
        start_time = time.perf_counter()
        
        # Simulate AI work
        time.sleep(0.05) # 50ms inference time

        # Simulate Defect Detection Logic (Corresponds to the frame color change)
        is_defect_frame = (self.defect_cycle // 60) % 2 == 1
        
        # Determine simulated statistics based on the cycle
        if is_defect_frame:
            defect_stats = {'total_defects': 3, 'top_defect_type': 'Solder Bridge', 'latency_ms': 50.0}
        else:
            defect_stats = {'total_defects': 0, 'top_defect_type': 'N/A', 'latency_ms': 50.0}

        # --- Simulate Annotation ---
        annotated_frame = frame.copy()
        if is_defect_frame:
             # Draw a simulated bounding box to prove annotation works
            cv2.rectangle(annotated_frame, (150, 150), (250, 250), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "DEFECT", (155, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        end_time = time.perf_counter()
        defect_stats['latency_ms'] = (end_time - start_time) * 1000 # Update with real latency

        self.status_signal.emit(f"Mock Inference Latency: {defect_stats['latency_ms']:.2f} ms")
        self.detection_ready.emit(annotated_frame, defect_stats)
        
        self.defect_cycle += 1