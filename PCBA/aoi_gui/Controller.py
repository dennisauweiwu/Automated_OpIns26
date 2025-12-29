# aoi_gui/Controller.py

from PyQt5.QtCore import QThread

from aoi_gui.WorkerThreads import CameraWorker, AIWorker
from aoi_core.HardwareControl import TowerLightController
from aoi_core.DataPublisher import MQTTPublisher

class Controller:
    """Encapsulates all non-UI logic: camera/AI threads and hardware/publisher setup.

    Responsibilities:
    - Instantiate workers and threads
    - Wire internal worker signals
    - Start/stop threads and perform clean shutdown
    """
    def __init__(self, camera_index=0, tower_port='COM3'):
        # Camera
        self.cam_thread = QThread()
        self.cam_worker = CameraWorker(camera_index=camera_index)
        self.cam_worker.moveToThread(self.cam_thread)

        # AI
        self.ai_thread = QThread()
        self.ai_worker = AIWorker()
        self.ai_worker.moveToThread(self.ai_thread)

        # Hardware / comms
        self.tower_controller = TowerLightController(port=tower_port)
        self.mqtt_publisher = MQTTPublisher()

        # Internal wiring
        # Camera frames -> AI processing
        self.cam_worker.frame_ready.connect(self.ai_worker.process_frame)

        # On thread starts, run the camera worker's run method
        self.cam_thread.started.connect(self.cam_worker.run)
        # Start AI thread with a no-op or status if needed; UI can connect to ai_worker.status_signal

    def start(self):
        self.cam_thread.start()
        self.ai_thread.start()

    def stop(self):
        # Stop camera worker cleanly
        try:
            self.cam_worker.stop()
        except Exception:
            pass

        # Stop threads
        try:
            self.cam_thread.quit()
            self.cam_thread.wait()
        except Exception:
            pass

        try:
            self.ai_thread.quit()
            self.ai_thread.wait()
        except Exception:
            pass

        # Close hardware and publisher
        try:
            self.tower_controller.close()
        except Exception:
            pass
        try:
            self.mqtt_publisher.close()
        except Exception:
            pass
