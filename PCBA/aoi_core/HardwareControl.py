# aoi_core/HardwareControl.py

from PyQt5.QtCore import QObject, pyqtSignal

class TowerLightController(QObject):
    """MOCK: Controls serial comms to ESP32 for Tower Light."""
    status_signal = pyqtSignal(str) 

    def __init__(self, port='COM3'):
        super().__init__()
        self.port = port
        self.status_signal.emit(f"TowerLightController Mock Ready (Port: {port}).")

    def set_status(self, is_pass: bool):
        """MOCK: Simulates sending command to the ESP32."""
        status = "GREEN (PASS)" if is_pass else "RED (FAIL)"
        # You would send self.ser.write("COMMAND") here
        print(f"[HW_MOCK] Status set to: {status}")

    def close(self):
        """MOCK: Simulates closing the serial port."""
        print("[HW_MOCK] Disconnecting TowerLightController.")
        self.status_signal.emit("Tower Light Disconnected.")