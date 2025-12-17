# aoi_core/DataPublisher.py

from PyQt5.QtCore import QObject, pyqtSignal

class MQTTPublisher(QObject):
    """MOCK: Handles publishing defect data via MQTT."""
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # In the real version, paho-mqtt connection happens here
        self.status_signal.emit("MQTT Publisher Mock Ready.")

    def publish_alert(self, defect_stats: dict):
        """MOCK: Simulates publishing a critical alert."""
        # You would send self.client.publish(...) here
        print(f"[MQTT_MOCK] Published Alert: Top Defect is {defect_stats.get('top_defect_type')}")

    def close(self):
        """MOCK: Simulates stopping the MQTT loop."""
        print("[MQTT_MOCK] Disconnecting Publisher.")
        self.status_signal.emit("MQTT Broker Disconnected.")