# aoi_gui/Hardware.py
"""
Hardware interfaces: signal tower over serial, alert publishing over MQTT.

Both degrade to no-ops with a status message when the dependency or the device
is missing, so the GUI runs on a development machine without pyserial, without
a broker, and without the tower attached.
"""

from __future__ import annotations

import json
import time

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from aoi_gui.config import EngineConfig

try:
    import serial
    _SERIAL = True
except ImportError:
    _SERIAL = False

try:
    import paho.mqtt.client as mqtt
    _MQTT = True
except ImportError:
    _MQTT = False


class TowerController(QObject):
    """Three-colour signal tower. Green = pass, red = fail, amber = busy."""

    status_signal = pyqtSignal(str)

    CMD = {"pass": b"G\n", "fail": b"R\n", "busy": b"A\n", "off": b"O\n"}

    def __init__(self, cfg: EngineConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._port = None
        self.connected = False
        self._closing = False

    def connect(self):
        if not _SERIAL:
            self.status_signal.emit("pyserial not installed - tower disabled.")
            return
        try:
            # write_timeout is the critical one. `timeout` is the READ timeout;
            # without write_timeout pyserial blocks FOREVER on write() if the
            # device stops draining its buffer or asserts hardware flow control,
            # which hangs whatever thread called it - including shutdown.
            # rtscts/dsrdtr are pinned off for the same reason: a floating CTS
            # line on a USB-serial adapter will stall every write.
            self._port = serial.Serial(
                self.cfg.tower_port,
                self.cfg.tower_baud,
                timeout=0.5,
                write_timeout=0.5,
                rtscts=False,
                dsrdtr=False,
            )
            time.sleep(0.2)
            self.connected = True
            self.status_signal.emit(f"Signal tower connected on {self.cfg.tower_port}")
        except Exception as e:
            self.connected = False
            self._port = None
            self.status_signal.emit(f"Tower unavailable ({e}) - continuing without it.")

    @pyqtSlot(bool)
    def set_status(self, is_pass: bool):
        self._write("pass" if is_pass else "fail")

    def set_busy(self):
        self._write("busy")

    def _write(self, key: str):
        if self._closing or not (self.connected and self._port):
            return
        try:
            self._port.write(self.CMD[key])
        except Exception as e:
            # Covers SerialTimeoutException plus unplug/permission errors. One
            # failure disables the tower for the rest of the session rather than
            # retrying a port that is not responding.
            self.connected = False
            self.status_signal.emit(f"Tower write failed ({e}) - tower disabled.")

    def close(self):
        """Must never block: this runs on the GUI thread during shutdown."""
        self._closing = True
        port, self._port = self._port, None
        self.connected = False
        if port is None:
            return

        # Best-effort lamps-off. Bounded by write_timeout, and skipped entirely
        # if the port already failed, so a dead tower cannot stall the exit.
        try:
            port.write(self.CMD["off"])
        except Exception:
            pass
        # Discard anything still queued so close() has nothing left to flush.
        for method in ("reset_output_buffer", "reset_input_buffer"):
            try:
                getattr(port, method)()
            except Exception:
                pass
        try:
            port.close()
        except Exception:
            pass


class MQTTPublisher(QObject):
    """Publishes defect alerts to the line MQTT broker."""

    status_signal = pyqtSignal(str)

    def __init__(self, cfg: EngineConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._client = None
        self.connected = False

    def connect(self):
        if not _MQTT:
            self.status_signal.emit("paho-mqtt not installed - alerts disabled.")
            return
        try:
            self._client = mqtt.Client()
            self._client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
            self._client.loop_start()
            # Some paho versions spawn a NON-daemon network thread, which keeps
            # the interpreter alive after the Qt loop exits even though the
            # window is gone. Force daemon so a missed loop_stop cannot hang
            # the process.
            try:
                if getattr(self._client, "_thread", None) is not None:
                    self._client._thread.daemon = True
            except Exception:
                pass
            self.connected = True
            self.status_signal.emit(
                f"MQTT connected to {self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        except Exception as e:
            self.connected = False
            self.status_signal.emit(f"MQTT unavailable ({e}) - continuing without it.")

    @pyqtSlot(dict)
    def publish_alert(self, stats: dict):
        if not (self.connected and self._client):
            return
        payload = {
            "timestamp": time.time(),
            "total_defects": stats.get("total_defects", 0),
            "top_defect_type": stats.get("top_defect_type", "N/A"),
            "defect_counts": stats.get("defect_counts", {}),
            "latency_ms": round(stats.get("latency_ms", 0.0), 2),
            "engine": stats.get("engine", "unknown"),
        }
        try:
            self._client.publish(self.cfg.mqtt_topic, json.dumps(payload), qos=1)
        except Exception as e:
            self.status_signal.emit(f"MQTT publish failed: {e}")

    def close(self):
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self.connected = False
