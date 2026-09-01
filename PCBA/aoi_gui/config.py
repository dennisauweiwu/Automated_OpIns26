# aoi_gui/config.py
"""
Single source of truth for the comparative benchmark.

Both Test Path A (Python) and Test Path B (C++) read the SAME EngineConfig
instance. Any parameter that differs between the two paths would contaminate
the measurement, so nothing here is duplicated per-engine on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EngineConfig:
    # --- Model artefacts -----------------------------------------------------
    # Both must be exported from the SAME training run, or accuracy differences
    # will be attributed to the language when they are really weight drift.
    model_pt: str = str(PROJECT_ROOT / "datasets/weights/yolov8_pcb_4k.pt")
    model_onnx: str = str(PROJECT_ROOT / "datasets/weights/yolov8_pcb_4k.onnx")

    class_names: Tuple[str, ...] = (
        "missing_hole", "mouse_bite", "open_circuit",
        "short", "spur", "spurious_copper",
    )

    # --- Inference -----------------------------------------------------------
    input_size: int = 640
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    use_cuda: bool = True
    use_fp16: bool = True
    warmup_runs: int = 5

    # --- SAHI slicing --------------------------------------------------------
    tile_width: int = 1920
    tile_height: int = 1080
    overlap_ratio: float = 0.20

    # --- Capture -------------------------------------------------------------
    camera_index: int = 0
    # Skip cv2.VideoCapture entirely. A camera driver stuck inside read() is a
    # classic cause of a thread that will not join on shutdown; this isolates it.
    force_synthetic: bool = False
    frame_width: int = 3840
    frame_height: int = 2160
    target_fps: int = 30

    # --- Hardware ------------------------------------------------------------
    tower_port: str = "COM3"
    tower_baud: int = 9600
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "opins/aoi/alerts"

    # --- Output --------------------------------------------------------------
    report_dir: str = str(PROJECT_ROOT / "reports")

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EngineConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        data["class_names"] = tuple(data.get("class_names", cls.class_names))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
