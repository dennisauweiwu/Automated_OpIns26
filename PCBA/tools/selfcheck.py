#!/usr/bin/env python3
"""
Environment self-check. Run this before the GUI to see exactly what is wired up
and what is missing, without waiting for a window to appear.

    python tools/selfcheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "aoi_gui"))

OK, BAD, WARN = "  [ok]  ", "  [--]  ", "  [!!]  "


def check_import(mod: str, why: str, required: bool = False):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "")
        print(f"{OK}{mod:<18} {ver:<10} {why}")
        return True
    except Exception:
        print(f"{BAD if not required else WARN}{mod:<18} {'':<10} MISSING - {why}")
        return False


def main():
    print("=" * 74)
    print("Op-Ins AOI - environment self-check")
    print("=" * 74)
    print(f"Python {sys.version.split()[0]}")
    print(f"Project root: {PROJECT_ROOT}\n")

    print("Core (GUI will not start without these):")
    core = all([
        check_import("PyQt5", "GUI toolkit", required=True),
        check_import("numpy", "arrays", required=True),
        check_import("cv2", "image ops / DNN", required=True),
    ])

    print("\nMetrics:")
    check_import("psutil", "CPU + RAM columns")
    check_import("pynvml", "GPU load + VRAM columns")

    print("\nTest Path A (Python engine):")
    check_import("ultralytics", "YOLOv8 inference")
    check_import("torch", "PyTorch runtime")

    print("\nTest Path B (C++ engine):")
    check_import("yolo_backend_cpp", "compiled pybind11 extension")

    print("\nHardware (optional):")
    check_import("serial", "signal tower")
    check_import("paho.mqtt.client", "line alerts")

    print("\nProject files:")
    for rel in ("main.py", "ui/style.qss", "ui/style_light.qss",
                "cpp_backend/yolo_backend_cpp.cpp", "cpp_backend/CMakeLists.txt"):
        p = PROJECT_ROOT / rel
        print(f"{OK if p.exists() else BAD}{rel}")

    if not core:
        print("\nCore dependencies missing - install with: pip install -r requirements.txt")
        return 1

    from aoi_gui.config import EngineConfig
    from aoi_gui.engines import available_engines, slice_origins, unavailable_reason

    cfg = EngineConfig()
    print("\nModel weights:")
    for label, path in (("Test Path A (.pt)", cfg.model_pt),
                        ("Test Path B (.onnx)", cfg.model_onnx)):
        p = Path(path)
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else ""
        print(f"{OK if p.exists() else BAD}{label:<22} {size:<10} {path}")

    print("\nSAHI tiling plan:")
    origins = slice_origins((cfg.frame_height, cfg.frame_width, 3), cfg)
    xs = sorted({x for x, _ in origins})
    ys = sorted({y for _, y in origins})
    ov_x = [cfg.tile_width - (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    ov_y = [cfg.tile_height - (ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
    print(f"       frame {cfg.frame_width}x{cfg.frame_height}, "
          f"tile {cfg.tile_width}x{cfg.tile_height}, requested overlap {cfg.overlap_ratio:.0%}")
    print(f"       -> {len(xs)} cols x {len(ys)} rows = {len(origins)} tiles per frame")
    if ov_x:
        print(f"       -> actual overlap x {ov_x[0] / cfg.tile_width:.0%}, "
              f"y {ov_y[0] / cfg.tile_height:.0%}")
        if ov_x[0] / cfg.tile_width > cfg.overlap_ratio + 0.05:
            print(f"{WARN}actual overlap exceeds the request: the tile size does not")
            print("         divide the frame evenly, so an extra column/row is forced.")
            print("         Every frame costs len(tiles) inferences - see README.")

    print("\nEngines available now:")
    for name, ok in available_engines(cfg).items():
        note = "" if ok else f"  ({unavailable_reason(name, cfg)})"
        print(f"{OK if ok else BAD}{name}{note}")

    print("\nReady. Launch with:  python main.py --engine sim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
