#!/usr/bin/env python3
"""
Op-Ins 2601 - AOI Comparative Benchmark
Entry point.

    python main.py                    # simulation engine, runs anywhere
    python main.py --engine python    # Test Path A (needs ultralytics + .pt)
    python main.py --engine cpp       # Test Path B (needs the built .pyd + .onnx)
    python main.py --theme light
    python main.py --config my.json
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# The compiled extension is dropped into aoi_gui/ by CMake
sys.path.insert(0, str(PROJECT_ROOT / "aoi_gui"))

from PyQt5.QtCore import Qt                                    # noqa: E402
from PyQt5.QtWidgets import QApplication                       # noqa: E402

from aoi_gui.config import EngineConfig                        # noqa: E402
from aoi_gui.MainWindow import MainWindow                      # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Op-Ins AOI comparative benchmark GUI")
    p.add_argument("--engine", default="sim", choices=["sim", "python", "cpp"],
                   help="initial inference backend (default: sim)")
    p.add_argument("--theme", default="dark", choices=["dark", "light"])
    p.add_argument("--config", default=None, help="path to a JSON config file")
    p.add_argument("--camera", type=int, default=None, help="camera index override")
    p.add_argument("--no-cuda", action="store_true", help="force CPU on both paths")
    p.add_argument("--no-camera", action="store_true",
                   help="skip cv2.VideoCapture; use the synthetic feed only")
    p.add_argument("--debug-shutdown", action="store_true",
                   help="print timed checkpoints through the shutdown sequence")
    return p.parse_args()


def load_stylesheet(app: QApplication, theme: str):
    qss = PROJECT_ROOT / "ui" / ("style_light.qss" if theme == "light" else "style.qss")
    if qss.exists():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))
    else:
        print(f"[warn] stylesheet not found: {qss}")


def main():
    args = parse_args()

    cfg = EngineConfig.load(args.config) if args.config else EngineConfig()
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.no_cuda:
        cfg.use_cuda = False
        cfg.use_fp16 = False
    if args.no_camera:
        cfg.force_synthetic = True
    if args.debug_shutdown:
        os.environ["AOI_DEBUG_SHUTDOWN"] = "1"

    # Lets Ctrl-\ (SIGQUIT) or the shutdown watchdog dump every thread's stack.
    faulthandler.enable()

    Path(cfg.report_dir).mkdir(parents=True, exist_ok=True)

    # Must be set before the QApplication exists, or 4K frames render blurry
    # on a scaled Windows display.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Op-Ins AOI")
    load_stylesheet(app, args.theme)

    window = MainWindow(cfg, default_engine=args.engine)
    window.current_theme = args.theme
    window.show()

    rc = app.exec_()

    # closeEvent normally hard-exits before this. If we do get here, flush and
    # bypass interpreter teardown: native threads from OpenCV/CUDA/MQTT can
    # otherwise keep the process alive with no window on screen.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("AOI_NO_HARD_EXIT") == "1":
        sys.exit(rc)
    os._exit(rc)


if __name__ == "__main__":
    main()
