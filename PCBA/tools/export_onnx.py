#!/usr/bin/env python3
"""
Export the trained YOLOv8 .pt to the .onnx that Test Path B consumes.

Run this whenever the .pt changes. If the two artefacts come from different
training runs, accuracy differences between the paths get wrongly attributed to
the language instead of to weight drift, which invalidates the whole study.

    python tools/export_onnx.py
    python tools/export_onnx.py --pt datasets/weights/other.pt --opset 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aoi_gui.config import EngineConfig  # noqa: E402


def sha256(path: Path, blocks: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(blocks), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = EngineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=cfg.model_pt)
    ap.add_argument("--imgsz", type=int, default=cfg.input_size)
    ap.add_argument("--opset", type=int, default=12,
                    help="12 is the safest for OpenCV DNN; 17+ often fails to parse")
    ap.add_argument("--simplify", action="store_true", default=True)
    args = ap.parse_args()

    pt = Path(args.pt)
    if not pt.exists():
        sys.exit(f"Weights not found: {pt}")

    from ultralytics import YOLO

    print(f"Loading {pt}")
    model = YOLO(str(pt))

    # dynamic=False and a fixed imgsz matter: OpenCV DNN handles static shapes
    # far more reliably, and the C++ engine always feeds exactly imgsz x imgsz.
    print(f"Exporting ONNX (imgsz={args.imgsz}, opset={args.opset}, static shape)")
    out = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                       simplify=args.simplify, dynamic=False, half=False)

    out = Path(out)
    target = Path(cfg.model_onnx)
    target.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() != target.resolve():
        out.replace(target)

    names = model.names
    class_names = [names[i] for i in sorted(names)]

    manifest = {
        "pt": str(pt),
        "pt_sha256": sha256(pt),
        "onnx": str(target),
        "onnx_sha256": sha256(target),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "class_names": class_names,
    }
    man_path = target.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nONNX     : {target}")
    print(f"Manifest : {man_path}")
    print(f"Classes  : {class_names}")
    if list(cfg.class_names) != class_names:
        print("\n[!] class_names in aoi_gui/config.py does not match the model:")
        print(f"    config : {list(cfg.class_names)}")
        print(f"    model  : {class_names}")
        print("    Update config.py or the labels in the GUI will be wrong.")


if __name__ == "__main__":
    main()
