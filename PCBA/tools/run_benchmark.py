#!/usr/bin/env python3
"""
Headless comparative benchmark. Produces the performance matrix without opening
the GUI, so the measurement is not competing with Qt repaints for CPU.

    python tools/run_benchmark.py                       # all available backends
    python tools/run_benchmark.py --engines python cpp --frames 50
    python tools/run_benchmark.py --image board.png     # real board instead of synthetic
    python tools/run_benchmark.py --no-cuda             # CPU baseline

Writes reports/benchmark_<timestamp>.xlsx (or .csv) plus per-frame raw data.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "aoi_gui"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aoi_gui.config import EngineConfig  # noqa: E402
from aoi_gui.engines import (  # noqa: E402
    available_engines, create_engine, slice_origins, unavailable_reason,
)
from aoi_gui.Report import export_report  # noqa: E402


def load_frame(cfg: EngineConfig, image: str | None) -> np.ndarray:
    if image:
        p = Path(image)
        if not p.exists():
            sys.exit(f"Image not found: {p}")
        frame = cv2.imread(str(p))
        if frame is None:
            sys.exit(f"Could not decode: {p}")
        if frame.shape[1] != cfg.frame_width or frame.shape[0] != cfg.frame_height:
            print(f"  resizing {frame.shape[1]}x{frame.shape[0]} -> "
                  f"{cfg.frame_width}x{cfg.frame_height}")
            frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height),
                               interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    from aoi_gui.CameraWorker import make_test_pattern
    print("  using synthetic PCB test pattern")
    return make_test_pattern(cfg.frame_width, cfg.frame_height)


def bench_one(name: str, cfg: EngineConfig, frame: np.ndarray, frames: int) -> dict:
    print(f"\n  Loading '{name}'...")
    t0 = time.perf_counter()
    engine = create_engine(name, cfg)     # includes warmup
    load_s = time.perf_counter() - t0
    print(f"    {engine.label} | {engine.backend_desc}")
    print(f"    load+warmup: {load_s:.2f}s")

    runs = []
    for i in range(frames):
        runs.append(engine.run_frame(frame, i))
        done = i + 1
        if done % max(1, frames // 10) == 0 or done == frames:
            pct = done * 100 // frames
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r    [{bar}] {done}/{frames}", end="", flush=True)
    print()

    def col(key):
        return [r.metrics.get(key, 0.0) for r in runs]

    lat = sorted(col("total_ms"))
    result = {
        "label": engine.label,
        "backend_desc": engine.backend_desc,
        "frames": len(runs),
        "tiles": runs[0].metrics.get("tiles", 0),
        "load_warmup_s": round(load_s, 2),
        "mean_ms": statistics.fmean(lat),
        "p50_ms": lat[len(lat) // 2],
        "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        "min_ms": lat[0],
        "max_ms": lat[-1],
        # Population stdev: these are all the frames measured, not a sample.
        "stdev_ms": statistics.pstdev(lat) if len(lat) > 1 else 0.0,
        "fps": statistics.fmean(col("fps")),
        "preprocess_ms": statistics.fmean(col("preprocess_ms")),
        "inference_ms": statistics.fmean(col("inference_ms")),
        "postprocess_ms": statistics.fmean(col("postprocess_ms")),
        "cpu_percent": statistics.fmean(col("cpu_percent")),
        "gpu_percent": statistics.fmean(col("gpu_percent")),
        "rss_mb": statistics.fmean(col("rss_mb")),
        "peak_rss_mb": max(col("peak_rss_mb")),
        "detections": statistics.fmean(col("detections")),
    }
    engine.close()
    return result, runs


def main():
    cfg = EngineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=None,
                    choices=["sim", "python", "cpp"],
                    help="default: every backend that can load")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--image", default=None, help="4K board image; omit for synthetic")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--csv", action="store_true", help="force CSV instead of xlsx")
    args = ap.parse_args()

    if args.no_cuda:
        cfg.use_cuda = cfg.use_fp16 = False

    print("=" * 70)
    print("Op-Ins 2601 - headless comparative benchmark")
    print("=" * 70)
    print(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    frame = load_frame(cfg, args.image)
    origins = slice_origins(frame.shape, cfg)
    print(f"  frame {frame.shape[1]}x{frame.shape[0]}, "
          f"{len(origins)} SAHI tiles of {cfg.tile_width}x{cfg.tile_height}")
    print(f"  conf {cfg.conf_threshold}  nms {cfg.nms_threshold}  "
          f"cuda {cfg.use_cuda}  frames {args.frames}")

    avail = available_engines(cfg)
    targets = args.engines or [n for n, ok in avail.items() if ok and n != "sim"]
    if not targets:
        print("\nNo real backend available:")
        for n in ("python", "cpp"):
            print(f"  {n}: {unavailable_reason(n, cfg)}")
        print("\nFalling back to the simulation engine so the harness is still exercised.")
        targets = ["sim"]

    summary, history = {}, []
    for name in targets:
        if not avail.get(name):
            summary[name] = {"error": unavailable_reason(name, cfg)}
            print(f"\n  skipping '{name}': {summary[name]['error']}")
            continue
        try:
            result, runs = bench_one(name, cfg, frame, args.frames)
            summary[name] = result
            for r in runs:
                rec = {k: r.metrics.get(k) for k in (
                    "engine", "tiles", "preprocess_ms", "inference_ms",
                    "postprocess_ms", "fps", "cpu_percent", "gpu_percent", "rss_mb")}
                rec["latency_ms"] = r.metrics.get("total_ms")
                rec["total_defects"] = r.metrics.get("detections")
                rec["timestamp"] = round(time.time(), 3)
                history.append(rec)
        except Exception as e:
            summary[name] = {"error": str(e)}
            print(f"\n  '{name}' failed: {e}")

    py, cpp = summary.get("python", {}), summary.get("cpp", {})
    if "error" not in py and "error" not in cpp and py and cpp and cpp.get("mean_ms"):
        summary["speedup"] = py["mean_ms"] / cpp["mean_ms"]
        summary["memory_delta_mb"] = py["rss_mb"] - cpp["rss_mb"]

    # ---- console matrix ----
    rows = [
        ("Backend", "backend_desc", "{}"),
        ("Frames", "frames", "{}"),
        ("Tiles/frame", "tiles", "{}"),
        ("Mean latency", "mean_ms", "{:.2f} ms"),
        ("p50 / p95", None, None),
        ("Std dev", "stdev_ms", "{:.2f} ms"),
        ("Effective FPS", "fps", "{:.2f}"),
        ("Preprocess", "preprocess_ms", "{:.2f} ms"),
        ("Inference", "inference_ms", "{:.2f} ms"),
        ("Postprocess", "postprocess_ms", "{:.2f} ms"),
        ("CPU load", "cpu_percent", "{:.1f} %"),
        ("GPU load", "gpu_percent", "{:.1f} %"),
        ("Memory RSS", "rss_mb", "{:.0f} MB"),
        ("Peak RSS", "peak_rss_mb", "{:.0f} MB"),
    ]
    cols = [n for n in targets if isinstance(summary.get(n), dict)
            and "error" not in summary[n]]

    print("\n" + "=" * 70)
    print("PERFORMANCE MATRIX")
    print("=" * 70)
    header = f"{'Metric':<16}" + "".join(f"{n:<26}" for n in cols)
    print(header)
    print("-" * len(header))
    for label, key, fmt in rows:
        line = f"{label:<16}"
        for n in cols:
            d = summary[n]
            if key is None:
                v = f"{d['p50_ms']:.2f} / {d['p95_ms']:.2f} ms"
            else:
                raw = d.get(key)
                v = fmt.format(raw) if isinstance(raw, (int, float)) else str(raw)
            line += f"{v[:25]:<26}"
        print(line)

    if "speedup" in summary:
        print("-" * len(header))
        print(f"{'C++ speedup':<16}{summary['speedup']:.2f}x on mean latency")
        print(f"{'Memory saved':<16}{summary['memory_delta_mb']:.0f} MB")
        print("\nNote: Path A runs .pt via PyTorch, Path B runs .onnx via OpenCV DNN.")
        print("Part of the gap is runtime, not language.")

    latest = dict(history[-1]) if history else {}
    latest["backend_desc"] = summary.get(cols[-1], {}).get("backend_desc", "") if cols else ""
    path = export_report(cfg.report_dir, latest, history, summary,
                         prefer_xlsx=not args.csv, prefix="benchmark")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
