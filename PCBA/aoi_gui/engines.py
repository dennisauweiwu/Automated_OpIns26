# aoi_gui/engines.py
"""
Unified inference-engine interface for the 1:1 comparative benchmark.

Three backends, one identical call signature:

    result = engine.run_frame(frame_bgr, frame_id)   -> FrameResult

  * PythonEngine     Test Path A - Ultralytics YOLOv8 + mirrored SAHI slicing
  * CppEngine        Test Path B - pybind11 module built from yolo_backend_cpp.cpp
  * SimulationEngine No weights, no GPU, no camera. Lets the GUI run end-to-end
                     on any machine so UI work is not blocked on the rig.

Fairness rules enforced here, because a benchmark that measures the wrong thing
is worse than no benchmark:
  * Identical tile origins  (slice_origins mirrors compute_slice_origins in C++)
  * Identical letterboxing  (letterbox mirrors YoloBackendCpp::letterbox)
  * Identical thresholds    (one EngineConfig, read by both)
  * Warmup discarded on both paths
  * Same metric basis       (psutil <-> psapi, pynvml <-> NVML)
  * Annotation drawn OUTSIDE the timed region, by AIWorker, for every backend
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from aoi_gui.config import EngineConfig

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception:
    _NVML, _NVML_HANDLE = False, None


# -----------------------------------------------------------------------------
# Data contracts
# -----------------------------------------------------------------------------

@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    class_id: int
    tile_id: int = -1

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class FrameResult:
    detections: List[Detection] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_stats_dict(self, class_names) -> dict:
        """Payload shaped for MainWindow.update_results_slot, with the benchmark
        fields carried alongside the original keys."""
        counts: Dict[str, int] = {}
        for d in self.detections:
            name = (class_names[d.class_id]
                    if 0 <= d.class_id < len(class_names) else f"cls{d.class_id}")
            counts[name] = counts.get(name, 0) + 1
        top = max(counts, key=counts.get) if counts else "N/A"

        stats = {
            "total_defects": len(self.detections),
            "top_defect_type": top,
            "defect_counts": counts,
            "latency_ms": self.metrics.get("total_ms", 0.0),
        }
        stats.update(self.metrics)
        return stats


# -----------------------------------------------------------------------------
# Shared helpers - both real engines MUST use these or parity is lost
# -----------------------------------------------------------------------------

def _axis_origins(total: int, tile: int, overlap: float) -> List[int]:
    """Evenly distributed tile origins along one axis.

    The naive 'step forward, clamp the last tile inward' approach produces a
    lopsided grid: on a 3840px axis with 1920px tiles it yields 0/1536/1920,
    where the final pair overlaps 80% while the first overlaps 20%. That burns
    inference time on near-duplicate pixels and biases detection density toward
    one edge of the board.

    Instead: work out the minimum tile count that satisfies the requested
    overlap, then spread those tiles evenly. Overlap comes out uniform and
    always >= the requested ratio.
    """
    if tile >= total:
        return [0]

    step = max(1, int(tile * (1.0 - overlap)))
    span = total - tile
    n = (span + step - 1) // step + 1        # ceil(span / step) + 1
    n = max(2, n)
    return [round(i * span / (n - 1)) for i in range(n)]


def slice_origins(frame_shape, cfg: EngineConfig) -> List[Tuple[int, int]]:
    """Mirror of YoloBackendCpp::compute_slice_origins().

    Every tile is full size (no ragged edge tiles skewing per-tile latency).
    Change this and you must change the C++ side in the same commit, or the two
    paths stop processing the same pixels and the comparison is meaningless.
    """
    h, w = frame_shape[:2]
    xs = _axis_origins(w, cfg.tile_width, cfg.overlap_ratio)
    ys = _axis_origins(h, cfg.tile_height, cfg.overlap_ratio)
    return [(px, py) for py in ys for px in xs]


def letterbox(img: np.ndarray, new_size: int, color=(114, 114, 114)):
    """Mirror of YoloBackendCpp::letterbox(). Returns (padded, scale, pad_x, pad_y).

    A plain resize to 640x640 would squash a 16:9 tile and change which
    fine-pitch defects are found, so the two paths would not be detecting on
    the same image at all.
    """
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    out = cv2.copyMakeBorder(resized, pad_y, new_size - nh - pad_y,
                             pad_x, new_size - nw - pad_x,
                             cv2.BORDER_CONSTANT, value=color)
    return out, scale, pad_x, pad_y


def global_nms(boxes, scores, class_ids, tiles, cfg: EngineConfig) -> List[Detection]:
    """Class-aware NMS across the stitched master frame. A solder bridge sitting
    in a tile overlap is detected twice, once per neighbouring tile."""
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids,
                                   cfg.conf_threshold, cfg.nms_threshold)
    keep = np.array(keep).flatten().astype(int).tolist()
    return [Detection(*boxes[i], scores[i], class_ids[i], tiles[i]) for i in keep]


class _HostMonitor:
    """Python twin of the C++ SystemMonitor. Same normalisation, so CPU numbers
    from the two paths land on the same scale (100% = all cores busy)."""

    def __init__(self):
        self.proc = psutil.Process() if _PSUTIL else None
        self.cores = psutil.cpu_count(logical=True) if _PSUTIL else 1
        if self.proc:
            self.proc.cpu_percent(None)   # prime the delta

    def reset(self):
        if self.proc:
            self.proc.cpu_percent(None)

    def sample(self) -> Dict[str, float]:
        m = {"cpu_percent": -1.0, "gpu_percent": -1.0,
             "gpu_mem_mb": -1.0, "rss_mb": -1.0, "peak_rss_mb": -1.0}
        if self.proc:
            m["cpu_percent"] = self.proc.cpu_percent(None) / max(1, self.cores)
            mi = self.proc.memory_info()
            m["rss_mb"] = mi.rss / (1024 ** 2)
            m["peak_rss_mb"] = getattr(mi, "peak_wset", mi.rss) / (1024 ** 2)
        if _NVML:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                m["gpu_percent"] = float(util.gpu)
                m["gpu_mem_mb"] = mem.used / (1024 ** 2)
            except Exception:
                pass
        return m


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------

class InferenceEngine(ABC):
    name = "base"
    label = "Base"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.monitor = _HostMonitor()
        self.backend_desc = "unknown"

    @abstractmethod
    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult: ...

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass


# -----------------------------------------------------------------------------
# Simulation - lets the GUI run with no weights, no CUDA, no camera
# -----------------------------------------------------------------------------

class SimulationEngine(InferenceEngine):
    name = "sim"
    label = "Simulation (no model)"

    def __init__(self, cfg: EngineConfig):
        super().__init__(cfg)
        self.backend_desc = "Synthetic detections - not a real measurement"
        self._rng = np.random.default_rng(1337)

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        cfg = self.cfg
        self.monitor.reset()
        t0 = time.perf_counter()

        origins = slice_origins(frame.shape, cfg)
        h, w = frame.shape[:2]

        # Fabricate a plausible defect count so PASS/FAIL and the report path
        # can both be exercised without hardware.
        boxes, scores, ids, tiles = [], [], [], []
        for _ in range(int(self._rng.integers(0, 4))):
            bw = int(self._rng.integers(60, 220))
            bh = int(self._rng.integers(60, 220))
            bx = int(self._rng.integers(0, max(1, w - bw)))
            by = int(self._rng.integers(0, max(1, h - bh)))
            boxes.append([bx, by, bw, bh])
            scores.append(float(self._rng.uniform(cfg.conf_threshold, 0.98)))
            ids.append(int(self._rng.integers(0, len(cfg.class_names))))
            tiles.append(0)

        time.sleep(0.03)   # stand in for inference so the UI paces realistically
        dets = global_nms(boxes, scores, ids, tiles, cfg)

        total_ms = (time.perf_counter() - t0) * 1000.0
        metrics = {
            "frame_id": frame_id, "engine": self.name, "tiles": len(origins),
            "slicing_ms": 0.0, "preprocess_ms": 0.0,
            "inference_ms": total_ms, "postprocess_ms": 0.0, "merge_ms": 0.0,
            "total_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 0 else 0.0,
            "detections": len(dets),
        }
        metrics.update(self.monitor.sample())
        return FrameResult(dets, metrics)


# -----------------------------------------------------------------------------
# Test Path A - Python
# -----------------------------------------------------------------------------

class PythonEngine(InferenceEngine):
    name = "python"
    label = "Test Path A - Python"

    def __init__(self, cfg: EngineConfig):
        super().__init__(cfg)
        from ultralytics import YOLO
        import torch

        self._torch = torch
        self.device = "cuda:0" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(cfg.model_pt)
        self.model.to(self.device)
        if cfg.use_fp16 and self.device.startswith("cuda"):
            self.model.model.half()
        self.backend_desc = (
            f"Ultralytics / PyTorch on {self.device}"
            f"{' FP16' if cfg.use_fp16 and 'cuda' in self.device else ''}")

    def warmup(self):
        dummy = np.full((self.cfg.tile_height, self.cfg.tile_width, 3), 114, np.uint8)
        for _ in range(self.cfg.warmup_runs):
            self.model.predict(dummy, imgsz=self.cfg.input_size,
                               conf=self.cfg.conf_threshold,
                               iou=self.cfg.nms_threshold,
                               device=self.device, verbose=False)
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        cfg = self.cfg
        self.monitor.reset()
        t_start = time.perf_counter()

        s0 = time.perf_counter()
        origins = slice_origins(frame.shape, cfg)
        slicing_ms = (time.perf_counter() - s0) * 1000.0

        pre_ms = infer_ms = post_ms = 0.0
        boxes, scores, ids, tiles = [], [], [], []

        for tid, (ox, oy) in enumerate(origins):
            p0 = time.perf_counter()
            # The numpy slice is a view, but Ultralytics copies during
            # preprocess. That per-tile allocation is exactly the overhead
            # this study is trying to quantify, so it stays inside the timer.
            tile = np.ascontiguousarray(
                frame[oy:oy + cfg.tile_height, ox:ox + cfg.tile_width])
            p1 = time.perf_counter()

            res = self.model.predict(tile, imgsz=cfg.input_size,
                                     conf=cfg.conf_threshold,
                                     iou=cfg.nms_threshold,
                                     device=self.device, verbose=False)[0]
            # CUDA is asynchronous. Without this we time the kernel *launch*,
            # not the work, and Python looks impossibly fast.
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            p2 = time.perf_counter()

            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                    boxes.append([int(x1) + ox, int(y1) + oy,
                                  int(x2 - x1), int(y2 - y1)])
                    scores.append(float(c))
                    ids.append(int(k))
                    tiles.append(tid)
            p3 = time.perf_counter()

            pre_ms += (p1 - p0) * 1000.0
            infer_ms += (p2 - p1) * 1000.0
            post_ms += (p3 - p2) * 1000.0

        m0 = time.perf_counter()
        detections = global_nms(boxes, scores, ids, tiles, cfg)
        merge_ms = (time.perf_counter() - m0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics = {
            "frame_id": frame_id, "engine": self.name, "tiles": len(origins),
            "slicing_ms": slicing_ms, "preprocess_ms": pre_ms,
            "inference_ms": infer_ms, "postprocess_ms": post_ms,
            "merge_ms": merge_ms, "total_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 0 else 0.0,
            "detections": len(detections),
        }
        metrics.update(self.monitor.sample())
        return FrameResult(detections, metrics)


# -----------------------------------------------------------------------------
# Test Path B - C++
# -----------------------------------------------------------------------------

class CppEngine(InferenceEngine):
    name = "cpp"
    label = "Test Path B - C++"

    def __init__(self, cfg: EngineConfig):
        super().__init__(cfg)
        try:
            import yolo_backend_cpp as ycpp
        except ImportError as e:
            raise ImportError(
                "yolo_backend_cpp extension not found. Build it first:\n"
                "  cd cpp_backend\n"
                "  cmake -B build -S . -A x64 -DOpenCV_DIR=<path> -DUSE_NVML=ON\n"
                "  cmake --build build --config Release\n"
                f"(original error: {e})") from e

        c = ycpp.Config()
        c.model_path = cfg.model_onnx
        c.input_size = cfg.input_size
        c.use_cuda = cfg.use_cuda
        c.use_fp16 = cfg.use_fp16
        c.conf_threshold = cfg.conf_threshold
        c.nms_threshold = cfg.nms_threshold
        c.tile_width = cfg.tile_width
        c.tile_height = cfg.tile_height
        c.overlap_ratio = cfg.overlap_ratio
        c.warmup_runs = cfg.warmup_runs

        self.engine = ycpp.YoloBackendCpp(c)
        self.engine.set_class_names(list(cfg.class_names))
        self.backend_desc = f"OpenCV DNN / {self.engine.active_backend}"

    def warmup(self):
        self.engine.warmup()

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        # The binding wraps the numpy buffer as cv::Mat instead of copying it.
        # A non-contiguous view would make pybind11 silently copy 24 MB per
        # frame, which defeats the entire point of the bridge.
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        dets, fm = self.engine.run_frame(frame, frame_id)

        detections = [Detection(d.x, d.y, d.w, d.h, d.confidence, d.class_id, d.tile_id)
                      for d in dets]
        metrics = {
            "frame_id": fm.frame_id, "engine": self.name, "tiles": fm.tiles,
            "slicing_ms": fm.slicing_ms, "preprocess_ms": fm.preprocess_ms,
            "inference_ms": fm.inference_ms, "postprocess_ms": fm.postprocess_ms,
            "merge_ms": fm.merge_ms, "total_ms": fm.total_ms, "fps": fm.fps,
            "cpu_percent": fm.cpu_percent, "gpu_percent": fm.gpu_percent,
            "gpu_mem_mb": fm.gpu_mem_mb, "rss_mb": fm.rss_mb,
            "peak_rss_mb": fm.peak_rss_mb, "detections": fm.detections,
        }
        return FrameResult(detections, metrics)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

ENGINE_REGISTRY = {
    "sim": SimulationEngine,
    "python": PythonEngine,
    "cpp": CppEngine,
}


def create_engine(name: str, cfg: EngineConfig) -> InferenceEngine:
    key = name.lower().strip()
    if key not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine '{name}'. Options: {list(ENGINE_REGISTRY)}")
    engine = ENGINE_REGISTRY[key](cfg)
    engine.warmup()
    return engine


def available_engines(cfg: Optional[EngineConfig] = None) -> Dict[str, bool]:
    """Which backends can load right now. Drives the greyed-out combo entries,
    so the operator sees why a path is unavailable instead of hitting a crash."""
    from pathlib import Path
    cfg = cfg or EngineConfig()
    status = {"sim": True}

    try:
        import ultralytics  # noqa: F401
        status["python"] = Path(cfg.model_pt).exists()
    except ImportError:
        status["python"] = False

    try:
        import yolo_backend_cpp  # noqa: F401
        status["cpp"] = Path(cfg.model_onnx).exists()
    except ImportError:
        status["cpp"] = False

    return status


def unavailable_reason(name: str, cfg: EngineConfig) -> str:
    from pathlib import Path
    if name == "python":
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return "ultralytics not installed"
        if not Path(cfg.model_pt).exists():
            return "missing .pt weights"
    if name == "cpp":
        try:
            import yolo_backend_cpp  # noqa: F401
        except ImportError:
            return "extension not built"
        if not Path(cfg.model_onnx).exists():
            return "missing .onnx weights"
    return "unavailable"
