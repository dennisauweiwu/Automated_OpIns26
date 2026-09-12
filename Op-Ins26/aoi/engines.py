"""
Inference backends behind one interface.

    result = engine.run_frame(frame_bgr, frame_id)   -> FrameResult

  sim           Ground-truth-driven stand-in. No weights, no GPU, no camera.
  python        Test Path A  - Ultralytics YOLOv8 high-level predict() API.
  python-lean   Test Path A' - same PyTorch weights, but the pre/post-processing
                is the SAME code the C++ path uses. An ablation, not a third
                test path: it separates *framework* overhead from *language*
                overhead, which the headline A-vs-B number confounds.
  cpp           Test Path B  - compiled aoi_engine_cpp (ONNX Runtime + CUDA EP).

What keeps the comparison honest
--------------------------------
  * one RunConfig read by every engine, so no threshold can drift;
  * identical tile origins (pipeline.slice_origins <-> AoiEngine::slice_origins);
  * identical letterbox padding;
  * identical decode rule and NMS parameters;
  * warmup discarded on every path;
  * device synchronisation before stopping the inference clock, on every path;
  * annotation performed outside every timer, by the caller, for all engines.

The stage decomposition (pipeline.STAGES) is produced identically by the Python
and C++ paths, which is what lets the report attribute the gap to a stage rather
than asserting a single end-to-end ratio.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from aoi.config import RunConfig, bootstrap
from aoi.pipeline import (
    STAGES, Detection, FrameResult, StageTimer,
    global_nms, letterbox, slice_origins,
)

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
except Exception:                                               # noqa: BLE001
    _NVML, _NVML_HANDLE = False, None


# =============================================================================
# Host metrics
# =============================================================================

class HostMonitor:
    """Python twin of the C++ SystemMonitor.

    CPU is normalised by logical core count on both sides, so 100% means "every
    core busy" rather than "1200% on a 12-core box" on one path and something
    else on the other.
    """

    def __init__(self):
        self.proc = psutil.Process() if _PSUTIL else None
        self.cores = psutil.cpu_count(logical=True) if _PSUTIL else 1
        if self.proc:
            self.proc.cpu_percent(None)

    def reset(self) -> None:
        if self.proc:
            self.proc.cpu_percent(None)

    def gpu(self) -> Dict[str, float]:
        if not _NVML:
            return {"gpu_percent": -1.0, "gpu_mem_mb": -1.0}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            return {"gpu_percent": float(util.gpu),
                    "gpu_mem_mb": mem.used / 1024 ** 2}
        except Exception:                                       # noqa: BLE001
            return {"gpu_percent": -1.0, "gpu_mem_mb": -1.0}

    def sample(self) -> Dict[str, float]:
        m = {"cpu_percent": -1.0, "rss_mb": -1.0, "peak_rss_mb": -1.0}
        if self.proc:
            m["cpu_percent"] = self.proc.cpu_percent(None) / max(1, self.cores)
            info = self.proc.memory_info()
            m["rss_mb"] = info.rss / 1024 ** 2
            m["peak_rss_mb"] = getattr(info, "peak_wset", info.rss) / 1024 ** 2
        m.update(self.gpu())
        return m


# =============================================================================
# Shared decode - the exact rule the C++ path implements
# =============================================================================

def decode_yolov8(raw: np.ndarray, conf_threshold: float,
                  scale: float, pad_x: int, pad_y: int,
                  origin: Tuple[int, int]
                  ) -> Tuple[List[List[int]], List[float], List[int]]:
    """YOLOv8 head -> boxes in master-frame pixels.

    The tensor is [1, 4+nc, anchors], channel-major, with no objectness channel
    (unlike YOLOv5). Fully vectorised: the naive per-anchor loop over ~8400
    anchors costs more in Python than the forward pass itself and would show up
    as a language penalty that is really an implementation choice.
    """
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < 5:            # some exports emit [anchors, 4+nc]
        raw = raw.T

    scores = raw[4:]
    class_ids = np.argmax(scores, axis=0)
    confidences = scores[class_ids, np.arange(scores.shape[1])]
    keep = confidences >= conf_threshold
    if not np.any(keep):
        return [], [], []

    cx, cy, bw, bh = raw[0, keep], raw[1, keep], raw[2, keep], raw[3, keep]
    ox, oy = origin
    x = (cx - bw / 2.0 - pad_x) / scale + ox
    y = (cy - bh / 2.0 - pad_y) / scale + oy
    w = bw / scale
    h = bh / scale

    boxes = np.stack([x, y, w, h], axis=1).round().astype(int)
    return (boxes.tolist(),
            confidences[keep].astype(float).tolist(),
            class_ids[keep].astype(int).tolist())


def tile_nms(boxes, scores, class_ids, cfg: RunConfig):
    """Per-tile suppression before boxes are pooled onto the master frame.

    Doing it here as well as globally is what the C++ path does, so it stays in
    both. It also keeps the cross-tile merge cheap on a busy board.
    """
    if not boxes:
        return [], [], []
    keep = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids,
                                   cfg.conf_threshold, cfg.nms_threshold)
    idx = np.asarray(keep).flatten().astype(int).tolist()
    return ([boxes[i] for i in idx], [scores[i] for i in idx],
            [class_ids[i] for i in idx])


# =============================================================================
# Base
# =============================================================================

class InferenceEngine(ABC):
    name = "base"
    label = "Base"
    is_reference = False        # True = not a real measurement

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.monitor = HostMonitor()
        self.backend_desc = "unknown"
        self.device_desc = "unknown"

    @abstractmethod
    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...

    def _finish(self, timer: StageTimer, t_start: float, origins, dets,
                frame_id: int) -> FrameResult:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics: Dict[str, float] = {
            "frame_id": frame_id,
            "engine": self.name,
            "tiles": len(origins),
            "total_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 0 else 0.0,
            "detections": len(dets),
        }
        metrics.update(timer.finish(total_ms))
        metrics.update(self.monitor.sample())
        return FrameResult(dets, metrics)


# =============================================================================
# Simulation - ground-truth driven
# =============================================================================

class SimulationEngine(InferenceEngine):
    """Stands in for a trained model so the whole application runs anywhere.

    It is NOT a measurement and is labelled as such everywhere it surfaces. What
    it does give is a faithful exercise of the full data path - slicing, box
    remapping, NMS, annotation, reporting, PASS/FAIL - and, because it draws
    from the synthetic board's ground truth, plausible detections with a
    controllable miss and false-alarm rate. That is what makes precision/recall
    plumbing testable without weights on the machine.
    """

    name = "sim"
    label = "Simulation (no model)"
    is_reference = True

    def __init__(self, cfg: RunConfig, recall: float = 0.86,
                 false_alarms: float = 0.6):
        super().__init__(cfg)
        self.backend_desc = "Synthetic detections from ground truth - not a measurement"
        self.device_desc = "n/a"
        self._rng = np.random.default_rng(1337)
        self._recall = recall
        self._false_alarms = false_alarms
        self.truth: List[Detection] = []

    def set_truth(self, truth: Sequence[Detection]) -> None:
        """Called by the frame source. Without it the engine reports nothing."""
        self.truth = list(truth)

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        cfg = self.cfg
        self.monitor.reset()
        timer = StageTimer()
        t_start = time.perf_counter()

        t0 = timer.mark()
        origins = slice_origins(frame.shape, cfg)
        timer.since("slice_ms", t0)

        boxes, scores, ids, tiles = [], [], [], []
        for gt in self.truth:
            if self._rng.random() > self._recall:
                continue                                   # simulated miss
            jx = int(self._rng.normal(0, max(2, gt.w * 0.05)))
            jy = int(self._rng.normal(0, max(2, gt.h * 0.05)))
            js = float(self._rng.uniform(0.9, 1.1))
            boxes.append([gt.x + jx, gt.y + jy,
                          max(4, int(gt.w * js)), max(4, int(gt.h * js))])
            scores.append(float(self._rng.uniform(cfg.conf_threshold + 0.15, 0.97)))
            ids.append(gt.class_id)
            tiles.append(0)

        h, w = frame.shape[:2]
        for _ in range(self._rng.poisson(self._false_alarms)):
            bw = int(self._rng.integers(40, 160))
            bh = int(self._rng.integers(40, 160))
            boxes.append([int(self._rng.integers(0, max(1, w - bw))),
                          int(self._rng.integers(0, max(1, h - bh))), bw, bh])
            scores.append(float(self._rng.uniform(cfg.conf_threshold, 0.55)))
            ids.append(int(self._rng.integers(0, len(cfg.class_names))))
            tiles.append(0)

        # Stand in for the forward pass so the UI paces like the real thing,
        # scaled by tile count so changing the grid still changes the cadence.
        time.sleep(0.004 * max(1, len(origins)))
        timer.add("inference_ms", 0.004 * max(1, len(origins)))

        t0 = timer.mark()
        dets = global_nms(boxes, scores, ids, tiles, cfg)
        timer.since("merge_ms", t0)

        return self._finish(timer, t_start, origins, dets, frame_id)


# =============================================================================
# Test Path A - Python
# =============================================================================

class PythonEngine(InferenceEngine):
    """Ultralytics' high-level predict() call, per tile.

    This is what the "rapid prototyping ecosystem" claim actually looks like in
    code, so it is the honest representative of Path A. Its cost includes
    Ultralytics' own preprocessing, Results-object construction and internal
    NMS, none of which can be separated from the outside - which is precisely
    why PythonLeanEngine exists alongside it.
    """

    name = "python"
    label = "Path A - Python (Ultralytics API)"

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        from ultralytics import YOLO
        import torch

        self._torch = torch
        self.device = "cuda:0" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(cfg.model_pt)
        self.model.to(self.device)
        # half=True is passed to predict() rather than calling .half() on the
        # module: halving the weights alone leaves Ultralytics' preprocessing
        # emitting float32, and the first matmul then raises
        # "expected mat1 and mat2 to have the same dtype".
        self._half = bool(cfg.use_fp16 and self.device.startswith("cuda"))
        self.device_desc = (torch.cuda.get_device_name(0)
                            if self.device.startswith("cuda") else "CPU")
        self.backend_desc = (f"PyTorch {torch.__version__} / {self.device}"
                             f"{' FP16' if self._half else ' FP32'}")

    def warmup(self) -> None:
        dummy = np.full((self.cfg.tile_height, self.cfg.tile_width, 3), 114, np.uint8)
        for _ in range(self.cfg.warmup_runs):
            self._predict(dummy)
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def _predict(self, tile):
        return self.model.predict(tile, imgsz=self.cfg.input_size,
                                  conf=self.cfg.conf_threshold,
                                  iou=self.cfg.nms_threshold,
                                  device=self.device, half=self._half,
                                  verbose=False)[0]

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        cfg = self.cfg
        self.monitor.reset()
        timer = StageTimer()
        t_start = time.perf_counter()

        t0 = timer.mark()
        origins = slice_origins(frame.shape, cfg)
        t0 = timer.since("slice_ms", t0)

        boxes, scores, ids, tiles = [], [], [], []
        cuda = self.device.startswith("cuda")

        for tid, (ox, oy) in enumerate(origins):
            t0 = timer.mark()
            # The numpy slice is a view; Ultralytics copies during preprocess.
            # Making the copy explicit puts it in the marshal bucket where it
            # belongs instead of hiding inside the framework's timing.
            tile = np.ascontiguousarray(
                frame[oy:oy + cfg.tile_height, ox:ox + cfg.tile_width])
            t0 = timer.since("marshal_ms", t0)

            res = self._predict(tile)
            # CUDA is asynchronous. Without the sync we would time the kernel
            # launch, not the work, and Python would look impossibly fast.
            if cuda:
                self._torch.cuda.synchronize()
            t0 = timer.since("inference_ms", t0)

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
            timer.since("decode_ms", t0)

        t0 = timer.mark()
        dets = global_nms(boxes, scores, ids, tiles, cfg)
        timer.since("merge_ms", t0)

        return self._finish(timer, t_start, origins, dets, frame_id)


class PythonLeanEngine(InferenceEngine):
    """Path A ablation: the same weights, driven the way the C++ path drives them.

    Letterbox, tensor construction, decode and NMS are the shared functions in
    aoi.pipeline and this module - byte-for-byte the rule the C++ engine
    implements. Only the runtime (PyTorch vs ONNX Runtime) and the host language
    still differ.

    Reporting this alongside `python` is what turns "C++ is N times faster" into
    a decomposition: the gap between `python` and `python-lean` is framework
    overhead, and the gap between `python-lean` and `cpp` is what is left for
    runtime and language. Without it the headline ratio cannot be attributed.
    """

    name = "python-lean"
    label = "Path A' - Python (lean, mirrors C++)"

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        from ultralytics import YOLO
        import torch

        self._torch = torch
        self.device = torch.device(
            "cuda:0" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
        wrapper = YOLO(cfg.model_pt)
        net = wrapper.model.float().eval().to(self.device)
        try:
            net = net.fuse()             # conv+bn folding, as predict() does
        except Exception:                                       # noqa: BLE001
            pass
        self._half = bool(cfg.use_fp16 and self.device.type == "cuda")
        if self._half:
            net = net.half()
        self.net = net
        self._dtype = torch.float16 if self._half else torch.float32
        self.device_desc = (torch.cuda.get_device_name(0)
                            if self.device.type == "cuda" else "CPU")
        self.backend_desc = (f"PyTorch {torch.__version__} raw module / "
                             f"{self.device.type}{' FP16' if self._half else ' FP32'}")

    def warmup(self) -> None:
        dummy = np.full((self.cfg.tile_height, self.cfg.tile_width, 3), 114, np.uint8)
        for _ in range(self.cfg.warmup_runs):
            self._forward(dummy)
        if self.device.type == "cuda":
            self._torch.cuda.synchronize()

    def _forward(self, tile: np.ndarray):
        cfg = self.cfg
        padded, scale, pad_x, pad_y = letterbox(tile, (cfg.input_size, cfg.input_size))
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, padded.shape[1::-1],
                                     swapRB=True, crop=False)
        tensor = self._torch.from_numpy(blob).to(self.device, self._dtype,
                                                 non_blocking=True)
        with self._torch.inference_mode():
            out = self.net(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out, scale, pad_x, pad_y

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        cfg = self.cfg
        self.monitor.reset()
        timer = StageTimer()
        t_start = time.perf_counter()
        torch = self._torch
        cuda = self.device.type == "cuda"

        t0 = timer.mark()
        origins = slice_origins(frame.shape, cfg)
        t0 = timer.since("slice_ms", t0)

        boxes, scores, ids, tiles = [], [], [], []

        for tid, (ox, oy) in enumerate(origins):
            t0 = timer.mark()
            tile = np.ascontiguousarray(
                frame[oy:oy + cfg.tile_height, ox:ox + cfg.tile_width])
            t0 = timer.since("marshal_ms", t0)

            padded, scale, pad_x, pad_y = letterbox(
                tile, (cfg.input_size, cfg.input_size))
            blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, padded.shape[1::-1],
                                         swapRB=True, crop=False)
            tensor = torch.from_numpy(blob).to(self.device, self._dtype)
            if cuda:
                torch.cuda.synchronize()
            t0 = timer.since("preprocess_ms", t0)

            with torch.inference_mode():
                out = self.net(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if cuda:
                torch.cuda.synchronize()
            t0 = timer.since("inference_ms", t0)

            raw = out.detach().float().cpu().numpy()
            tb, ts, tc = decode_yolov8(raw, cfg.conf_threshold,
                                       scale, pad_x, pad_y, (ox, oy))
            t0 = timer.since("decode_ms", t0)

            tb, ts, tc = tile_nms(tb, ts, tc, cfg)
            boxes += tb
            scores += ts
            ids += tc
            tiles += [tid] * len(tb)
            timer.since("nms_ms", t0)

        t0 = timer.mark()
        dets = global_nms(boxes, scores, ids, tiles, cfg)
        timer.since("merge_ms", t0)

        return self._finish(timer, t_start, origins, dets, frame_id)

    def close(self) -> None:
        try:
            if self.device.type == "cuda":
                self._torch.cuda.empty_cache()
        except Exception:                                       # noqa: BLE001
            pass


# =============================================================================
# Test Path B - C++
# =============================================================================

class CppEngine(InferenceEngine):
    """Compiled C++17 engine, reached through pybind11.

    The numpy buffer is wrapped as a cv::Mat rather than copied, and the GIL is
    released for the duration of the frame, so the GUI keeps repainting while
    inference runs. The extension returns the same stage names the Python paths
    produce, so the report can put them in one table.
    """

    name = "cpp"
    label = "Path B - C++ (ONNX Runtime)"

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        bootstrap()
        try:
            import aoi_engine_cpp as native
        except ImportError as exc:
            raise ImportError(
                "aoi_engine_cpp is not built. From cpp_backend/:\n"
                "  cmake -B build -S . -A x64 -DOpenCV_DIR=... "
                "-DONNXRUNTIME_ROOT=... -DUSE_NVML=ON\n"
                "  cmake --build build --config Release\n"
                f"(underlying error: {exc})") from exc

        self._native = native
        c = native.Config()
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

        self.engine = native.AoiEngine(c)
        self.engine.set_class_names(list(cfg.class_names))
        self.backend_desc = self.engine.backend_description
        self.device_desc = self.engine.device_name

    def warmup(self) -> None:
        self.engine.warmup()

    def run_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        self.monitor.reset()
        dets_native, fm = self.engine.run_frame(frame, frame_id)

        dets = [Detection(d.x, d.y, d.w, d.h, d.confidence, d.class_id, d.tile_id)
                for d in dets_native]
        metrics: Dict[str, float] = {
            "frame_id": fm.frame_id, "engine": self.name, "tiles": fm.tiles,
            "total_ms": fm.total_ms, "fps": fm.fps, "detections": fm.detections,
            "cpu_percent": fm.cpu_percent, "gpu_percent": fm.gpu_percent,
            "gpu_mem_mb": fm.gpu_mem_mb, "rss_mb": fm.rss_mb,
            "peak_rss_mb": fm.peak_rss_mb,
        }
        for stage in STAGES:
            metrics[stage] = float(getattr(fm, stage, 0.0))

        # The extension only reports GPU counters when compiled with -DUSE_NVML,
        # which needs nvml.h from the CUDA Toolkit. nvml.dll itself ships with
        # the driver and pynvml talks to it directly, so fill the sentinel from
        # the same handle Path A uses - one instrument for both paths is a
        # cleaner fairness claim than one path on the C API and one on pynvml.
        #
        # Caveat for the write-up: nvmlDeviceGetUtilizationRates averages over
        # the driver's sampling window (order of a second), not over this frame.
        # Treat gpu_percent as a run-level figure, never a per-frame one.
        if metrics["gpu_percent"] < 0 or metrics["gpu_mem_mb"] < 0:
            gpu = self.monitor.gpu()
            for k in ("gpu_percent", "gpu_mem_mb"):
                if metrics[k] < 0:
                    metrics[k] = gpu[k]

        return FrameResult(dets, metrics)

    def close(self) -> None:
        self.engine = None


# =============================================================================
# Registry
# =============================================================================

REGISTRY = {
    "sim": SimulationEngine,
    "python": PythonEngine,
    "python-lean": PythonLeanEngine,
    "cpp": CppEngine,
}

# The pair the headline comparison is drawn from. python-lean is an ablation
# that is measured when available but never replaces Path A in the headline.
TEST_PATHS = ("python", "cpp")
ENGINE_LABELS = {
    "sim": "Simulation (no model)",
    "python": "Path A - Python",
    "python-lean": "Path A' - Python lean",
    "cpp": "Path B - C++",
}


def create_engine(name: str, cfg: RunConfig, warmup: bool = True) -> InferenceEngine:
    key = name.lower().strip()
    if key not in REGISTRY:
        raise ValueError(f"Unknown engine '{name}'. Options: {sorted(REGISTRY)}")
    engine = REGISTRY[key](cfg)
    if warmup:
        engine.warmup()
    return engine


def available_engines(cfg: Optional[RunConfig] = None) -> Dict[str, bool]:
    """Which backends can load right now. Drives the greyed-out combo entries so
    the operator sees why a path is unavailable instead of hitting a crash."""
    cfg = cfg or RunConfig()
    status = {"sim": True}

    try:
        import ultralytics  # noqa: F401
        ok = Path(cfg.model_pt).exists()
    except ImportError:
        ok = False
    status["python"] = ok
    status["python-lean"] = ok

    try:
        bootstrap()
        import aoi_engine_cpp  # noqa: F401
        status["cpp"] = Path(cfg.model_onnx).exists()
    except ImportError:
        status["cpp"] = False

    return status


def unavailable_reason(name: str, cfg: RunConfig) -> str:
    if name in ("python", "python-lean"):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return "ultralytics not installed"
        if not Path(cfg.model_pt).exists():
            return "missing .pt weights"
    if name == "cpp":
        try:
            bootstrap()
            import aoi_engine_cpp  # noqa: F401
        except ImportError:
            return "extension not built"
        if not Path(cfg.model_onnx).exists():
            return "missing .onnx weights"
    return "unavailable"
