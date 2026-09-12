"""
Configuration and platform bootstrap.

Two responsibilities, deliberately together because both must be settled before
anything heavyweight is imported:

  * RunConfig  - the single parameter set both test paths read. One object, so
                 a threshold cannot drift between them and be misread as a
                 language effect.
  * bootstrap  - Windows DLL ordering and CUDA discovery. Must run before PyQt5
                 and before the native extension is imported.

Stdlib only above the RunConfig section, so bootstrap() is safe to call as the
very first statement of any entry point.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# Platform bootstrap  (stdlib only - runs before PyQt5 / cv2 / torch)
# =============================================================================

# opencv_world's CRT dependencies, per `dumpbin /dependents`.
_CRT_DLLS = ("VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "MSVCP140.dll", "CONCRT140.dll")

_state: Dict[str, Any] = {"crt": None, "cuda": None}


def _preload_system_crt(verbose: bool = False) -> List[str]:
    """Pin System32's VC++ runtime into the process before Qt can load its own.

    PyQt5 registers its bundled Qt5/bin folder, which ships a VS2019-era
    MSVCP140.dll. Windows resolves DLLs by base name, so once that copy is
    resident every later request reuses it - including opencv_world's, which
    was built against a much newer CRT and faults during static init. There is
    no search-order fix after the fact: the wrong copy is already in memory.
    Loading System32's copies first, by absolute path, wins the race. Newer
    CRTs are backward compatible, so Qt on the new runtime is the supported
    direction; the reverse is what crashes.
    """
    if _state["crt"] is not None:
        return _state["crt"]
    loaded: List[str] = []
    _state["crt"] = loaded
    if os.name != "nt":
        return loaded

    import ctypes

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in _CRT_DLLS:
        path = os.path.join(system32, name)
        if not os.path.isfile(path):
            continue
        try:
            ctypes.WinDLL(path)
            loaded.append(name)
            if verbose:
                print(f"[bootstrap] preloaded {path}")
        except OSError as exc:
            if verbose:
                print(f"[bootstrap] could not preload {path}: {exc}")
    return loaded


def _cuda_candidate_dirs() -> List[str]:
    """Folders that may hold cudart / cublas / cudnn, best first.

    Toolkit installs come first; pip wheels (torch/lib, nvidia/*/bin) are the
    fallback that makes the GPU path work on a machine with no nvcc.
    """
    out: List[str] = []

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        out.append(os.path.join(cuda_path, "bin"))

    root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(root):
        out += [os.path.join(root, n, "bin") for n in sorted(os.listdir(root), reverse=True)]

    cudnn_root = r"C:\Program Files\NVIDIA\CUDNN"
    if os.path.isdir(cudnn_root):
        for name in sorted(os.listdir(cudnn_root), reverse=True):
            binp = os.path.join(cudnn_root, name, "bin")
            if os.path.isdir(binp):
                out.append(binp)
                # cuDNN 9 nests one level deeper by CUDA major, e.g. bin/12.6
                out += [os.path.join(binp, s) for s in sorted(os.listdir(binp), reverse=True)]

    from importlib.util import find_spec

    for pkg in ("torch", "nvidia"):
        try:
            spec = find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        locations = list(spec.submodule_search_locations or [])
        if not locations and spec.origin:
            locations = [os.path.dirname(spec.origin)]
        for base in locations:
            if pkg == "torch":
                out.append(os.path.join(base, "lib"))
            elif os.path.isdir(base):
                out += [os.path.join(base, c, "bin") for c in sorted(os.listdir(base))]
    return out


def _preload_cuda(verbose: bool = False) -> List[str]:
    """Register CUDA/cuDNN folders so the ORT CUDA provider can load.

    Python 3.8+ ignores PATH when resolving a native extension's dependencies -
    only directories registered through os.add_dll_directory() are searched. Miss
    this and the import fails with a bare "error 126", or the session silently
    falls back to CPU and the benchmark compares a GPU path against a CPU one.
    """
    if _state["cuda"] is not None:
        return _state["cuda"]
    dirs: List[str] = []
    _state["cuda"] = dirs
    if os.name != "nt":
        return dirs

    seen = set()
    for d in _cuda_candidate_dirs():
        if not d or not os.path.isdir(d):
            continue
        key = os.path.normcase(os.path.abspath(d))
        if key in seen:
            continue
        seen.add(key)
        try:
            os.add_dll_directory(d)
            dirs.append(d)
            if verbose:
                print(f"[bootstrap] cuda dir: {d}")
        except OSError:
            pass
    return dirs


def bootstrap(verbose: bool = False) -> Dict[str, List[str]]:
    """Idempotent. Call as the first statement of every entry point."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # The compiled extension is dropped beside this package by CMake.
    for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "aoi")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return {"crt": _preload_system_crt(verbose), "cuda": _preload_cuda(verbose)}


# =============================================================================
# RunConfig
# =============================================================================

CLASS_NAMES: Tuple[str, ...] = (
    "mouse_bite", "spur", "missing_hole",
    "short", "open_circuit", "spurious_copper",
)


@dataclass
class RunConfig:
    """Every parameter both test paths read. Nothing is duplicated per engine.

    A field that differs between Path A and Path B would show up in the results
    as a language effect. Keeping one object is the cheapest possible guard.
    """

    # --- model artefacts (must come from the same training run) --------------
    model_pt: str = str(PROJECT_ROOT / "datasets/weights/yolov8_pcb_4k.pt")
    model_onnx: str = str(PROJECT_ROOT / "datasets/weights/yolov8_pcb_4k.onnx")
    class_names: Tuple[str, ...] = CLASS_NAMES

    # --- inference -----------------------------------------------------------
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

    # --- capture -------------------------------------------------------------
    camera_index: int = 0
    camera_backend: str = "auto"          # auto | dshow | msmf   (Windows only)
    force_synthetic: bool = False
    # Two roles, and only one of them is binding:
    #   * the size the synthetic board is rendered at - binding;
    #   * the capture mode REQUESTED from the camera - a preference. The driver
    #     clamps to a mode it supports and the pipeline uses whatever it gets.
    #     Frames are never rescaled to these numbers: upscaling a 1080p sensor to
    #     4K invents no detail, multiplies the tile count and would make the
    #     benchmark time interpolated pixels. Tile geometry is derived from the
    #     frame's own shape everywhere downstream.
    frame_width: int = 3840
    frame_height: int = 2160
    target_fps: int = 30

    # --- synthetic board / ground truth --------------------------------------
    # The synthetic feed plants defects at known coordinates. That is what makes
    # precision/recall computable with no annotated dataset on hand, and it is
    # why the demo shows detections instead of an empty PASS screen.
    synth_defects: int = 7
    synth_seed: int = 2601

    # --- study ---------------------------------------------------------------
    trial_frames: int = 30                # measured frames per trial
    trials: int = 3                       # independent trials, for between-run variance
    discard_frames: int = 3               # dropped after warmup, per trial

    # --- output --------------------------------------------------------------
    report_dir: str = str(PROJECT_ROOT / "reports")

    # -----------------------------------------------------------------------
    @property
    def frame_size(self) -> Tuple[int, int]:
        return self.frame_width, self.frame_height

    def variant(self, **kw) -> "RunConfig":
        """A copy with fields overridden - how a sweep varies one factor."""
        return replace(self, **kw)

    def fingerprint(self) -> str:
        """Short digest of everything that could change a measurement.

        Printed on every report so two result sets can be proven comparable.
        """
        import hashlib

        keys = ("input_size", "conf_threshold", "nms_threshold", "use_cuda",
                "use_fp16", "tile_width", "tile_height", "overlap_ratio",
                "frame_width", "frame_height", "warmup_runs")
        blob = "|".join(f"{k}={getattr(self, k)}" for k in keys)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        data["class_names"] = tuple(data.get("class_names", cls.class_names))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# Named tile geometries used by the sweep and the GUI recipe selector.
# 1920x1080 on a 3840x2160 frame is the pathological case: 3840 is exactly
# 2x1920, so any non-zero overlap jumps straight from 2 columns to 3 and the
# real overlap lands at 50%, not the 20% requested - 9 inferences per frame
# instead of 4. The alternatives below hit their requested overlap honestly.
# Tile sizes are solved for the grid they claim: for n tiles at overlap r on an
# axis of length W, t = W / (n - (n-1)r). Guessing round numbers is what produced
# the 1920x1080 case in the first place.
TILE_PRESETS: Dict[str, Tuple[int, int, float]] = {
    "2x2 (no overlap)":      (1920, 1080, 0.00),
    "3x3 (forced 50%)":      (1920, 1080, 0.20),
    "3x3 (true 25%)":        (1536,  864, 0.25),
    "4x4 (true 20%)":        (1130,  636, 0.20),
    "6x6 (true 21%)":        ( 776,  436, 0.20),
    "1x1 (no slicing)":      (3840, 2160, 0.00),
}


# =============================================================================
# Environment report  (replaces the old selfcheck / ort_probe / diagnose_dll)
# =============================================================================

def environment_report(cfg: RunConfig | None = None) -> Dict[str, Any]:
    """Everything a reader of the results needs to reproduce them."""
    cfg = cfg or RunConfig()
    rep: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or "unknown",
        "config_fingerprint": cfg.fingerprint(),
    }

    try:
        import cv2
        rep["opencv"] = cv2.__version__
    except Exception as exc:                                    # noqa: BLE001
        rep["opencv"] = f"missing ({exc})"

    try:
        import numpy
        rep["numpy"] = numpy.__version__
    except Exception:                                           # noqa: BLE001
        rep["numpy"] = "missing"

    try:
        import psutil
        rep["cpu_cores"] = psutil.cpu_count(logical=True)
        rep["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:                                           # noqa: BLE001
        rep["cpu_cores"] = os.cpu_count()

    try:
        import torch
        rep["torch"] = torch.__version__
        rep["torch_cuda"] = torch.version.cuda or "cpu build"
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rep["gpu"] = torch.cuda.get_device_name(0)
    except Exception:                                           # noqa: BLE001
        rep["torch"] = "not installed"

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        rep["gpu"] = pynvml.nvmlDeviceGetName(h)
        if isinstance(rep["gpu"], bytes):
            rep["gpu"] = rep["gpu"].decode()
        rep["driver"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(rep["driver"], bytes):
            rep["driver"] = rep["driver"].decode()
        rep["nvml"] = "ok"
    except Exception:                                           # noqa: BLE001
        rep["nvml"] = "unavailable"

    bootstrap()
    try:
        import aoi_engine_cpp
        rep["cpp_extension"] = getattr(aoi_engine_cpp, "__version__", "built")
        rep["cpp_providers"] = ", ".join(aoi_engine_cpp.available_providers())
    except Exception as exc:                                    # noqa: BLE001
        rep["cpp_extension"] = f"not built ({type(exc).__name__})"

    rep["weights_pt"] = "present" if Path(cfg.model_pt).exists() else "MISSING"
    rep["weights_onnx"] = "present" if Path(cfg.model_onnx).exists() else "MISSING"
    rep["cuda_dirs_registered"] = len(_state.get("cuda") or [])
    return rep
