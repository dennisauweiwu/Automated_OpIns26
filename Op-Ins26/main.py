#!/usr/bin/env python3
"""
AOI Comparative Benchmark - single entry point.

    python main.py                       launch the GUI (simulation engine)
    python main.py --engine cpp          launch with Test Path B selected
    python main.py bench                 headless study -> workbook + figures
    python main.py bench --sweep-all     add the tile and resolution sweeps
    python main.py verify                assert the parity + statistics invariants
    python main.py doctor                environment, cameras, SAHI grid check
    python main.py board out.png         render a synthetic 4K board + labels
    python main.py export-onnx           .pt -> .onnx with a hash manifest

Everything the seven scripts under the old tools/ directory did now lives here
as a subcommand, so there is one place to look and one import path to keep
working.
"""

from __future__ import annotations

# MUST be first: Windows DLL ordering and CUDA discovery both have to be settled
# before PyQt5, cv2 or the native extension are imported. See aoi/config.py.
from aoi.config import PROJECT_ROOT, RunConfig, TILE_PRESETS, bootstrap, environment_report

bootstrap()

import argparse                                                 # noqa: E402
import faulthandler                                             # noqa: E402
import os                                                       # noqa: E402
import sys                                                      # noqa: E402
import time                                                     # noqa: E402
from pathlib import Path                                        # noqa: E402


# =============================================================================
# Shared argument plumbing
# =============================================================================

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--tile", choices=list(TILE_PRESETS),
                   help="SAHI grid preset (overrides the config file)")
    p.add_argument("--conf", type=float, help="confidence threshold")
    p.add_argument("--no-cuda", action="store_true", help="force CPU on every path")
    p.add_argument("--defects", type=int, help="planted defects per synthetic board")


def build_config(args) -> RunConfig:
    cfg = RunConfig.load(args.config) if getattr(args, "config", None) else RunConfig()
    if getattr(args, "tile", None):
        tw, th, ov = TILE_PRESETS[args.tile]
        cfg = cfg.variant(tile_width=tw, tile_height=th, overlap_ratio=ov)
    if getattr(args, "conf", None) is not None:
        cfg = cfg.variant(conf_threshold=args.conf)
    if getattr(args, "no_cuda", False):
        cfg = cfg.variant(use_cuda=False, use_fp16=False)
    if getattr(args, "defects", None):
        cfg = cfg.variant(synth_defects=args.defects)
    if getattr(args, "camera", None) is not None:
        cfg = cfg.variant(camera_index=args.camera)
    if getattr(args, "camera_backend", None):
        cfg = cfg.variant(camera_backend=args.camera_backend)
    if getattr(args, "no_camera", False):
        cfg = cfg.variant(force_synthetic=True)
    if getattr(args, "trials", None):
        cfg = cfg.variant(trials=args.trials)
    if getattr(args, "frames", None):
        cfg = cfg.variant(trial_frames=args.frames)
    return cfg


# =============================================================================
# gui
# =============================================================================

def cmd_gui(args) -> int:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from aoi.ui.window import MainWindow, load_theme

    cfg = build_config(args)
    if args.debug_shutdown:
        os.environ["AOI_DEBUG_SHUTDOWN"] = "1"
    # Ctrl-\ or the shutdown watchdog can then dump every thread's stack.
    faulthandler.enable()
    Path(cfg.report_dir).mkdir(parents=True, exist_ok=True)

    # Must be set before QApplication exists, or 4K frames render blurry on a
    # scaled Windows display.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AOI Comparative Benchmark")
    load_theme(app, args.theme)

    window = MainWindow(cfg, default_engine=args.engine, theme=args.theme)
    window.show()
    rc = app.exec_()

    # closeEvent normally hard-exits before this. If we do reach here, flush and
    # bypass interpreter teardown: native threads from OpenCV and CUDA can
    # otherwise keep the process alive with no window on screen.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("AOI_NO_HARD_EXIT") == "1":
        return rc
    os._exit(rc)


# =============================================================================
# bench
# =============================================================================

def cmd_bench(args) -> int:
    from aoi.analysis import run_study, sweep_resolution, sweep_tiles
    from aoi.engines import available_engines, unavailable_reason
    from aoi.report import figures_available, write_figures, write_workbook

    cfg = build_config(args)
    avail = available_engines(cfg)
    engines = args.engines or [e for e in ("python", "python-lean", "cpp")
                               if avail.get(e)]
    if not engines:
        print("No real backend is available:")
        for e in ("python", "cpp"):
            print(f"  {e}: {unavailable_reason(e, cfg)}")
        print("\nFalling back to the simulation engine so the harness is still "
              "exercised. Nothing it prints is a measurement.")
        engines = ["sim"]

    _banner(cfg, engines)

    last = [0.0]

    def progress(msg: str, frac: float):
        # Throttled: printing every frame would cost more than it reports on a
        # fast path, and would itself perturb the measurement.
        if time.monotonic() - last[0] < 0.25 and frac < 1.0:
            return
        last[0] = time.monotonic()
        width = 34
        filled = int(width * max(0.0, min(1.0, frac)))
        print(f"\r  [{'#' * filled}{'.' * (width - filled)}] {msg[:52]:<52}",
              end="", flush=True)

    study = run_study(cfg, engines, progress=progress)
    print()

    if args.sweep_tiles or args.sweep_all:
        print("\n  Sweeping SAHI grid...")
        study.sweeps["tiles"] = sweep_tiles(
            cfg, engines, TILE_PRESETS,
            frames_per_point=max(6, cfg.trial_frames // 2), progress=progress)
        print()
    if args.sweep_resolution or args.sweep_all:
        print("\n  Sweeping input resolution...")
        study.sweeps["resolution"] = sweep_resolution(
            cfg, engines, frames_per_point=max(6, cfg.trial_frames // 3),
            progress=progress)
        print()

    _print_matrix(study)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or cfg.report_dir)
    book = write_workbook(study, out_dir / f"study_{stamp}.xlsx")
    print(f"\n  Workbook : {book}")
    if figures_available() and not args.no_figures:
        figs = write_figures(study, out_dir / f"figures_{stamp}")
        print(f"  Figures  : {len(figs)} written to {out_dir / f'figures_{stamp}'}")
    elif not figures_available():
        print("  Figures  : skipped (pip install matplotlib)")
    return 0


def _banner(cfg: RunConfig, engines) -> None:
    from aoi.pipeline import effective_overlap, grid_shape

    cols, rows = grid_shape((cfg.frame_height, cfg.frame_width), cfg)
    ov = effective_overlap(cfg.frame_width, cfg.tile_width, cfg.overlap_ratio)
    print("=" * 74)
    print("  AOI Comparative Study - headless")
    print("=" * 74)
    print(f"  started    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  input      {cfg.frame_width}x{cfg.frame_height} synthetic board, "
          f"{cfg.synth_defects} planted defects (seed {cfg.synth_seed})")
    print(f"  SAHI grid  {cols}x{rows} = {cols * rows} tiles of "
          f"{cfg.tile_width}x{cfg.tile_height}")
    print(f"             requested overlap {cfg.overlap_ratio:.0%}, "
          f"effective {ov:.0%}"
          + ("   <- the grid cannot honour the request at this tile size"
             if ov > cfg.overlap_ratio + 0.1 else ""))
    print(f"  protocol   {cfg.trials} trials x {cfg.trial_frames} frames, "
          f"{cfg.warmup_runs} warmup + {cfg.discard_frames} settling discarded")
    print(f"  thresholds conf {cfg.conf_threshold}, NMS IoU {cfg.nms_threshold}, "
          f"CUDA {cfg.use_cuda}")
    print(f"  paths      {', '.join(engines)}")
    print(f"  config     {cfg.fingerprint()}")
    print("-" * 74)


def _print_matrix(study) -> None:
    from aoi.pipeline import STAGES

    engines = [e for e in study.engines() if study.latencies(e)]
    if not engines:
        print("\n  No measurements were produced.")
        return

    w = 22
    print("\n" + "=" * (18 + w * len(engines)))
    print("  PERFORMANCE MATRIX")
    print("=" * (18 + w * len(engines)))
    print(f"  {'Metric':<16}" + "".join(f"{e:<{w}}" for e in engines))
    print("  " + "-" * (16 + w * len(engines)))

    def line(label, fn):
        print(f"  {label:<16}" + "".join(f"{fn(e)[:w - 1]:<{w}}" for e in engines))

    line("Backend", lambda e: study.backend(e))
    line("Frames", lambda e: str(study.summary(e).n))
    line("Mean", lambda e: f"{study.summary(e).mean:.2f} ms")
    line("95% CI", lambda e: f"[{study.summary(e).ci95[0]:.1f},"
                             f"{study.summary(e).ci95[1]:.1f}]")
    line("Median", lambda e: f"{study.summary(e).median:.2f} ms")
    line("p95", lambda e: f"{study.summary(e).p95:.2f} ms")
    line("CV", lambda e: f"{study.summary(e).cv * 100:.1f} %")
    line("FPS", lambda e: f"{1000 / study.summary(e).mean:.2f}"
                          if study.summary(e).mean else "-")
    print("  " + "-" * (16 + w * len(engines)))
    for stage in STAGES:
        vals = {e: study.stage_means(e)[stage] for e in engines}
        if all(v < 0.005 for v in vals.values()):
            continue
        line(stage.replace("_ms", ""), lambda e, v=vals: f"{v[e]:.3f} ms")
    print("  " + "-" * (16 + w * len(engines)))
    # An empty sample means the counter was unavailable (no psutil, no NVML),
    # which is a different statement from "measured zero" and must not be
    # printed as 0.0.
    def res(engine, key, fmt):
        vals = study.resource(engine, key)
        return fmt.format(_avg(vals)) if vals else "n/a"

    line("CPU", lambda e: res(e, "cpu_percent", "{:.1f} %"))
    line("GPU", lambda e: res(e, "gpu_percent", "{:.1f} %"))
    line("RSS", lambda e: res(e, "rss_mb", "{:.0f} MB"))

    acc_engines = [e for e in engines if study.accuracy(e)]
    if acc_engines:
        print("  " + "-" * (16 + w * len(engines)))
        for key in ("precision", "recall", "f1", "ap50"):
            line(key, lambda e, k=key: f"{study.accuracy(e).get(k, 0):.3f}")

    cmp_ = study.headline()
    if cmp_:
        print("\n  " + "=" * (16 + w * len(engines)))
        print(f"  {cmp_.verdict()}")
        attr = [r for r in study.stage_attribution() if abs(r["delta_ms"]) > 0.01]
        if attr:
            total = sum(r["delta_ms"] for r in attr)
            print(f"\n  Gap attribution ({total:+.1f} ms/frame):")
            for r in attr[:5]:
                print(f"    {r['stage']:<14} {r['delta_ms']:+8.3f} ms  "
                      f"{r['share'] * 100:5.1f}% of the gap")

    par = study.parity_summary()
    if par:
        flag = "" if par["agreement"] >= 0.9 else "   <- LOW, speed claim unsafe"
        print(f"\n  Detection parity: {par['agreement'] * 100:.1f}% agreement, "
              f"mean IoU {par['mean_iou']:.2f}{flag}")

    for n in study.notes:
        print(f"  note: {n}")


def _avg(vals) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# =============================================================================
# doctor
# =============================================================================

def cmd_doctor(args) -> int:
    from aoi.engines import available_engines, unavailable_reason
    from aoi.pipeline import effective_overlap, grid_shape
    from aoi.workers import list_cameras

    cfg = build_config(args)
    print("=" * 70)
    print("  Environment")
    print("=" * 70)
    for k, v in environment_report(cfg).items():
        print(f"  {k:<22} {v}")

    print("\n" + "=" * 70)
    print("  Test paths")
    print("=" * 70)
    avail = available_engines(cfg)
    for key in ("sim", "python", "python-lean", "cpp"):
        mark = "OK " if avail.get(key) else "-- "
        why = "" if avail.get(key) else f"   ({unavailable_reason(key, cfg)})"
        print(f"  {mark}{key:<14}{why}")

    print("\n" + "=" * 70)
    print("  SAHI grid at the configured frame size")
    print("=" * 70)
    for name, (tw, th, ov) in TILE_PRESETS.items():
        v = cfg.variant(tile_width=tw, tile_height=th, overlap_ratio=ov)
        cols, rows = grid_shape((cfg.frame_height, cfg.frame_width), v)
        ex = effective_overlap(cfg.frame_width, tw, ov)
        ey = effective_overlap(cfg.frame_height, th, ov)
        warn = "  <- effective overlap far above requested" if ex > ov + 0.1 else ""
        print(f"  {name:<20} {cols}x{rows} = {cols * rows:>2} tiles   "
              f"requested {ov:.0%}, effective {ex:.0%}/{ey:.0%}{warn}")

    if not args.no_cameras:
        print("\n" + "=" * 70)
        print(f"  Cameras (probing indices 0-{args.max_index - 1})")
        print("=" * 70)
        cams = list_cameras(max_index=args.max_index)
        if not cams:
            print("  none found - the synthetic board will be used")
            print("\n  If a camera you expect is missing:")
            print("    - close any other app holding it, including a running "
                  "instance of this GUI")
            print("    - raise the range with --max-index 10")
            print("    - check it appears under Imaging devices in Device Manager")
        else:
            # Grouped by index, because the same device often appears under one
            # backend and not the other - and which one it is decides the
            # --camera-backend flag that will actually work.
            for idx in sorted({int(c["index"]) for c in cams}):
                rows = [c for c in cams if c["index"] == idx]
                print(f"\n  index {idx}")
                for c in rows:
                    state = "frame OK" if c["readable"] else "OPENS BUT NO FRAME"
                    got = (f" (delivered {c['frame'][0]}x{c['frame'][1]})"
                           if c["frame"] and
                           c["frame"] != (c["width"], c["height"]) else "")
                    print(f"    {c['flag']:<6} {c['width']}x{c['height']}"
                          f" @{c['fps']:g}  {c['backend']:<10} {state}{got}")
                usable = [c for c in rows if c["readable"]]
                if usable:
                    best = usable[0]
                    print(f"    -> python main.py --camera {idx}"
                          f" --camera-backend {best['flag']}")
            missing = [f for f in ("dshow", "msmf")
                       if not any(c["flag"] == f for c in cams)]
            if missing and os.name == "nt":
                print(f"\n  Note: {', '.join(missing)} enumerated nothing. That is "
                      "normal - Windows exposes different device sets through "
                      "DirectShow and Media Foundation.")
    return 0


# =============================================================================
# board
# =============================================================================

def cmd_board(args) -> int:
    """Render a synthetic board plus its YOLO label file.

    Useful in its own right: the same generator that feeds the benchmark can
    emit an annotated training set, so a model can be fine-tuned on exactly the
    defect appearance the demo shows.
    """
    import cv2

    from aoi.synth import SyntheticBoard

    cfg = build_config(args)
    board = SyntheticBoard(cfg.frame_width, cfg.frame_height, cfg.synth_seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        frame, truth = board.frame(cfg.synth_defects, seed=cfg.synth_seed + i)
        path = out if args.count == 1 else out.with_name(f"{out.stem}_{i:03d}{out.suffix}")
        cv2.imwrite(str(path), frame)
        # YOLO format: class cx cy w h, all normalised - directly trainable.
        label = path.with_suffix(".txt")
        label.write_text("\n".join(
            f"{d.class_id} "
            f"{(d.x + d.w / 2) / cfg.frame_width:.6f} "
            f"{(d.y + d.h / 2) / cfg.frame_height:.6f} "
            f"{d.w / cfg.frame_width:.6f} {d.h / cfg.frame_height:.6f}"
            for d in truth), encoding="utf-8")
        print(f"  {path}  ({len(truth)} defects, labels -> {label.name})")

    if args.golden:
        golden = out.with_name(f"{out.stem}_golden{out.suffix}")
        cv2.imwrite(str(golden), board.clean)
        print(f"  {golden}  (defect-free reference)")
    return 0


# =============================================================================
# export-onnx
# =============================================================================

def cmd_export_onnx(args) -> int:
    """Export .pt -> .onnx and record both hashes.

    The manifest is the guard against the single most damaging silent error in
    this project: comparing a Python path running one training run against a C++
    path running another. If the hashes in the manifest do not match the files
    on disk at benchmark time, the comparison is invalid.
    """
    import hashlib
    import json

    cfg = build_config(args)
    src = Path(args.weights or cfg.model_pt)
    if not src.exists():
        print(f"  weights not found: {src}")
        return 1

    from ultralytics import YOLO

    print(f"  exporting {src.name} at imgsz={args.imgsz}, opset={args.opset}...")
    model = YOLO(str(src))
    produced = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                                 simplify=True, dynamic=False))
    target = Path(cfg.model_onnx)
    target.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != target.resolve():
        produced.replace(target)

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = target.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pt": str(src), "pt_sha256": sha(src),
        "onnx": str(target), "onnx_sha256": sha(target),
        "imgsz": args.imgsz, "opset": args.opset,
        "class_names": list(cfg.class_names),
    }, indent=2), encoding="utf-8")

    print(f"  wrote {target}")
    print(f"  wrote {manifest}")
    print("  Both paths must load artefacts whose hashes match this manifest, or "
          "weight drift will be reported as a language effect.")
    return 0



# =============================================================================
# verify
# =============================================================================

def cmd_verify(args) -> int:
    """Assert the invariants the comparison rests on, instead of asserting them
    in a comment.

    Four things can silently invalidate every number this project produces: the
    two paths slicing different pixels, the two paths padding differently, the
    two paths loading weights from different training runs, and the statistics
    being wrong. Each is checked here, and each failure is a hard stop.
    """
    import hashlib
    import json

    import numpy as np

    from aoi.analysis import summarise, t_critical, t_two_sided_p, welch
    from aoi.engines import available_engines, create_engine
    from aoi.pipeline import axis_origins, letterbox, parity, slice_origins
    from aoi.synth import SyntheticBoard

    cfg = build_config(args)
    checks: list = []          # (name, ok, detail)

    def check(name, ok, detail=""):
        # None means "not applicable here", which is a skip, not a failure -
        # bool(None) would quietly turn every skipped check into a red line.
        checks.append((name, None if ok is None else bool(ok), detail))

    # ---- 1. statistics ---------------------------------------------------
    # Reference values from SciPy 1.17 (stats.t.sf / stats.t.ppf). The tail is
    # evaluated here through a continued fraction so the project carries no
    # SciPy dependency; this is what proves that substitution is sound.
    for t, df, want in ((2.0, 10, 0.07338803), (1.5, 5, 0.19390368),
                        (3.2, 25, 0.00371597), (0.5, 100, 0.61817357)):
        got = t_two_sided_p(t, df)
        check(f"t tail p(t={t}, df={df})", abs(got - want) < 1e-7,
              f"{got:.8f} vs SciPy {want:.8f}")
    for df, want in ((5, 2.570582), (10, 2.228139), (30, 2.042272), (100, 1.983972)):
        got = t_critical(df)
        check(f"t critical df={df}", abs(got - want) < 1e-5,
              f"{got:.6f} vs SciPy {want:.6f}")

    rng = np.random.default_rng(0)
    a, b = rng.normal(100, 9, 40), rng.normal(60, 3, 45)
    t, df, pv = welch(a, b)
    check("Welch t-test", abs(t - 31.483436) < 1e-4 and pv < 1e-30,
          f"t={t:.4f}, p={pv:.2e}")
    s = summarise(a)
    check("95% CI on the mean", abs(s.ci95[0] - 97.161530) < 1e-4,
          f"[{s.ci95[0]:.4f}, {s.ci95[1]:.4f}]")

    # ---- 2. Python <-> C++ geometry parity --------------------------------
    try:
        bootstrap()
        import aoi_engine_cpp as native

        cases = [(3840, 1920, 0.20), (3840, 1536, 0.25), (3840, 1130, 0.20),
                 (2160, 1080, 0.20), (2160, 636, 0.20), (3840, 776, 0.20),
                 (1920, 1920, 0.00)]
        mismatches = [c for c in cases
                      if list(axis_origins(*c)) != list(native.axis_origins(*c))]
        check("SAHI axis origins identical in both languages", not mismatches,
              "all cases match" if not mismatches else f"differs at {mismatches}")
        check("ONNX Runtime providers registered", True,
              ", ".join(native.available_providers()))
    except ImportError:
        check("SAHI axis origins identical in both languages", None,
              "skipped - the C++ extension is not built")

    # ---- 3. weight provenance --------------------------------------------
    manifest = Path(cfg.model_onnx).with_suffix(".manifest.json")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))

        def sha(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        for key, path_key in (("pt_sha256", "model_pt"), ("onnx_sha256", "model_onnx")):
            path = Path(getattr(cfg, path_key))
            if not path.exists():
                check(f"{path.name} matches the manifest", None, "file missing")
                continue
            got = sha(path)
            # Both paths must run artefacts exported from the SAME training run.
            # Weight drift between them would show up as a language effect.
            check(f"{path.name} matches the manifest", got == data.get(key),
                  "hash matches" if got == data.get(key) else f"got {got[:16]}...")
    else:
        check("weight manifest present", None,
              f"skipped - no {manifest.name}; run `python main.py export-onnx`")

    # ---- 4. ground-truth round trip ---------------------------------------
    board = SyntheticBoard(1280, 720, cfg.synth_seed)
    frame, truth = board.frame(6)
    check("synthetic board plants ground truth", len(truth) > 0,
          f"{len(truth)} defects with known coordinates")
    small = cfg.variant(frame_width=1280, frame_height=720,
                        tile_width=640, tile_height=360)
    check("tile grid covers the frame",
          max(x + small.tile_width for x, _ in slice_origins(frame.shape, small)) == 1280,
          f"{len(slice_origins(frame.shape, small))} tiles")
    padded, scale, pad_x, pad_y = letterbox(frame, (640, 640))
    check("letterbox preserves aspect ratio",
          padded.shape[:2] == (640, 640) and pad_y > 0 and pad_x == 0,
          f"scale {scale:.4f}, pad ({pad_x}, {pad_y})")

    # ---- 5. detection parity between the two paths ------------------------
    avail = available_engines(cfg)
    if avail.get("python") and avail.get("cpp"):
        big, big_truth = SyntheticBoard(cfg.frame_width, cfg.frame_height,
                                        cfg.synth_seed).frame(cfg.synth_defects)
        results = {}
        for name in ("python", "cpp"):
            eng = create_engine(name, cfg)
            results[name] = eng.run_frame(big, 0).detections
            eng.close()
        par = parity(results["python"], results["cpp"])
        # Below 0.9 the two paths are not doing the same work, and any latency
        # ratio drawn from them is meaningless.
        check("the two paths return the same detections", par["agreement"] >= 0.9,
              f"{par['agreement'] * 100:.1f}% agreement, mean IoU "
              f"{par['mean_iou']:.3f} ({int(par['count_a'])} vs "
              f"{int(par['count_b'])} boxes)")
    else:
        check("the two paths return the same detections", None,
              "skipped - both paths must be available")

    # ---- report -----------------------------------------------------------
    print("=" * 74)
    print("  Validation")
    print("=" * 74)
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        failed += int(ok is False)
        print(f"  [{mark}] {name:<48} {detail}")
    print("-" * 74)
    print(f"  {sum(1 for _, o, _ in checks if o)} passed, "
          f"{sum(1 for _, o, _ in checks if o is None)} skipped, {failed} failed")
    if failed:
        print("\n  A failure here means the comparison is not valid. "
              "Fix it before quoting any result.")
    return 1 if failed else 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="main.py", description="AOI comparative benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="launch the inspection GUI (default)")
    g.add_argument("--engine", default="sim",
                   choices=["sim", "python", "python-lean", "cpp"])
    g.add_argument("--theme", default="dark", choices=["dark", "light"])
    g.add_argument("--camera", type=int)
    g.add_argument("--camera-backend", choices=["auto", "dshow", "msmf"],
                   help="Windows only: force a capture backend when a driver is "
                        "unstable under OpenCV's default choice")
    g.add_argument("--no-camera", action="store_true",
                   help="skip VideoCapture; synthetic board only")
    g.add_argument("--debug-shutdown", action="store_true")
    add_common(g)
    g.set_defaults(func=cmd_gui)

    b = sub.add_parser("bench", help="headless comparative study")
    b.add_argument("--engines", nargs="+",
                   choices=["sim", "python", "python-lean", "cpp"])
    b.add_argument("--trials", type=int)
    b.add_argument("--frames", type=int)
    b.add_argument("--sweep-tiles", action="store_true")
    b.add_argument("--sweep-resolution", action="store_true")
    b.add_argument("--sweep-all", action="store_true")
    b.add_argument("--no-figures", action="store_true")
    b.add_argument("--out", help="output directory (default: reports/)")
    add_common(b)
    b.set_defaults(func=cmd_bench)

    d = sub.add_parser("doctor", help="environment and configuration check")
    d.add_argument("--no-cameras", action="store_true",
                   help="skip camera probing, which can be slow on Windows")
    d.add_argument("--max-index", type=int, default=6,
                   help="highest camera index to probe (default: 6). Each miss "
                        "costs about a second per backend, so raise it only if "
                        "a camera you expect is absent.")
    add_common(d)
    d.set_defaults(func=cmd_doctor)

    bd = sub.add_parser("board", help="render synthetic boards + YOLO labels")
    bd.add_argument("output", nargs="?", default="reports/board.png")
    bd.add_argument("--count", type=int, default=1)
    bd.add_argument("--golden", action="store_true",
                    help="also write the defect-free reference board")
    add_common(bd)
    bd.set_defaults(func=cmd_board)

    v = sub.add_parser("verify", help="assert the parity and statistics invariants")
    add_common(v)
    v.set_defaults(func=cmd_verify)

    e = sub.add_parser("export-onnx", help=".pt -> .onnx with a hash manifest")
    e.add_argument("--weights", help="source .pt (default: the configured one)")
    e.add_argument("--imgsz", type=int, default=640)
    # OpenCV DNN frequently fails to parse opset 17+, and the C++ path may be
    # rebuilt against it, so 12 stays the default.
    e.add_argument("--opset", type=int, default=12)
    add_common(e)
    e.set_defaults(func=cmd_export_onnx)

    # No subcommand -> GUI, so `python main.py` still just runs the application.
    argv = sys.argv[1:]
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["gui"] + argv

    args = ap.parse_args(argv)
    if not hasattr(args, "func"):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
