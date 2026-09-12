"""
Statistics and the benchmark study runner.

A single mean latency per path is not a result - it is one draw from a noisy
process on a machine with thermal throttling, a scheduler and a driver that
batches work. This module turns the raw per-frame timings into something that
can be defended:

  * interval estimates, not point estimates (95% CI on the mean);
  * a significance test appropriate for unequal variances (Welch);
  * a non-parametric effect size, so "significant" is not confused with "large";
  * a bootstrap interval on the SPEEDUP RATIO itself, because the ratio of two
    means has no closed-form standard error;
  * dispersion reported as CV and p95/p99, since a real-time inspection line
    cares about the tail, not the average;
  * detection parity between the paths, without which a latency claim is void.

No SciPy dependency: the t-distribution tail is evaluated through a continued
fraction for the regularised incomplete beta function, which is exact to well
beyond the precision anyone will quote.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aoi.config import RunConfig, environment_report
from aoi.pipeline import (
    STAGES, Detection, FrameResult, detection_metrics, average_precision, parity,
)

# =============================================================================
# Distributions  (no SciPy)
# =============================================================================

def _betacf(a: float, b: float, x: float, itmax: int = 200,
            eps: float = 3.0e-12) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with df degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return 1.0
    return regularised_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df: float, confidence: float = 0.95) -> float:
    """Two-sided critical value, by bisection on the tail probability.

    A normal-approximation z would understate the interval at the sample sizes
    a benchmark trial actually uses (n = 30 or fewer per trial).
    """
    if df <= 0:
        return float("nan")
    target = 1.0 - confidence
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# =============================================================================
# Descriptive statistics
# =============================================================================

@dataclass
class Summary:
    """Everything worth quoting about one sample of latencies."""

    n: int
    mean: float
    sd: float
    ci95: Tuple[float, float]
    median: float
    p95: float
    p99: float
    iqr: float
    minimum: float
    maximum: float
    cv: float                     # coefficient of variation, dimensionless

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n, "mean": self.mean, "sd": self.sd,
            "ci95_lo": self.ci95[0], "ci95_hi": self.ci95[1],
            "median": self.median, "p95": self.p95, "p99": self.p99,
            "iqr": self.iqr, "min": self.minimum, "max": self.maximum,
            "cv": self.cv,
        }

    def quote(self, unit: str = "ms") -> str:
        return (f"{self.mean:.2f} {unit} "
                f"[{self.ci95[0]:.2f}, {self.ci95[1]:.2f}] (n={self.n})")


def summarise(values: Sequence[float], confidence: float = 0.95) -> Summary:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(xs)
    if n == 0:
        z = 0.0
        return Summary(0, z, z, (z, z), z, z, z, z, z, z, z)
    if n == 1:
        v = xs[0]
        return Summary(1, v, 0.0, (v, v), v, v, v, 0.0, v, v, 0.0)

    arr = np.asarray(xs, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    half = t_critical(n - 1, confidence) * sd / math.sqrt(n)
    q1, q3 = np.percentile(arr, [25, 75])
    return Summary(
        n=n, mean=mean, sd=sd, ci95=(mean - half, mean + half),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)), p99=float(np.percentile(arr, 99)),
        iqr=float(q3 - q1), minimum=float(arr.min()), maximum=float(arr.max()),
        cv=sd / mean if mean else 0.0,
    )


# =============================================================================
# Comparison
# =============================================================================

@dataclass
class Comparison:
    """Path A vs Path B on one metric, with everything a reviewer will ask for."""

    a_label: str
    b_label: str
    a: Summary
    b: Summary
    t_statistic: float
    df: float
    p_value: float
    cliffs_delta: float
    effect_label: str
    speedup: float
    speedup_ci95: Tuple[float, float]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self) -> str:
        if self.a.n < 2 or self.b.n < 2:
            return "insufficient samples"
        direction = "faster" if self.speedup > 1 else "slower"
        sig = "significant" if self.significant else "not significant"
        return (f"{self.b_label} is {self.speedup:.2f}x {direction} "
                f"[{self.speedup_ci95[0]:.2f}, {self.speedup_ci95[1]:.2f}], "
                f"p={self.p_value:.2g} ({sig}), "
                f"effect {self.effect_label} (delta={self.cliffs_delta:+.2f})")

    def as_dict(self) -> Dict[str, object]:
        return {
            "a_label": self.a_label, "b_label": self.b_label,
            "a": self.a.as_dict(), "b": self.b.as_dict(),
            "t": self.t_statistic, "df": self.df, "p_value": self.p_value,
            "cliffs_delta": self.cliffs_delta, "effect": self.effect_label,
            "speedup": self.speedup, "speedup_ci95_lo": self.speedup_ci95[0],
            "speedup_ci95_hi": self.speedup_ci95[1],
            "significant": self.significant, "verdict": self.verdict(),
        }


def welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Welch's unequal-variance t-test. Returns (t, df, two-sided p).

    Student's pooled-variance test is the wrong tool here: the two paths have
    visibly different spread - the interpreted path's tail is dominated by GC
    pauses the compiled path does not have - and pooling would understate the
    standard error of the difference.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 0.0, 1.0
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    sa, sb = va / na, vb / nb
    denom = math.sqrt(sa + sb)
    if denom == 0:
        return 0.0, 0.0, 1.0
    t = (ma - mb) / denom
    df_den = (sa ** 2 / (na - 1)) + (sb ** 2 / (nb - 1))
    df = (sa + sb) ** 2 / df_den if df_den > 0 else float(na + nb - 2)
    return t, df, t_two_sided_p(t, df)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """Non-parametric effect size: P(a>b) - P(a<b), in [-1, 1].

    Reported because a p-value only says the difference is detectable, not that
    it matters. With 30 frames per path, a 2% latency difference can be highly
    significant and completely irrelevant to a production line.

    Thresholds are Romano et al.'s: 0.147 / 0.33 / 0.474.
    """
    if not a or not b:
        return 0.0, "none"
    xa = np.sort(np.asarray(a, dtype=float))
    xb = np.asarray(b, dtype=float)
    # searchsorted instead of the O(n*m) double loop: 30x30 is fine, but the
    # sweep calls this hundreds of times.
    greater = int(np.sum(np.searchsorted(xa, xb, side="left")))
    less = int(np.sum(len(xa) - np.searchsorted(xa, xb, side="right")))
    delta = (less - greater) / (len(xa) * len(xb))
    mag = abs(delta)
    label = ("negligible" if mag < 0.147 else
             "small" if mag < 0.33 else
             "medium" if mag < 0.474 else "large")
    return float(delta), label


def bootstrap_ratio_ci(a: Sequence[float], b: Sequence[float],
                       iterations: int = 4000, confidence: float = 0.95,
                       seed: int = 7) -> Tuple[float, Tuple[float, float]]:
    """Percentile bootstrap CI for mean(a)/mean(b).

    The ratio of two means is not normally distributed and has no usable
    closed-form standard error, so the headline "N times faster" figure is
    otherwise quoted with no uncertainty at all. Resampling both samples with
    replacement gives an honest interval.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    xa, xb = np.asarray(a, float), np.asarray(b, float)
    point = float(xa.mean() / xb.mean()) if xb.mean() else 0.0
    ratios = np.empty(iterations)
    for i in range(iterations):
        ra = rng.choice(xa, size=len(xa), replace=True).mean()
        rb = rng.choice(xb, size=len(xb), replace=True).mean()
        ratios[i] = ra / rb if rb else np.nan
    lo, hi = np.nanpercentile(ratios, [(1 - confidence) / 2 * 100,
                                       (1 + confidence) / 2 * 100])
    return point, (float(lo), float(hi))


def compare(a: Sequence[float], b: Sequence[float],
            a_label: str = "A", b_label: str = "B") -> Comparison:
    t, df, p = welch(a, b)
    delta, effect = cliffs_delta(a, b)
    speedup, ci = bootstrap_ratio_ci(a, b)
    return Comparison(a_label, b_label, summarise(a), summarise(b),
                      t, df, p, delta, effect, speedup, ci)


# =============================================================================
# Study runner
# =============================================================================

@dataclass
class TrialResult:
    engine: str
    label: str
    backend_desc: str
    device: str
    trial: int
    latencies: List[float] = field(default_factory=list)
    stages: Dict[str, List[float]] = field(default_factory=dict)
    resources: Dict[str, List[float]] = field(default_factory=dict)
    detections: List[List[Detection]] = field(default_factory=list)
    accuracy: List[Dict[str, float]] = field(default_factory=list)
    load_seconds: float = 0.0


@dataclass
class StudyResult:
    """One complete A/B study: config, environment, per-engine trials, verdicts."""

    config: RunConfig
    environment: Dict[str, object]
    trials: List[TrialResult] = field(default_factory=list)
    parity_frames: List[Dict[str, float]] = field(default_factory=list)
    sweeps: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    started: str = ""
    duration_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    # ---- accessors ------------------------------------------------------
    def engines(self) -> List[str]:
        seen: List[str] = []
        for t in self.trials:
            if t.engine not in seen:
                seen.append(t.engine)
        return seen

    def latencies(self, engine: str) -> List[float]:
        return [v for t in self.trials if t.engine == engine for v in t.latencies]

    def stage(self, engine: str, stage: str) -> List[float]:
        return [v for t in self.trials if t.engine == engine
                for v in t.stages.get(stage, [])]

    def resource(self, engine: str, key: str) -> List[float]:
        return [v for t in self.trials if t.engine == engine
                for v in t.resources.get(key, [])]

    def label(self, engine: str) -> str:
        for t in self.trials:
            if t.engine == engine:
                return t.label
        return engine

    def backend(self, engine: str) -> str:
        for t in self.trials:
            if t.engine == engine:
                return t.backend_desc
        return ""

    def summary(self, engine: str) -> Summary:
        return summarise(self.latencies(engine))

    def stage_means(self, engine: str) -> Dict[str, float]:
        return {s: (statistics.fmean(self.stage(engine, s))
                    if self.stage(engine, s) else 0.0) for s in STAGES}

    def accuracy(self, engine: str) -> Dict[str, float]:
        rows = [a for t in self.trials if t.engine == engine for a in t.accuracy]
        if not rows:
            return {}
        return {k: statistics.fmean([r[k] for r in rows]) for k in rows[0]}

    def between_trial_cv(self, engine: str) -> float:
        """Dispersion of the trial MEANS - the run-to-run reproducibility.

        Within-trial CV mixes in per-frame jitter. A study whose trial means
        disagree by more than a few percent was measured on a machine that was
        not thermally settled, and should be re-run before anything is claimed.
        """
        means = [statistics.fmean(t.latencies) for t in self.trials
                 if t.engine == engine and t.latencies]
        if len(means) < 2:
            return 0.0
        m = statistics.fmean(means)
        return statistics.stdev(means) / m if m else 0.0

    def headline(self, a: str = "python", b: str = "cpp") -> Optional[Comparison]:
        la, lb = self.latencies(a), self.latencies(b)
        if len(la) < 2 or len(lb) < 2:
            return None
        return compare(la, lb, self.label(a), self.label(b))

    def stage_attribution(self, a: str = "python", b: str = "cpp"
                          ) -> List[Dict[str, object]]:
        """Where the difference actually comes from, stage by stage.

        This is the table the headline ratio is decomposed into: each stage's
        absolute contribution to the gap and its share of it. A result showing
        90% of the gap in `marshal_ms` supports a very different conclusion from
        one showing it in `inference_ms`.
        """
        ma, mb = self.stage_means(a), self.stage_means(b)
        total_gap = sum(ma.values()) - sum(mb.values())
        rows = []
        for s in STAGES:
            gap = ma[s] - mb[s]
            rows.append({
                "stage": s.replace("_ms", ""),
                "a_ms": ma[s], "b_ms": mb[s], "delta_ms": gap,
                "share": gap / total_gap if total_gap else 0.0,
                "ratio": ma[s] / mb[s] if mb[s] > 1e-9 else float("nan"),
            })
        rows.sort(key=lambda r: -abs(r["delta_ms"]))
        return rows

    def parity_summary(self) -> Dict[str, float]:
        if not self.parity_frames:
            return {}
        keys = self.parity_frames[0].keys()
        return {k: statistics.fmean([p[k] for p in self.parity_frames]) for k in keys}


# ---------------------------------------------------------------------------

ProgressFn = Callable[[str, float], None]
AbortFn = Callable[[], bool]


def run_study(cfg: RunConfig,
              engines: Sequence[str],
              frames: Optional[Sequence[Tuple[np.ndarray, List[Detection]]]] = None,
              progress: Optional[ProgressFn] = None,
              abort: Optional[AbortFn] = None) -> StudyResult:
    """Run the full comparative study, headless.

    Every engine sees the SAME frame objects in the SAME order, and each trial
    replays that identical sequence, so the only thing varying between trials is
    machine state. Warmup and the first `cfg.discard_frames` measured frames are
    discarded on every path: the first passes after a cold start pay for cuDNN
    algorithm search, workspace allocation and page faults on the frame buffer,
    none of which recur.
    """
    from aoi.engines import create_engine, unavailable_reason
    from aoi.synth import SyntheticBoard

    t_study = time.perf_counter()
    say = progress or (lambda msg, frac: None)
    stopped = abort or (lambda: False)

    if frames is None:
        say("Rendering synthetic 4K board...", 0.0)
        board = SyntheticBoard(cfg.frame_width, cfg.frame_height, cfg.synth_seed)
        frames = list(board.sequence(max(4, cfg.trial_frames), cfg.synth_defects))

    result = StudyResult(
        config=cfg,
        environment=environment_report(cfg),
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Per-engine detections on frame 0, kept so parity can be computed after
    # every engine has run without holding every frame's boxes in memory.
    first_frame_dets: Dict[str, List[Detection]] = {}
    total_units = max(1, len(engines) * cfg.trials)
    unit = 0

    for engine_name in engines:
        try:
            say(f"Loading {engine_name}...", unit / total_units)
            t0 = time.perf_counter()
            engine = create_engine(engine_name, cfg)
            load_s = time.perf_counter() - t0
        except Exception as exc:                                # noqa: BLE001
            result.notes.append(f"{engine_name}: {exc}")
            unit += cfg.trials
            continue

        for trial_idx in range(cfg.trials):
            if stopped():
                result.notes.append("Study aborted by the operator.")
                engine.close()
                return _finalise(result, t_study)

            tr = TrialResult(
                engine=engine_name, label=engine.label,
                backend_desc=engine.backend_desc, device=engine.device_desc,
                trial=trial_idx, load_seconds=load_s,
                stages={s: [] for s in STAGES},
                resources={k: [] for k in ("cpu_percent", "gpu_percent",
                                           "gpu_mem_mb", "rss_mb", "peak_rss_mb")},
            )

            for i, (frame, truth) in enumerate(frames):
                if stopped():
                    break
                if hasattr(engine, "set_truth"):
                    engine.set_truth(truth)

                res = engine.run_frame(frame, i)

                if i < cfg.discard_frames:
                    continue                       # settle, do not record

                tr.latencies.append(res.metrics.get("total_ms", 0.0))
                for s in STAGES:
                    tr.stages[s].append(float(res.metrics.get(s, 0.0)))
                for k in tr.resources:
                    v = float(res.metrics.get(k, -1.0))
                    if v >= 0:
                        tr.resources[k].append(v)

                if truth:
                    acc = detection_metrics(res.detections, truth)
                    acc["ap50"] = average_precision(res.detections, truth)
                    tr.accuracy.append(acc)

                if trial_idx == 0 and i == cfg.discard_frames:
                    first_frame_dets[engine_name] = res.detections

                if i % 5 == 0:
                    frac = (unit + i / max(1, len(frames))) / total_units
                    say(f"{engine_name} trial {trial_idx + 1}/{cfg.trials} "
                        f"frame {i + 1}/{len(frames)}", frac)

            result.trials.append(tr)
            unit += 1

        engine.close()

    # ---- parity between the two headline paths --------------------------
    if "python" in first_frame_dets and "cpp" in first_frame_dets:
        result.parity_frames.append(
            parity(first_frame_dets["python"], first_frame_dets["cpp"]))
    if "python-lean" in first_frame_dets and "cpp" in first_frame_dets:
        result.notes.append(
            "Path A' vs B parity: "
            + ", ".join(f"{k}={v:.3f}" for k, v in
                        parity(first_frame_dets["python-lean"],
                               first_frame_dets["cpp"]).items()))

    return _finalise(result, t_study)


def _finalise(result: StudyResult, t0: float) -> StudyResult:
    result.duration_s = time.perf_counter() - t0
    for engine in result.engines():
        cv = result.between_trial_cv(engine)
        if cv > 0.05:
            result.notes.append(
                f"{engine}: between-trial CV is {cv * 100:.1f}% (>5%). The machine "
                "was probably not thermally settled; re-run before quoting.")
    return result


# =============================================================================
# Parameter sweeps
# =============================================================================

def sweep_tiles(cfg: RunConfig, engines: Sequence[str],
                presets: Dict[str, Tuple[int, int, float]],
                frames_per_point: int = 12,
                progress: Optional[ProgressFn] = None,
                abort: Optional[AbortFn] = None) -> List[Dict[str, object]]:
    """Latency as a function of the SAHI grid.

    The point of the sweep is that a single operating point cannot show whether
    the language gap is fixed overhead or scales with the number of inferences.
    Fixed overhead flattens as tiles increase; per-call overhead does not.
    """
    from aoi.engines import create_engine
    from aoi.pipeline import slice_origins
    from aoi.synth import SyntheticBoard

    say = progress or (lambda msg, frac: None)
    stopped = abort or (lambda: False)

    say("Rendering board for sweep...", 0.0)
    board = SyntheticBoard(cfg.frame_width, cfg.frame_height, cfg.synth_seed)
    frames = list(board.sequence(frames_per_point, cfg.synth_defects))

    rows: List[Dict[str, object]] = []
    total = max(1, len(presets) * len(engines))
    done = 0

    for preset_name, (tw, th, ov) in presets.items():
        point_cfg = cfg.variant(tile_width=tw, tile_height=th, overlap_ratio=ov)
        n_tiles = len(slice_origins((cfg.frame_height, cfg.frame_width), point_cfg))
        for engine_name in engines:
            if stopped():
                return rows
            say(f"{preset_name} / {engine_name}", done / total)
            try:
                engine = create_engine(engine_name, point_cfg)
            except Exception as exc:                            # noqa: BLE001
                rows.append({"preset": preset_name, "engine": engine_name,
                             "tiles": n_tiles, "error": str(exc)})
                done += 1
                continue

            lat, infer = [], []
            for i, (frame, truth) in enumerate(frames):
                if hasattr(engine, "set_truth"):
                    engine.set_truth(truth)
                r = engine.run_frame(frame, i)
                if i >= 2:
                    lat.append(r.metrics.get("total_ms", 0.0))
                    infer.append(r.metrics.get("inference_ms", 0.0))
            engine.close()

            s = summarise(lat)
            rows.append({
                "preset": preset_name, "engine": engine_name,
                "tile_w": tw, "tile_h": th, "overlap": ov, "tiles": n_tiles,
                "mean_ms": s.mean, "ci_lo": s.ci95[0], "ci_hi": s.ci95[1],
                "p95_ms": s.p95, "fps": 1000.0 / s.mean if s.mean else 0.0,
                "inference_ms": statistics.fmean(infer) if infer else 0.0,
                "ms_per_tile": s.mean / n_tiles if n_tiles else 0.0,
            })
            done += 1

    return rows


def sweep_resolution(cfg: RunConfig, engines: Sequence[str],
                     resolutions: Sequence[Tuple[int, int]] = (
                         (1920, 1080), (2560, 1440), (3200, 1800), (3840, 2160)),
                     frames_per_point: int = 10,
                     progress: Optional[ProgressFn] = None,
                     abort: Optional[AbortFn] = None) -> List[Dict[str, object]]:
    """Latency against input resolution - the scaling curve.

    Tile geometry is held proportional to the frame so the tile COUNT stays
    constant across resolutions. Anything that still changes is pixel-throughput
    cost, not a change in the amount of dispatch work.
    """
    from aoi.engines import create_engine
    from aoi.synth import SyntheticBoard

    say = progress or (lambda msg, frac: None)
    stopped = abort or (lambda: False)
    rows: List[Dict[str, object]] = []
    total = max(1, len(resolutions) * len(engines))
    done = 0

    for (w, h) in resolutions:
        scale = w / cfg.frame_width
        point_cfg = cfg.variant(
            frame_width=w, frame_height=h,
            tile_width=max(64, int(cfg.tile_width * scale)),
            tile_height=max(64, int(cfg.tile_height * scale)))
        say(f"Rendering {w}x{h}...", done / total)
        board = SyntheticBoard(w, h, cfg.synth_seed)
        frames = list(board.sequence(frames_per_point, cfg.synth_defects))

        for engine_name in engines:
            if stopped():
                return rows
            say(f"{w}x{h} / {engine_name}", done / total)
            try:
                engine = create_engine(engine_name, point_cfg)
            except Exception as exc:                            # noqa: BLE001
                rows.append({"width": w, "height": h, "engine": engine_name,
                             "error": str(exc)})
                done += 1
                continue

            lat = []
            for i, (frame, truth) in enumerate(frames):
                if hasattr(engine, "set_truth"):
                    engine.set_truth(truth)
                r = engine.run_frame(frame, i)
                if i >= 2:
                    lat.append(r.metrics.get("total_ms", 0.0))
            engine.close()

            s = summarise(lat)
            mp = (w * h) / 1e6
            rows.append({
                "width": w, "height": h, "megapixels": mp, "engine": engine_name,
                "mean_ms": s.mean, "ci_lo": s.ci95[0], "ci_hi": s.ci95[1],
                "fps": 1000.0 / s.mean if s.mean else 0.0,
                "ms_per_megapixel": s.mean / mp if mp else 0.0,
            })
            done += 1

    return rows
