"""
The shared inference pipeline: data contracts, SAHI geometry, letterboxing,
cross-tile NMS, and the box-matching used for both accuracy and path parity.

Everything here is engine-agnostic and must stay behaviourally identical to its
C++ counterpart in cpp_backend/aoi_engine.cpp. If the two paths do not slice the
same pixels, pad them the same way and suppress them with the same rule, the
comparison measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from aoi.config import RunConfig

# =============================================================================
# Timing model
# =============================================================================

# The research question is where Python's time goes relative to C++, so the
# frame is decomposed further than "pre / infer / post". Two of these stages
# exist specifically to isolate the language variable:
#
#   marshal_ms  - moving pixels across the language boundary and into a
#                 contiguous buffer the runtime will accept. In Python this is
#                 the ascontiguousarray copy and the tensor wrap; in C++ it is
#                 a cv::Mat ROI view, which is free. This is the "data
#                 marshaling overhead" the study set out to quantify.
#   dispatch_ms - per-call framework overhead around the actual kernels:
#                 argument validation, Python attribute lookups, result-object
#                 construction. Measured as (observed - accounted-for) so it
#                 cannot be silently absorbed into another stage.
STAGES: Tuple[str, ...] = (
    "slice_ms",       # compute the tile grid
    "marshal_ms",     # cross-boundary pixel handoff
    "preprocess_ms",  # letterbox + blob/normalise
    "inference_ms",   # forward pass, device-synchronised
    "decode_ms",      # raw tensor -> candidate boxes
    "nms_ms",         # per-tile suppression
    "merge_ms",       # cross-tile suppression on the master frame
    "dispatch_ms",    # residual framework overhead
)


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

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class FrameResult:
    """One frame through one engine: what was found, and where the time went."""

    detections: List[Detection] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return float(self.metrics.get("total_ms", 0.0))

    def stage_breakdown(self) -> Dict[str, float]:
        return {s: float(self.metrics.get(s, 0.0)) for s in STAGES}

    def to_stats(self, class_names: Sequence[str]) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            name = (class_names[d.class_id]
                    if 0 <= d.class_id < len(class_names) else f"cls{d.class_id}")
            counts[name] = counts.get(name, 0) + 1
        stats: Dict[str, object] = {
            "total_defects": len(self.detections),
            "top_defect_type": max(counts, key=counts.get) if counts else "N/A",
            "defect_counts": counts,
            "latency_ms": self.metrics.get("total_ms", 0.0),
        }
        stats.update(self.metrics)
        return stats


class StageTimer:
    """Accumulates per-stage milliseconds across a frame's tiles.

    Held as a plain dict of floats rather than nested objects: the per-tile loop
    runs nine times a frame and the timer must not become part of what is being
    measured.
    """

    __slots__ = ("_acc", "_t0")

    def __init__(self):
        import time

        self._acc: Dict[str, float] = {s: 0.0 for s in STAGES}
        self._t0 = time.perf_counter

    def add(self, stage: str, seconds: float) -> None:
        self._acc[stage] = self._acc.get(stage, 0.0) + seconds * 1000.0

    def mark(self) -> float:
        return self._t0()

    def since(self, stage: str, t0: float) -> float:
        now = self._t0()
        self.add(stage, now - t0)
        return now

    def finish(self, total_ms: float) -> Dict[str, float]:
        """Close the books and attribute the unexplained remainder.

        Anything the named stages did not account for is framework overhead, so
        it is reported as such instead of quietly inflating whichever stage
        happens to bracket it.
        """
        out = dict(self._acc)
        accounted = sum(v for k, v in out.items() if k != "dispatch_ms")
        out["dispatch_ms"] = max(0.0, total_ms - accounted)
        return out


# =============================================================================
# SAHI geometry
# =============================================================================

def axis_origins(total: int, tile: int, overlap: float) -> List[int]:
    """Evenly spaced tile origins along one axis.

    The naive "step forward, clamp the last tile inward" approach gives a
    lopsided grid - on a 3840px axis with 1920px tiles it yields 0/1536/1920,
    where the final pair overlaps 80% while the first overlaps 20%. That spends
    inference time on near-duplicate pixels and biases detection density toward
    one edge of the board.

    Instead: find the minimum tile count that satisfies the requested overlap,
    then distribute evenly. Overlap comes out uniform and always >= requested.
    """
    if tile >= total:
        return [0]
    step = max(1, int(tile * (1.0 - overlap)))
    span = total - tile
    n = max(2, (span + step - 1) // step + 1)
    return [round(i * span / (n - 1)) for i in range(n)]


def slice_origins(frame_shape: Sequence[int], cfg: RunConfig) -> List[Tuple[int, int]]:
    """Mirror of AoiEngine::slice_origins(). Every tile is full size, so no
    ragged edge tile skews per-tile latency."""
    h, w = frame_shape[:2]
    xs = axis_origins(w, cfg.tile_width, cfg.overlap_ratio)
    ys = axis_origins(h, cfg.tile_height, cfg.overlap_ratio)
    return [(x, y) for y in ys for x in xs]


def grid_shape(frame_shape: Sequence[int], cfg: RunConfig) -> Tuple[int, int]:
    h, w = frame_shape[:2]
    return (len(axis_origins(w, cfg.tile_width, cfg.overlap_ratio)),
            len(axis_origins(h, cfg.tile_height, cfg.overlap_ratio)))


def effective_overlap(total: int, tile: int, overlap: float) -> float:
    """What the grid ACTUALLY overlaps, which is often not what was requested.

    Reported in the GUI and the report because the gap between requested and
    effective is the single most misread number in this pipeline.
    """
    origins = axis_origins(total, tile, overlap)
    if len(origins) < 2:
        return 0.0
    step = origins[1] - origins[0]
    return max(0.0, 1.0 - step / tile)


def letterbox(img: np.ndarray, size: Tuple[int, int],
              colour: Tuple[int, int, int] = (114, 114, 114)):
    """Aspect-preserving pad to (w, h). Mirror of AoiEngine::letterbox().

    Returns (padded, scale, pad_x, pad_y). A plain resize would squash a 16:9
    tile and change which fine-pitch defects survive, so the two paths would not
    be detecting on the same image at all.
    """
    new_w, new_h = size
    h, w = img.shape[:2]
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (new_w - nw) // 2, (new_h - nh) // 2
    out = cv2.copyMakeBorder(resized, pad_y, new_h - nh - pad_y,
                             pad_x, new_w - nw - pad_x,
                             cv2.BORDER_CONSTANT, value=colour)
    return out, scale, pad_x, pad_y


def global_nms(boxes: List[List[int]], scores: List[float], class_ids: List[int],
               tiles: List[int], cfg: RunConfig) -> List[Detection]:
    """Class-aware NMS across the stitched master frame.

    A solder bridge sitting in a tile overlap is detected once per neighbouring
    tile. Without this the overlap ratio would silently inflate the defect count.
    """
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids,
                                   cfg.conf_threshold, cfg.nms_threshold)
    idx = np.asarray(keep).flatten().astype(int).tolist()
    return [Detection(*boxes[i], scores[i], class_ids[i], tiles[i]) for i in idx]


# =============================================================================
# Box matching - used for accuracy AND for path parity
# =============================================================================

def iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    """Greedy IoU matching between two detection sets."""

    matched: List[Tuple[int, int, float]] = field(default_factory=list)  # (i, j, iou)
    unmatched_a: List[int] = field(default_factory=list)
    unmatched_b: List[int] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    @property
    def mean_iou(self) -> float:
        return float(np.mean([m[2] for m in self.matched])) if self.matched else 0.0


def match_boxes(a: Sequence[Detection], b: Sequence[Detection],
                iou_threshold: float = 0.5,
                class_aware: bool = True) -> MatchResult:
    """Greedy highest-IoU-first matching.

    Greedy rather than Hungarian on purpose: detections after NMS are already
    well separated, so the optimal assignment and the greedy one agree in
    practice, and greedy stays O(n*m) with no dependency.
    """
    pairs = []
    for i, da in enumerate(a):
        for j, db in enumerate(b):
            if class_aware and da.class_id != db.class_id:
                continue
            v = iou(da, db)
            if v >= iou_threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    used_a, used_b, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, v))

    return MatchResult(
        matched=matched,
        unmatched_a=[i for i in range(len(a)) if i not in used_a],
        unmatched_b=[j for j in range(len(b)) if j not in used_b],
    )


def detection_metrics(pred: Sequence[Detection], truth: Sequence[Detection],
                      iou_threshold: float = 0.5) -> Dict[str, float]:
    """Precision / recall / F1 against ground truth at one IoU threshold.

    Only meaningful when ground truth exists - which, for this project, means
    the synthetic board where defects are planted at known coordinates.
    """
    m = match_boxes(pred, truth, iou_threshold)
    tp = m.n_matched
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "mean_iou": m.mean_iou}


def average_precision(pred: Sequence[Detection], truth: Sequence[Detection],
                      iou_threshold: float = 0.5) -> float:
    """AP at one IoU threshold, by the standard all-point interpolation.

    Detections are ranked by confidence and each is scored as TP or FP against
    the remaining unmatched ground truth, exactly as COCO does before its
    101-point interpolation. Single-image AP is noisy; it is reported per-frame
    only so the two paths can be compared on identical input.
    """
    if not truth:
        return 0.0
    if not pred:
        return 0.0

    order = sorted(range(len(pred)), key=lambda i: -pred[i].confidence)
    taken = [False] * len(truth)
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))

    for rank, pi in enumerate(order):
        best, best_j = 0.0, -1
        for j, gt in enumerate(truth):
            if taken[j] or gt.class_id != pred[pi].class_id:
                continue
            v = iou(pred[pi], gt)
            if v > best:
                best, best_j = v, j
        if best >= iou_threshold and best_j >= 0:
            taken[best_j] = True
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / len(truth)
    precision = ctp / np.maximum(ctp + cfp, 1e-9)

    # All-point interpolation: precision is made monotonically decreasing before
    # integrating, which is what stops a single lucky late detection from
    # inflating the area.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def parity(a: Sequence[Detection], b: Sequence[Detection],
           iou_threshold: float = 0.5) -> Dict[str, float]:
    """Do the two test paths agree on WHAT they found?

    This is the guard that makes a latency claim meaningful. "C++ is 1.4x
    faster" says nothing if the two paths are not returning the same
    detections - a path that finds half as much will always be quicker.
    Agreement is the Jaccard index over matched boxes, so a path that reports
    extra boxes is penalised as much as one that misses them.
    """
    m = match_boxes(a, b, iou_threshold)
    union = len(a) + len(b) - m.n_matched
    return {
        "agreement": m.n_matched / union if union else 1.0,
        "mean_iou": m.mean_iou,
        "count_a": len(a),
        "count_b": len(b),
        "only_a": len(m.unmatched_a),
        "only_b": len(m.unmatched_b),
    }


# =============================================================================
# Drawing  (always outside the timed region, identical for every engine)
# =============================================================================

PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (86, 232, 148), (255, 176, 59), (232, 106, 220),
    (94, 138, 255), (255, 236, 92), (255, 96, 110), (120, 232, 232),
)


def annotate(frame: np.ndarray, dets: Iterable[Detection],
             class_names: Sequence[str], thickness: int = 3,
             truth: Optional[Sequence[Detection]] = None) -> np.ndarray:
    """Draw detections, and optionally ground truth as a dashed reference.

    Runs outside every timer and is identical for all backends, so drawing cost
    can never land in one engine's column.
    """
    out = frame.copy()

    if truth:
        for t in truth:
            x1, y1, x2, y2 = t.xyxy
            _dashed_rect(out, (x1, y1), (x2, y2), (200, 200, 200), 2)

    for d in dets:
        colour = PALETTE[d.class_id % len(PALETTE)]
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        name = (class_names[d.class_id]
                if 0 <= d.class_id < len(class_names) else f"cls{d.class_id}")
        label = f"{name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ty = max(th + 8, y1 - 6)
        cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 10, ty + 4), colour, -1)
        cv2.putText(out, label, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (16, 20, 28), 2, cv2.LINE_AA)
    return out


def _dashed_rect(img, p1, p2, colour, thickness, dash=24):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, thickness)


def crop(frame: np.ndarray, det: Detection, pad: int = 24) -> np.ndarray:
    """Defect patch for the gallery, with context around the box."""
    h, w = frame.shape[:2]
    x1 = max(0, det.x - pad)
    y1 = max(0, det.y - pad)
    x2 = min(w, det.x + det.w + pad)
    y2 = min(h, det.y + det.h + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), np.uint8)
    return frame[y1:y2, x1:x2].copy()
