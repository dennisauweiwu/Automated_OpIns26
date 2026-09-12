"""
Synthetic 4K PCB generator with ground-truth defect injection.

Why this exists
---------------
The original test pattern was random lines and circles. The detector correctly
found nothing in it, so every logged frame recorded zero detections and the
application demonstrated an empty PASS screen. Worse, with no ground truth there
was no way to show that the two test paths were finding the SAME defects - which
is the precondition for any latency comparison to mean anything.

This module renders a plausible board and then plants defects at coordinates it
returns. That gives three things at once:

  1. a demo where detections actually appear;
  2. precision / recall / AP without an annotated dataset on the machine;
  3. a fixed, seeded input, so a benchmark re-run is comparable to the last one.

The clean board is rendered once and cached. Defect injection copies the cached
board and paints a handful of small patches, so regenerating a frame costs a
memcpy rather than a full 4K render and never dominates a timing measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from aoi.pipeline import Detection

# Board palette, BGR. Chosen to survive JPEG-ish contrast and to give the
# defect classes visibly different local statistics.
SUBSTRATE = (46, 92, 44)
SUBSTRATE_DARK = (34, 70, 33)
COPPER = (58, 148, 196)
COPPER_BRIGHT = (86, 186, 230)
PAD = (126, 196, 226)
SILK = (232, 236, 238)
SOLDER = (168, 176, 182)
IC_BODY = (38, 40, 44)

CLASS_INDEX: Dict[str, int] = {
    "mouse_bite": 0, "spur": 1, "missing_hole": 2,
    "short": 3, "open_circuit": 4, "spurious_copper": 5,
}


@dataclass
class BoardGeometry:
    """Where the renderer put things, so defects can be planted ON them.

    A mouse bite has to land on a trace edge and a short has to bridge two
    traces that are actually adjacent; planting either at a random coordinate
    produces an artefact no detector should be expected to find.
    """

    h_traces: List[Tuple[int, int, int, int]]   # x1, y, x2, thickness
    v_traces: List[Tuple[int, int, int, int]]   # x, y1, y2, thickness
    pads: List[Tuple[int, int, int]]            # cx, cy, radius
    free_space: List[Tuple[int, int]]           # substrate-only points


# =============================================================================
# Board rendering
# =============================================================================

def render_board(width: int = 3840, height: int = 2160,
                 seed: int = 2601) -> Tuple[np.ndarray, BoardGeometry]:
    """Render a clean 4K board. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), SUBSTRATE, np.uint8)

    _add_substrate_texture(img, rng)

    s = width / 3840.0                       # scale factor for non-4K sizes
    h_traces: List[Tuple[int, int, int, int]] = []
    v_traces: List[Tuple[int, int, int, int]] = []
    pads: List[Tuple[int, int, int]] = []

    # --- ground plane hatching -----------------------------------------------
    for y in range(0, height, int(90 * s)):
        cv2.line(img, (0, y), (width, y), SUBSTRATE_DARK, max(1, int(3 * s)))

    # --- routed buses --------------------------------------------------------
    # Horizontal buses first, then vertical; the pair gives real adjacency for
    # the "short" class and real edges for "mouse_bite".
    for _ in range(34):
        y = int(rng.integers(int(90 * s), height - int(90 * s)))
        x1 = int(rng.integers(0, width // 2))
        x2 = int(min(width - 1, x1 + rng.integers(int(500 * s), int(2200 * s))))
        t = int(rng.integers(max(4, int(7 * s)), max(6, int(15 * s))))
        cv2.line(img, (x1, y), (x2, y), COPPER, t, cv2.LINE_AA)
        h_traces.append((x1, y, x2, t))
        # Parallel neighbour at a fine pitch - the pair a solder bridge shorts.
        gap = int(rng.integers(max(9, int(14 * s)), max(14, int(30 * s))))
        if y + gap < height - int(60 * s):
            cv2.line(img, (x1, y + gap), (x2, y + gap), COPPER, t, cv2.LINE_AA)
            h_traces.append((x1, y + gap, x2, t))

    for _ in range(26):
        x = int(rng.integers(int(90 * s), width - int(90 * s)))
        y1 = int(rng.integers(0, height // 2))
        y2 = int(min(height - 1, y1 + rng.integers(int(400 * s), int(1400 * s))))
        t = int(rng.integers(max(4, int(7 * s)), max(6, int(14 * s))))
        cv2.line(img, (x, y1), (x, y2), COPPER, t, cv2.LINE_AA)
        v_traces.append((x, y1, y2, t))

    # --- vias ----------------------------------------------------------------
    for _ in range(260):
        cx = int(rng.integers(int(40 * s), width - int(40 * s)))
        cy = int(rng.integers(int(40 * s), height - int(40 * s)))
        r = int(rng.integers(max(7, int(10 * s)), max(11, int(19 * s))))
        cv2.circle(img, (cx, cy), r, PAD, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(2, r // 2), SUBSTRATE_DARK, -1, cv2.LINE_AA)
        pads.append((cx, cy, r))

    # --- components ----------------------------------------------------------
    for _ in range(9):
        _draw_ic(img, rng, s, pads)
    for _ in range(46):
        _draw_smd(img, rng, s, pads)
    for _ in range(5):
        _draw_connector(img, rng, s, pads)

    # --- silkscreen ----------------------------------------------------------
    for _ in range(40):
        x = int(rng.integers(int(60 * s), width - int(200 * s)))
        y = int(rng.integers(int(60 * s), height - int(60 * s)))
        txt = f"{rng.choice(list('RCUJQD'))}{int(rng.integers(1, 400))}"
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s, SILK, max(1, int(2 * s)), cv2.LINE_AA)

    # --- board outline -------------------------------------------------------
    m = int(28 * s)
    cv2.rectangle(img, (m, m), (width - m, height - m), SILK, max(2, int(4 * s)))
    for cx, cy in ((m * 3, m * 3), (width - m * 3, m * 3),
                   (m * 3, height - m * 3), (width - m * 3, height - m * 3)):
        cv2.circle(img, (cx, cy), int(34 * s), SILK, max(2, int(4 * s)), cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(20 * s), SUBSTRATE_DARK, -1, cv2.LINE_AA)

    # --- free substrate, for spurious-copper placement -----------------------
    free: List[Tuple[int, int]] = []
    guard = int(60 * s)
    for _ in range(4000):
        px = int(rng.integers(guard, width - guard))
        py = int(rng.integers(guard, height - guard))
        patch = img[py - 14:py + 14, px - 14:px + 14]
        # Substrate is green-dominant; copper and silk are not. Cheap test that
        # keeps a spurious-copper blob off an existing trace.
        if patch.size and patch[..., 1].mean() > patch[..., 0].mean() + 24:
            free.append((px, py))
        if len(free) >= 400:
            break

    _add_illumination(img, rng)
    return img, BoardGeometry(h_traces, v_traces, pads, free)


def _add_substrate_texture(img: np.ndarray, rng) -> None:
    h, w = img.shape[:2]
    # Woven-glass weave at 1/8 scale, then upsampled: a full-resolution noise
    # field would cost more than the rest of the render combined.
    small = rng.integers(-10, 11, (h // 8, w // 8, 3), dtype=np.int16)
    noise = cv2.resize(small.astype(np.int16), (w, h), interpolation=cv2.INTER_LINEAR)
    np.clip(img.astype(np.int16) + noise, 0, 255, out=noise)
    img[:] = noise.astype(np.uint8)


def _add_illumination(img: np.ndarray, rng) -> None:
    """Ring-light falloff plus a couple of specular highlights.

    A perfectly flat exposure is the tell of a synthetic image, and uniform
    lighting also makes the inspection problem easier than the real one.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h:8, 0:w:8].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vig = np.clip(1.06 - 0.22 * r ** 2, 0.72, 1.08)
    for _ in range(3):
        sx = float(rng.uniform(0.15, 0.85)) * w
        sy = float(rng.uniform(0.15, 0.85)) * h
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / (0.16 * w)
        vig += 0.16 * np.exp(-d ** 2)
    full = cv2.resize(vig, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.astype(np.float32) * full[..., None]
    np.clip(out, 0, 255, out=out)
    img[:] = out.astype(np.uint8)


def _draw_ic(img, rng, s, pads):
    w = int(rng.integers(int(180 * s), int(460 * s)))
    h = int(rng.integers(int(140 * s), int(340 * s)))
    x = int(rng.integers(int(60 * s), max(int(61 * s), img.shape[1] - w - int(60 * s))))
    y = int(rng.integers(int(60 * s), max(int(61 * s), img.shape[0] - h - int(60 * s))))
    cv2.rectangle(img, (x, y), (x + w, y + h), IC_BODY, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 74, 80), max(1, int(3 * s)))
    cv2.circle(img, (x + int(24 * s), y + int(24 * s)), int(10 * s), (120, 124, 130), -1)

    pitch = max(int(16 * s), 14)
    pin_w = max(4, pitch // 2)
    for px in range(x + pitch, x + w - pitch, pitch):
        cv2.rectangle(img, (px, y - int(18 * s)), (px + pin_w, y), SOLDER, -1)
        cv2.rectangle(img, (px, y + h), (px + pin_w, y + h + int(18 * s)), SOLDER, -1)
        pads.append((px + pin_w // 2, y - int(9 * s), pin_w))
        pads.append((px + pin_w // 2, y + h + int(9 * s), pin_w))
    cv2.putText(img, f"U{int(rng.integers(1, 99))}", (x + int(30 * s), y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, SILK, max(1, int(2 * s)), cv2.LINE_AA)


def _draw_smd(img, rng, s, pads):
    w = int(rng.integers(int(34 * s), int(96 * s)))
    h = int(rng.integers(int(20 * s), int(44 * s)))
    if rng.random() < 0.5:
        w, h = h, w
    x = int(rng.integers(int(60 * s), max(int(61 * s), img.shape[1] - w - int(60 * s))))
    y = int(rng.integers(int(60 * s), max(int(61 * s), img.shape[0] - h - int(60 * s))))
    cv2.rectangle(img, (x, y), (x + w, y + h), (52, 54, 60), -1)
    cap = max(4, int(min(w, h) * 0.32))
    cv2.rectangle(img, (x, y), (x + cap, y + h), SOLDER, -1)
    cv2.rectangle(img, (x + w - cap, y), (x + w, y + h), SOLDER, -1)
    pads.append((x + cap // 2, y + h // 2, cap))


def _draw_connector(img, rng, s, pads):
    n = int(rng.integers(6, 18))
    pitch = int(rng.integers(int(30 * s), int(52 * s)))
    w, h = n * pitch, int(rng.integers(int(80 * s), int(140 * s)))
    x = int(rng.integers(int(60 * s), max(int(61 * s), img.shape[1] - w - int(60 * s))))
    y = int(rng.integers(int(60 * s), max(int(61 * s), img.shape[0] - h - int(60 * s))))
    cv2.rectangle(img, (x, y), (x + w, y + h), (28, 30, 34), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (86, 90, 96), max(1, int(3 * s)))
    for i in range(n):
        px = x + pitch // 2 + i * pitch
        cv2.circle(img, (px, y + h // 2), max(4, pitch // 4), SOLDER, -1, cv2.LINE_AA)
        pads.append((px, y + h // 2, max(4, pitch // 4)))


# =============================================================================
# Defect injection
# =============================================================================

def _inject_mouse_bite(img, rng, geom, s):
    """Semicircular notch eaten out of a trace edge."""
    if not geom.h_traces:
        return None
    x1, y, x2, t = geom.h_traces[int(rng.integers(len(geom.h_traces)))]
    if x2 - x1 < 200:
        return None
    px = int(rng.integers(x1 + 60, x2 - 60))
    r = max(6, int(t * rng.uniform(0.9, 1.5)))
    side = -1 if rng.random() < 0.5 else 1
    cy = y + side * (t // 2)
    cv2.circle(img, (px, cy), r, SUBSTRATE, -1, cv2.LINE_AA)
    return _box(px, y, 2 * (r + t), 2 * (r + t), "mouse_bite")


def _inject_spur(img, rng, geom, s):
    """Unwanted copper whisker growing off a trace."""
    if not geom.h_traces:
        return None
    x1, y, x2, t = geom.h_traces[int(rng.integers(len(geom.h_traces)))]
    if x2 - x1 < 200:
        return None
    px = int(rng.integers(x1 + 60, x2 - 60))
    length = int(rng.integers(int(22 * s), int(52 * s)))
    side = -1 if rng.random() < 0.5 else 1
    cv2.line(img, (px, y), (px + int(length * 0.4), y + side * length),
             COPPER_BRIGHT, max(3, t // 2), cv2.LINE_AA)
    return _box(px + length // 5, y + side * length // 2,
                int(length * 1.7), int(length * 1.7), "spur")


def _inject_missing_hole(img, rng, geom, s):
    """A via pad annular ring left undrilled - the barrel is simply absent."""
    if not geom.pads:
        return None
    cx, cy, r = geom.pads[int(rng.integers(len(geom.pads)))]
    if r < 6:
        return None
    cv2.circle(img, (cx, cy), max(2, r // 2) + 1, PAD, -1, cv2.LINE_AA)
    return _box(cx, cy, r * 4, r * 4, "missing_hole")


def _inject_short(img, rng, geom, s):
    """Solder bridge between two adjacent traces at a fine pitch."""
    pairs = []
    for i, (ax1, ay, ax2, at) in enumerate(geom.h_traces):
        for bx1, by, bx2, bt in geom.h_traces[i + 1:i + 3]:
            if 0 < by - ay < int(36 * s) and min(ax2, bx2) - max(ax1, bx1) > 220:
                pairs.append((ax1, ay, ax2, by, max(ax1, bx1), min(ax2, bx2)))
    if not pairs:
        return None
    _, ay, _, by, lo, hi = pairs[int(rng.integers(len(pairs)))]
    px = int(rng.integers(lo + 60, hi - 60))
    w = int(rng.integers(int(9 * s), int(20 * s)))
    cv2.rectangle(img, (px, ay), (px + w, by), COPPER_BRIGHT, -1)
    return _box(px + w // 2, (ay + by) // 2, w + int(30 * s), (by - ay) + int(30 * s),
                "short")


def _inject_open_circuit(img, rng, geom, s):
    """A gap etched clean through a trace."""
    pool = geom.h_traces + [(y1, x, y2, t) for x, y1, y2, t in geom.v_traces]
    if not geom.h_traces:
        return None
    x1, y, x2, t = geom.h_traces[int(rng.integers(len(geom.h_traces)))]
    if x2 - x1 < 240:
        return None
    px = int(rng.integers(x1 + 80, x2 - 80))
    gap = int(rng.integers(int(10 * s), int(26 * s)))
    cv2.rectangle(img, (px, y - t), (px + gap, y + t), SUBSTRATE, -1)
    return _box(px + gap // 2, y, gap + int(34 * s), t * 2 + int(30 * s), "open_circuit")


def _inject_spurious_copper(img, rng, geom, s):
    """Copper island stranded in the substrate where nothing should be."""
    if not geom.free_space:
        return None
    px, py = geom.free_space[int(rng.integers(len(geom.free_space)))]
    w = int(rng.integers(int(18 * s), int(46 * s)))
    h = int(rng.integers(int(14 * s), int(38 * s)))
    if rng.random() < 0.5:
        cv2.ellipse(img, (px, py), (w // 2, h // 2), float(rng.uniform(0, 180)),
                    0, 360, COPPER, -1, cv2.LINE_AA)
    else:
        pts = np.array([[px + int(rng.integers(-w // 2, w // 2)),
                         py + int(rng.integers(-h // 2, h // 2))] for _ in range(6)])
        cv2.fillPoly(img, [cv2.convexHull(pts)], COPPER, cv2.LINE_AA)
    return _box(px, py, w + int(28 * s), h + int(28 * s), "spurious_copper")


_INJECTORS = {
    "mouse_bite": _inject_mouse_bite,
    "spur": _inject_spur,
    "missing_hole": _inject_missing_hole,
    "short": _inject_short,
    "open_circuit": _inject_open_circuit,
    "spurious_copper": _inject_spurious_copper,
}


# Below roughly 40px on a 4K frame a defect survives the 1920 -> 640 letterbox as
# fewer than 14 pixels, which is under YOLOv8's smallest useful stride. Boxes are
# floored here so ground truth stays something a detector could plausibly find -
# an unfindable label would depress recall for reasons that have nothing to do
# with the language being measured.
MIN_BOX = 40


def _box(cx: int, cy: int, w: int, h: int, cls: str) -> Detection:
    """Ground-truth box, centred on the defect. Confidence 1.0 marks it as truth."""
    w, h = max(MIN_BOX, int(w)), max(MIN_BOX, int(h))
    return Detection(int(cx - w // 2), int(cy - h // 2), w, h,
                     1.0, CLASS_INDEX[cls], -1)


class SyntheticBoard:
    """Cached clean board plus on-demand defect injection.

    The clean render costs ~1s at 4K, so it happens once. Each new frame is a
    copy of the cached board with a few small patches painted on, which keeps
    frame generation far below inference cost and out of the measurement.
    """

    def __init__(self, width: int = 3840, height: int = 2160, seed: int = 2601):
        self.width, self.height, self.seed = width, height, seed
        self._clean, self._geom = render_board(width, height, seed)
        self._scale = width / 3840.0

    @property
    def clean(self) -> np.ndarray:
        """The golden reference: the same board with no defects."""
        return self._clean

    def frame(self, n_defects: int = 7, seed: Optional[int] = None
              ) -> Tuple[np.ndarray, List[Detection]]:
        """A board with n_defects planted, plus their ground-truth boxes."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        img = self._clean.copy()
        truth: List[Detection] = []
        names = list(_INJECTORS)

        attempts = 0
        while len(truth) < n_defects and attempts < n_defects * 8:
            attempts += 1
            fn = _INJECTORS[names[int(rng.integers(len(names)))]]
            box = fn(img, rng, self._geom, self._scale)
            if box is None:
                continue
            box.x = max(0, min(self.width - box.w, box.x))
            box.y = max(0, min(self.height - box.h, box.y))
            truth.append(box)

        return np.ascontiguousarray(img), truth

    def sequence(self, count: int, n_defects: int = 7):
        """Deterministic frame stream - frame k is always the same board.

        A benchmark must be able to re-run the identical input, and an operator
        demo wants the view to change between frames. A seed derived from the
        index gives both.
        """
        for k in range(count):
            yield self.frame(n_defects, seed=self.seed + 1000 + k)
