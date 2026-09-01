# aoi_gui/Report.py
"""
Inspection and benchmark report export.

Writes .xlsx when openpyxl is available and falls back to .csv otherwise, so a
missing optional dependency never costs the operator their data.

Derived cells (yield, pass rate, speedup) are written as real Excel formulas
rather than Python-computed literals, so the workbook still recalculates
correctly if someone edits a count by hand during review.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    _XLSX = True
except ImportError:
    _XLSX = False

FONT = "Arial"

_HEADER_FILL = "1E293B"
_HEADER_FONT = "FFFFFF"
_ACCENT_FILL = "E8EEF9"


def xlsx_available() -> bool:
    return _XLSX


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def export_report(out_dir: str | Path,
                  latest: Dict,
                  history: List[Dict],
                  summary: Optional[Dict] = None,
                  prefer_xlsx: bool = True,
                  prefix: str = "inspection_report") -> Path:
    """Write a report and return the path actually written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if prefer_xlsx and _XLSX:
        path = out_dir / f"{prefix}_{stamp}.xlsx"
        _write_xlsx(path, latest, history, summary or {})
        return path

    path = out_dir / f"{prefix}_{stamp}.csv"
    _write_csv(path, latest, history, summary or {})
    return path


# -----------------------------------------------------------------------------
# XLSX
# -----------------------------------------------------------------------------

def _style_header(ws, row: int, ncols: int):
    thin = Side(style="thin", color="94A3B8")
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color=_HEADER_FONT)
        cell.fill = PatternFill("solid", fgColor=_HEADER_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=thin)


def _autosize(ws, min_w=10, max_w=46):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        longest = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[letter].width = max(min_w, min(max_w, longest + 3))


def _write_xlsx(path: Path, latest: Dict, history: List[Dict], summary: Dict):
    wb = Workbook()

    # ---------------- Sheet 1: Inspection Summary ----------------
    ws = wb.active
    ws.title = "Inspection Summary"

    ws["A1"] = "Op-Ins 2601 - AOI Inspection Report"
    ws["A1"].font = Font(name=FONT, bold=True, size=14)
    ws["A2"] = f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}"
    ws["A2"].font = Font(name=FONT, italic=True, size=9, color="7B8794")

    row = 4
    ws.cell(row=row, column=1, value="Session Totals")
    ws.cell(row=row, column=1).font = Font(name=FONT, bold=True, size=11)
    row += 1

    ws.cell(row=row, column=1, value="Metric")
    ws.cell(row=row, column=2, value="Value")
    _style_header(ws, row, 2)
    header_row = row
    row += 1

    total = len(history)
    passed = sum(1 for h in history if h.get("total_defects", 0) == 0)
    first_data = row

    ws.cell(row=row, column=1, value="Boards inspected")
    ws.cell(row=row, column=2, value=total)
    boards_cell = f"B{row}"
    row += 1

    ws.cell(row=row, column=1, value="Boards passed")
    ws.cell(row=row, column=2, value=passed)
    passed_cell = f"B{row}"
    row += 1

    ws.cell(row=row, column=1, value="Boards failed")
    # Formula, not a literal: edit either count above and this follows.
    ws.cell(row=row, column=2, value=f"={boards_cell}-{passed_cell}")
    row += 1

    ws.cell(row=row, column=1, value="Inspection yield")
    ws.cell(row=row, column=2,
            value=f"=IFERROR({passed_cell}/{boards_cell},0)")
    ws.cell(row=row, column=2).number_format = "0.0%"
    ws.cell(row=row, column=2).fill = PatternFill("solid", fgColor=_ACCENT_FILL)
    row += 1

    ws.cell(row=row, column=1, value="Total defects found")
    ws.cell(row=row, column=2,
            value=sum(h.get("total_defects", 0) for h in history))
    row += 2

    ws.cell(row=row, column=1, value="Latest Inspection")
    ws.cell(row=row, column=1).font = Font(name=FONT, bold=True, size=11)
    row += 1
    ws.cell(row=row, column=1, value="Metric")
    ws.cell(row=row, column=2, value="Value")
    _style_header(ws, row, 2)
    row += 1

    for label, key, fmt in (
        ("Backend", "backend_desc", None),
        ("Engine", "engine", None),
        ("Total defects", "total_defects", "0"),
        ("Top defect type", "top_defect_type", None),
        ("Latency (ms)", "latency_ms", "0.00"),
        ("Effective FPS", "fps", "0.00"),
        ("SAHI tiles/frame", "tiles", "0"),
        ("Preprocess (ms)", "preprocess_ms", "0.00"),
        ("Inference (ms)", "inference_ms", "0.00"),
        ("Postprocess (ms)", "postprocess_ms", "0.00"),
        ("CPU load (%)", "cpu_percent", "0.0"),
        ("GPU load (%)", "gpu_percent", "0.0"),
        ("Memory RSS (MB)", "rss_mb", "0"),
    ):
        if key not in latest:
            continue
        ws.cell(row=row, column=1, value=label)
        c = ws.cell(row=row, column=2, value=latest.get(key))
        if fmt:
            c.number_format = fmt
        row += 1

    counts = latest.get("defect_counts") or {}
    if counts:
        row += 1
        ws.cell(row=row, column=1, value="Defect Breakdown (latest board)")
        ws.cell(row=row, column=1).font = Font(name=FONT, bold=True, size=11)
        row += 1
        ws.cell(row=row, column=1, value="Defect type")
        ws.cell(row=row, column=2, value="Count")
        _style_header(ws, row, 2)
        row += 1
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            ws.cell(row=row, column=1, value=name)
            ws.cell(row=row, column=2, value=n)
            row += 1

    for r in ws.iter_rows(min_row=1, max_row=row):
        for c in r:
            if c.font is None or not c.font.bold:
                c.font = Font(name=FONT)
    _autosize(ws)

    # ---------------- Sheet 2: Benchmark Matrix ----------------
    py = summary.get("python") or {}
    cpp = summary.get("cpp") or {}
    if (py and "error" not in py) or (cpp and "error" not in cpp):
        ws2 = wb.create_sheet("Benchmark Matrix")
        ws2["A1"] = "Comparative Performance Matrix"
        ws2["A1"].font = Font(name=FONT, bold=True, size=14)
        ws2["A2"] = "Same frame, same thresholds, same tile grid on both paths."
        ws2["A2"].font = Font(name=FONT, italic=True, size=9, color="7B8794")

        hrow = 4
        for col, text in enumerate(
                ["Metric", "Test Path A (Python)", "Test Path B (C++)", "Ratio A/B"], start=1):
            ws2.cell(row=hrow, column=col, value=text)
        _style_header(ws2, hrow, 4)

        rows = [
            ("Backend", "backend_desc", None, False),
            ("Frames measured", "frames", "0", False),
            ("SAHI tiles/frame", "tiles", "0", False),
            ("Mean latency (ms)", "mean_ms", "0.00", True),
            ("p50 latency (ms)", "p50_ms", "0.00", True),
            ("p95 latency (ms)", "p95_ms", "0.00", True),
            ("Effective FPS", "fps", "0.00", True),
            ("Preprocess (ms)", "preprocess_ms", "0.00", True),
            ("Inference (ms)", "inference_ms", "0.00", True),
            ("Postprocess (ms)", "postprocess_ms", "0.00", True),
            ("CPU load (%)", "cpu_percent", "0.0", False),
            ("GPU load (%)", "gpu_percent", "0.0", False),
            ("Memory RSS (MB)", "rss_mb", "0", True),
            ("Peak RSS (MB)", "peak_rss_mb", "0", True),
            ("Detections/frame", "detections", "0.0", False),
        ]

        r = hrow + 1
        for label, key, fmt, ratio in rows:
            ws2.cell(row=r, column=1, value=label)
            a = ws2.cell(row=r, column=2, value=py.get(key, "n/a"))
            b = ws2.cell(row=r, column=3, value=cpp.get(key, "n/a"))
            if fmt:
                a.number_format = fmt
                b.number_format = fmt
            if ratio and isinstance(py.get(key), (int, float)) and isinstance(cpp.get(key), (int, float)):
                # Guard the denominator - a zero would otherwise ship #DIV/0!
                c = ws2.cell(row=r, column=4,
                             value=f"=IFERROR(B{r}/C{r},\"n/a\")")
                c.number_format = "0.00x"
            r += 1

        r += 1
        ws2.cell(row=r, column=1, value="Headline speedup (mean latency)")
        ws2.cell(row=r, column=1).font = Font(name=FONT, bold=True)
        mean_row = hrow + 1 + [x[1] for x in rows].index("mean_ms")
        sp = ws2.cell(row=r, column=2, value=f"=IFERROR(B{mean_row}/C{mean_row},\"n/a\")")
        sp.number_format = "0.00x"
        sp.fill = PatternFill("solid", fgColor=_ACCENT_FILL)
        sp.font = Font(name=FONT, bold=True)

        r += 1
        ws2.cell(row=r, column=1, value="Note")
        ws2.cell(row=r, column=2, value=(
            "Path A runs .pt via PyTorch; Path B runs .onnx via OpenCV DNN. "
            "Part of the gap is runtime, not language."))
        ws2.cell(row=r, column=2).font = Font(name=FONT, italic=True, size=9)

        for rr in ws2.iter_rows(min_row=1, max_row=r):
            for c in rr:
                if c.font is None or not c.font.bold:
                    c.font = Font(name=FONT, italic=bool(c.font and c.font.italic))
        _autosize(ws2)

    # ---------------- Sheet 3: Per-frame history ----------------
    if history:
        ws3 = wb.create_sheet("Per-Frame Data")
        cols = ["timestamp", "engine", "total_defects", "top_defect_type",
                "latency_ms", "fps", "tiles", "preprocess_ms", "inference_ms",
                "postprocess_ms", "cpu_percent", "gpu_percent", "rss_mb"]
        for i, name in enumerate(cols, start=1):
            ws3.cell(row=1, column=i, value=name)
        _style_header(ws3, 1, len(cols))

        for r, h in enumerate(history, start=2):
            for i, name in enumerate(cols, start=1):
                ws3.cell(row=r, column=i, value=h.get(name))
        ws3.freeze_panes = "A2"
        for rr in ws3.iter_rows(min_row=2):
            for c in rr:
                c.font = Font(name=FONT)
        _autosize(ws3)

    wb.save(path)


# -----------------------------------------------------------------------------
# CSV fallback
# -----------------------------------------------------------------------------

def _write_csv(path: Path, latest: Dict, history: List[Dict], summary: Dict):
    total = len(history)
    passed = sum(1 for h in history if h.get("total_defects", 0) == 0)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Op-Ins 2601 - AOI Inspection Report"])
        w.writerow(["generated", datetime.now().isoformat(timespec="seconds")])
        w.writerow([])

        w.writerow(["section", "metric", "value"])
        w.writerow(["session", "boards_inspected", total])
        w.writerow(["session", "boards_passed", passed])
        w.writerow(["session", "boards_failed", total - passed])
        w.writerow(["session", "inspection_yield",
                    f"{(passed / total * 100 if total else 100):.1f}%"])
        w.writerow(["session", "total_defects",
                    sum(h.get("total_defects", 0) for h in history)])

        for k in ("backend_desc", "engine", "total_defects", "top_defect_type",
                  "latency_ms", "fps", "tiles", "preprocess_ms", "inference_ms",
                  "postprocess_ms", "cpu_percent", "gpu_percent", "rss_mb"):
            if k in latest:
                w.writerow(["latest", k, latest[k]])

        for name, n in (latest.get("defect_counts") or {}).items():
            w.writerow(["defect_count", name, n])

        for engine in ("python", "cpp"):
            d = summary.get(engine) or {}
            for k, v in d.items():
                w.writerow([f"benchmark_{engine}", k, v])
        if "speedup" in summary:
            w.writerow(["benchmark", "speedup_mean_latency", f"{summary['speedup']:.3f}"])

        if history:
            w.writerow([])
            cols = ["timestamp", "engine", "total_defects", "top_defect_type",
                    "latency_ms", "fps", "tiles", "cpu_percent", "rss_mb"]
            w.writerow(cols)
            for h in history:
                w.writerow([h.get(c) for c in cols])
