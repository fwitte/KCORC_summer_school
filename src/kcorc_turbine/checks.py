from __future__ import annotations

import math

from IPython.display import HTML, display


def checkpoint(
    name: str,
    value: float,
    expected_range: tuple[float, float],
    unit: str = "",
    digits: int = 4,
) -> bool:
    """Display a compact, unit-aware checkpoint for a student calculation."""
    lo, hi = expected_range
    finite = isinstance(value, (int, float)) and math.isfinite(float(value))
    ok = finite and lo <= float(value) <= hi
    symbol = "✓" if ok else "✗"
    background = "#eaf7ee" if ok else "#fff1e8"
    border = "#2e7d32" if ok else "#c45100"
    shown = f"{float(value):.{digits}g}" if finite else repr(value)
    display(
        HTML(
            f"<div style='padding:0.65rem 0.85rem;margin:0.35rem 0;"
            f"background:{background};border-left:5px solid {border};border-radius:4px'>"
            f"<b>{symbol} {name}</b>: {shown} {unit} &nbsp; "
            f"<span style='color:#555'>(expected {lo:g}–{hi:g} {unit})</span></div>"
        )
    )
    return ok


def compare_with_reference(
    name: str,
    student_value: float,
    reference_value: float,
    relative_tolerance: float = 0.02,
    unit: str = "",
) -> bool:
    if not math.isfinite(float(student_value)):
        ok = False
        error = float("inf")
    else:
        scale = max(abs(float(reference_value)), 1e-12)
        error = abs(float(student_value) - float(reference_value)) / scale
        ok = error <= relative_tolerance
    symbol = "✓" if ok else "✗"
    background = "#eaf7ee" if ok else "#fff1e8"
    border = "#2e7d32" if ok else "#c45100"
    display(
        HTML(
            f"<div style='padding:0.65rem 0.85rem;margin:0.35rem 0;"
            f"background:{background};border-left:5px solid {border};border-radius:4px'>"
            f"<b>{symbol} {name}</b>: student {student_value:.5g} {unit}, "
            f"reference {reference_value:.5g} {unit}, relative error {100*error:.2f}%</div>"
        )
    )
    return ok
