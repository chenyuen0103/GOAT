from __future__ import annotations

import math
from typing import Iterable, Sequence


def accuracy_percent(correct: int, total: int) -> float:
    if total <= 0:
        return math.nan
    return 100.0 * float(correct) / float(total)


def risk_from_accuracy_percent(accuracy: float) -> float:
    return 1.0 - float(accuracy) / 100.0


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(value) for value in values]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)
    return mean, math.sqrt(variance)


def final_risks_from_accuracy_curves(curves: Sequence[Sequence[float]]) -> list[float]:
    return [risk_from_accuracy_percent(curve[-1]) for curve in curves if curve]

