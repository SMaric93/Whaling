from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OCRTextBox:
    """A text box with polygon coordinates in page/image space."""

    points: tuple[tuple[float, float], ...]
    text: str
    confidence: float | None = None


def _x_left(box: OCRTextBox) -> float:
    return min(point[0] for point in box.points)


def _y_center(box: OCRTextBox) -> float:
    ys = [point[1] for point in box.points]
    return (min(ys) + max(ys)) / 2.0


def _height(box: OCRTextBox) -> float:
    ys = [point[1] for point in box.points]
    return max(ys) - min(ys)


def ocr_boxes_to_lines(
    boxes: Iterable[OCRTextBox],
    *,
    row_sort_descending: bool = False,
    min_row_threshold: float = 0.0,
) -> list[str]:
    """Group OCR boxes into reading-order lines."""
    clean_boxes = [
        box
        for box in boxes
        if box.points and str(box.text).strip()
    ]
    if not clean_boxes:
        return []

    row_threshold = max(
        min_row_threshold,
        statistics.median(_height(box) for box in clean_boxes) * 0.6,
    )
    row_key = (
        (lambda box: (-_y_center(box), _x_left(box)))
        if row_sort_descending
        else (lambda box: (_y_center(box), _x_left(box)))
    )
    sorted_boxes = sorted(clean_boxes, key=row_key)

    rows: list[dict[str, object]] = []
    for box in sorted_boxes:
        center = _y_center(box)
        if not rows or abs(center - float(rows[-1]["center"])) > row_threshold:
            rows.append({"center": center, "items": [box]})
            continue

        current_row = rows[-1]
        row_items = current_row["items"]
        assert isinstance(row_items, list)
        row_items.append(box)
        current_row["center"] = (
            float(current_row["center"]) * (len(row_items) - 1) + center
        ) / len(row_items)

    lines: list[str] = []
    for row in rows:
        row_items = row["items"]
        assert isinstance(row_items, list)
        line = " ".join(
            box.text.strip()
            for box in sorted(row_items, key=_x_left)
            if box.text.strip()
        ).strip()
        if line:
            lines.append(line)
    return lines
