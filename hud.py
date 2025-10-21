"""Heads-up display helpers used to draw runtime statistics."""
from __future__ import annotations

import os
from typing import Iterable, List
import cv2

COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)


def put_text_multiline(img, text_lines: Iterable[str], org=(10, 10), line_h=22, font_scale=0.55,
                        color=COLOR_WHITE, thickness=1) -> None:
    """Render a list of strings on top of the provided frame."""

    x, y = org
    for i, line in enumerate(text_lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_hud(frame, stats_lines: List[str], margin: int = 5, padding: int = 10, alpha: float = 0.35) -> None:
    """Draw a translucent box containing ``stats_lines``."""

    font_scale = 0.55
    thickness = 1
    line_h = 22
    box_w = 0
    for ln in stats_lines:
        (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        box_w = max(box_w, tw)
    box_h = padding * 2 + line_h * len(stats_lines)
    box_w = padding * 2 + box_w
    x1, y1 = margin, margin
    x2, y2 = x1 + box_w, y1 + box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BLUE, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    put_text_multiline(
        frame,
        stats_lines,
        org=(x1 + padding, y1 + padding),
        line_h=line_h,
        font_scale=font_scale,
        color=COLOR_WHITE,
        thickness=1,
    )


def natural_key(path: str):
    """Return a key useful for natural sorting of file paths."""

    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(path))]
