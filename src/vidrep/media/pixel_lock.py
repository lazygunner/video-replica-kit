"""
General-purpose Pixel-Lock for edited scene reference frames.
Automatically locates the replaced subject + shadow region via frame difference
(zero hardcoded coordinates), matches collar color, and feathers onto untouched original pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _auto_detect_edit_ellipse(orig: np.ndarray, edited: np.ndarray, pad_scale: float = 1.45) -> tuple[tuple[int, int], tuple[int, int]]:
    """Detect bounding ellipse around the largest coherent edited region (subject + shadow)."""
    h, w = orig.shape[:2]

    # Global coarse color normalization so mild full-frame tint shift from imagegen doesn't trigger mask
    ed_norm = edited.astype(np.float32).copy()
    for c in range(3):
        mo, so = float(np.median(orig[:, :, c])), float(orig[:, :, c].std()) + 1e-5
        me, se = float(np.median(ed_norm[:, :, c])), float(ed_norm[:, :, c].std()) + 1e-5
        ed_norm[:, :, c] = np.clip((ed_norm[:, :, c] - me) * (so / se) + mo, 0, 255)

    diff = np.mean(np.abs(ed_norm - orig.astype(np.float32)), axis=2)
    smooth = cv2.GaussianBlur(diff, (31, 31), 10)
    thr = max(18.0, float(np.percentile(smooth, 92)))
    bw = (smooth > thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if num_labels <= 1:
        return (w // 2, int(h * 0.65)), (int(w * 0.22), int(h * 0.26))

    # Keep components whose area is at least 15% of the largest changed component (captures detached cast shadow too)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    keep = np.zeros_like(bw)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= max(200, int(max_area * 0.15)):
            keep[labels == lbl] = 1

    ys, xs = np.where(keep > 0)
    if len(xs) < 100:
        return (w // 2, int(h * 0.65)), (int(w * 0.22), int(h * 0.26))

    x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
    y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    rx = int(np.clip(((x1 - x0) / 2.0) * pad_scale, w * 0.10, w * 0.42))
    ry = int(np.clip(((y1 - y0) / 2.0) * pad_scale, h * 0.12, h * 0.45))
    return (cx, cy), (rx, ry)


def pixel_lock_reference(
    orig_path: Path,
    edited_path: Path,
    out_path: Path,
    center: tuple[int, int] | None = None,
    axes: tuple[int, int] | None = None,
    pad_scale: float = 1.45,
) -> dict[str, Any]:
    """
    Composite ONLY the edited subject/shadow ellipse onto 100% untouched original pixels.
    Eliminates full-frame color/texture drift introduced by image-editing models.
    """
    fo = cv2.imread(str(orig_path))
    fe = cv2.imread(str(edited_path))
    if fo is None:
        raise FileNotFoundError(f"Could not read original keyframe: {orig_path}")
    if fe is None:
        raise FileNotFoundError(f"Could not read edited keyframe: {edited_path}")

    h, w = fo.shape[:2]
    if fe.shape[:2] != (h, w):
        fe = cv2.resize(fe, (w, h), interpolation=cv2.INTER_AREA)

    auto_used = center is None or axes is None
    det_center, det_axes = _auto_detect_edit_ellipse(fo, fe, pad_scale=pad_scale)
    c_use = center or det_center
    a_use = axes or det_axes

    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(outer, c_use, (int(a_use[0] * 1.35), int(a_use[1] * 1.35)), 0, 0, 360, 255, -1)
    cv2.ellipse(inner, c_use, (int(a_use[0] * 0.85), int(a_use[1] * 0.85)), 0, 0, 360, 255, -1)
    ring = (outer > 0) & (inner == 0)

    fe_cm = fe.astype(np.float32)
    if int(np.sum(ring)) > 300:
        for ch in range(3):
            mo, so = float(fo[:, :, ch][ring].mean()), float(fo[:, :, ch][ring].std()) + 1e-5
            me, se = float(fe[:, :, ch][ring].mean()), float(fe[:, :, ch][ring].std()) + 1e-5
            gain = float(np.clip(so / se, 0.82, 1.22))
            fe_cm[:, :, ch] = np.clip((fe_cm[:, :, ch] - me) * gain + mo, 0, 255)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, c_use, a_use, 0, 0, 360, 1.0, -1)
    ksize = int(np.clip((min(a_use) // 2) * 2 + 1, 51, 151))
    if ksize % 2 == 0:
        ksize += 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.30)
    mask3 = mask[:, :, None]

    blended = np.clip(fe_cm * mask3 + fo.astype(np.float32) * (1.0 - mask3), 0, 255).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)

    outer_bg = mask < 0.001
    outer_mae = float(np.mean(np.abs(blended[outer_bg].astype(np.float32) - fo[outer_bg].astype(np.float32)))) if np.any(outer_bg) else 0.0
    raw_outer_mae = float(np.mean(np.abs(fe[outer_bg].astype(np.float32) - fo[outer_bg].astype(np.float32)))) if np.any(outer_bg) else 0.0

    return {
        "output_path": str(out_path),
        "width": w,
        "height": h,
        "auto_detected": auto_used,
        "center": list(c_use),
        "axes": list(a_use),
        "outer_bg_mae_before": round(raw_outer_mae, 4),
        "outer_bg_mae_after": round(outer_mae, 4),
    }
