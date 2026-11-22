"""
General-purpose subject + cast-shadow + old-subject-erasure segmentation masks.
Zero hardcoded colors or coordinates; supports single AND multi-subject shots
(e.g., character + creature replaced simultaneously) with temporal stability smoothing.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .align import gaussian_smooth_1d


def _normalize_frame_to_ref(fg: np.ndarray, fo: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    out = fg.astype(np.float32).copy()
    for c in range(3):
        mo, so = float(fo[:, :, c][bg_mask].mean()), float(fo[:, :, c][bg_mask].std()) + 1e-5
        mg, sg = float(out[:, :, c][bg_mask].mean()), float(out[:, :, c][bg_mask].std()) + 1e-5
        gain = float(np.clip(so / sg, 0.85, 1.18))
        out[:, :, c] = np.clip((out[:, :, c] - mg) * gain + mo, 0, 255)
    return out


def compute_replacement_masks(
    orig_matched: list[np.ndarray],
    gen_aligned: list[np.ndarray],
    bg_weight_small: np.ndarray,
    mode: str = "auto_contour",
    diff_threshold: float = 20.0,
    dilate_frac: float = 0.032,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Compute temporally smooth soft alpha masks in [0, 1] covering ALL replaced regions
    (old subject erasure + new subject(s) + cast shadows) without relying on specific hues or paths.

    Modes:
      - 'auto_contour' (default): multi-subject coherent difference blobs + dilation + temporal smoothing.
      - 'ellipse': smooth single bounding ellipse around primary changed blob (ideal for tight single-subject paths).
    """
    n = len(gen_aligned)
    h, w = orig_matched[0].shape[:2]
    bg_mask = cv2.resize(bg_weight_small, (w, h)) > 0.5
    if int(bg_mask.sum()) < h * w * 0.15:
        bg_mask = np.ones((h, w), dtype=bool)

    work_w, work_h = 480, max(120, int(round(480 * h / w)))
    dil_px = max(9, int(round(work_w * dilate_frac)) * 2 + 1)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dil_px * 1.45) | 1, int(dil_px * 1.45) | 1))

    raw_masks_small: list[np.ndarray] = []
    centroids = np.zeros((n, 2), dtype=np.float64)
    extents = np.zeros((n, 2), dtype=np.float64)
    coverages: list[float] = []

    for i in range(n):
        fo = orig_matched[i]
        fg_norm = _normalize_frame_to_ref(gen_aligned[i], fo, bg_mask)

        fo_s = cv2.resize(fo, (work_w, work_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        fg_s = cv2.resize(fg_norm, (work_w, work_h), interpolation=cv2.INTER_AREA)

        # Combine RGB difference + chroma difference (catches dark shadows AND hue swaps)
        rgb_diff = np.mean(np.abs(fg_s - fo_s), axis=2)
        lab_o = cv2.cvtColor(fo_s.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_g = cv2.cvtColor(np.clip(fg_s, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_diff = np.sqrt(np.sum((lab_g - lab_o) ** 2, axis=2)) * 0.65
        score = cv2.GaussianBlur(0.55 * rgb_diff + 0.45 * lab_diff, (15, 15), 4.5)

        # Adaptive threshold: keeps strong subject/shadow edits while suppressing minor diffusion edge noise
        p75 = float(np.percentile(score, 75))
        thr = max(diff_threshold, p75 + 10.0)
        bw = (score > thr).astype(np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_k)

        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
        clean = np.zeros_like(bw)
        if num_lbl > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = int(areas.max())
            min_keep = max(int(work_w * work_h * 0.004), int(max_area * 0.12))
            for lbl in range(1, num_lbl):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_keep:
                    clean[labels == lbl] = 255

        dilated = cv2.dilate(clean, dil_k)
        raw_masks_small.append(dilated.astype(np.float32) / 255.0)
        coverages.append(float(np.mean(dilated > 0)))

        ys, xs = np.where(dilated > 0)
        if len(xs) > 30:
            sx, sy = w / work_w, h / work_h
            centroids[i] = [float(np.mean(xs) * sx), float(np.mean(ys) * sy)]
            extents[i] = [
                max(w * 0.10, float(np.percentile(np.abs(xs - np.mean(xs)), 98) * sx * 1.15)),
                max(h * 0.12, float(np.percentile(np.abs(ys - np.mean(ys)), 98) * sy * 1.15)),
            ]
        else:
            centroids[i] = [w * 0.5, h * 0.6]
            extents[i] = [w * 0.18, h * 0.22]

    centroids_s = gaussian_smooth_1d(centroids, sigma=3.5)
    extents_s = gaussian_smooth_1d(extents, sigma=5.0)

    final_masks: list[np.ndarray] = []
    if mode == "ellipse":
        for i in range(n):
            m = np.zeros((h, w), dtype=np.float32)
            cx, cy = int(round(centroids_s[i, 0])), int(round(centroids_s[i, 1]))
            rx, ry = int(round(min(w * 0.42, extents_s[i, 0]))), int(round(min(h * 0.45, extents_s[i, 1])))
            cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            ks = int(np.clip((min(rx, ry) // 2) * 2 + 1, 51, 151)) | 1
            final_masks.append(cv2.GaussianBlur(m, (ks, ks), ks * 0.28))
    else:
        # Temporal Gaussian blend across neighboring frames (radius=2) prevents mask flicker!
        stack = np.stack(raw_masks_small, axis=0)  # (n, work_h, work_w)
        weights = np.array([0.12, 0.22, 0.32, 0.22, 0.12], dtype=np.float32)
        padded = np.pad(stack, ((2, 2), (0, 0), (0, 0)), mode="edge")
        blur_k = max(21, int(round(work_w * 0.08)) | 1)
        for i in range(n):
            tm = np.tensordot(weights, padded[i : i + 5], axes=(0, 0))
            tm = np.clip(tm * 1.15, 0.0, 1.0)
            tm = cv2.GaussianBlur(tm, (blur_k, blur_k), blur_k * 0.28)
            final_masks.append(cv2.resize(tm, (w, h), interpolation=cv2.INTER_LINEAR))

    mean_cov = float(np.mean(coverages))
    max_cov = float(np.max(coverages)) if coverages else 0.0
    recommend_graded_fallback = mean_cov > 0.62 or max_cov > 0.78

    meta = {
        "mask_mode": mode,
        "mean_mask_coverage": round(mean_cov, 4),
        "max_mask_coverage": round(max_cov, 4),
        "recommend_graded_fallback": recommend_graded_fallback,
        "fallback_reason": (
            "Over 62% of frame area changed (e.g., full-scene restyle, extreme camera drift, or heavy occlusion); "
            "review final-graded-with-audio.mp4 alongside final-composited-with-audio.mp4."
            if recommend_graded_fallback
            else None
        ),
        "centroids": centroids_s.tolist(),
    }
    return final_masks, meta
