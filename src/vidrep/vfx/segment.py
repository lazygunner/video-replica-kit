"""
General-purpose subject + cast-shadow + old-subject-erasure segmentation masks.
Supports optional local ONNX refinement:
  - SAM 2.1 Hiera Tiny (`use_sam2=True`) for razor-sharp semantic subject matting
  - Depth Anything V2 Small (`use_depth=True`) for automatic foreground occluder subtraction
"""

from __future__ import annotations

import sys
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


def _extract_subject_prompts(
    binary_small: np.ndarray,
    score_small: np.ndarray,
    full_w: int,
    full_h: int,
    max_boxes: int = 6,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
    """Extract multi-scale subject boxes AND peak-diff interior anchor points (handles giant + tiny subjects)."""
    sh, sw = binary_small.shape[:2]
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary_small)
    if num_lbl <= 1:
        return [], []
    min_keep = max(45, int(sw * sh * 0.0005))
    sx, sy = full_w / sw, full_h / sh
    cands: list[tuple[int, tuple[float, float, float, float], tuple[float, float]]] = []
    for lbl in range(1, num_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_keep:
            continue
        x, y, bw, bh = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
        pad_x, pad_y = max(3.0, bw * 0.08), max(3.0, bh * 0.08)
        x0 = max(0.0, (x - pad_x) * sx)
        y0 = max(0.0, (y - pad_y) * sy)
        x1 = min(float(full_w - 1), (x + bw + pad_x) * sx)
        y1 = min(float(full_h - 1), (y + bh + pad_y) * sy)
        # Anchor positive point on the strongest semantic diff pixel inside this component (never empty hollow/bg)
        comp_scores = np.where(labels == lbl, score_small, -1.0)
        py_s, px_s = np.unravel_index(int(np.argmax(comp_scores)), comp_scores.shape)
        cands.append((area, (x0, y0, x1, y1), ((px_s + 0.5) * sx, (py_s + 0.5) * sy)))
    cands.sort(key=lambda t: t[0], reverse=True)
    cands = cands[:max_boxes]
    return [c[1] for c in cands], [c[2] for c in cands]


def compute_replacement_masks(
    orig_matched: list[np.ndarray],
    gen_aligned: list[np.ndarray],
    bg_weight_small: np.ndarray,
    mode: str = "auto_contour",
    diff_threshold: float = 20.0,
    dilate_frac: float = 0.032,
    use_sam2: bool = False,
    use_depth: bool = False,
    onnx_stride: int = 0,
    depth_margin: float = 0.14,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Compute temporally smooth soft alpha masks in [0, 1] covering ALL replaced regions
    (old subject erasure + new subject(s) + cast shadows), optionally refined with:
      - SAM 2.1 Tiny ONNX (`use_sam2=True`)
      - Depth Anything V2 Small ONNX foreground occlusion protection (`use_depth=True`)
    """
    n = len(gen_aligned)
    h, w = orig_matched[0].shape[:2]
    bg_mask = cv2.resize(bg_weight_small, (w, h)) > 0.5
    if int(bg_mask.sum()) < h * w * 0.15:
        bg_mask = np.ones((h, w), dtype=bool)

    work_w, work_h = 480, max(120, int(round(480 * h / w)))
    dil_px = max(7, int(round(work_w * dilate_frac * 0.72)) * 2 + 1)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dil_px * 1.25) | 1, int(dil_px * 1.25) | 1))

    raw_masks_small: list[np.ndarray] = []
    clean_bins_small: list[np.ndarray] = []
    scores_small: list[np.ndarray] = []
    centroids = np.zeros((n, 2), dtype=np.float64)
    extents = np.zeros((n, 2), dtype=np.float64)
    coverages: list[float] = []

    for i in range(n):
        fo = orig_matched[i]
        fg_norm = _normalize_frame_to_ref(gen_aligned[i], fo, bg_mask)

        fo_s = cv2.resize(fo, (work_w, work_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        fg_s = cv2.resize(fg_norm, (work_w, work_h), interpolation=cv2.INTER_AREA)

        # Dual-scale semantic difference:
        #   - Coarse (17x17) cancels high-frequency foliage/spiral/reed jitter & catches broad shaded faces/shadows
        #   - Medium-fine (7x7) preserves small/thin secondary subjects (distant dancers, water-sleeves, tentacles)
        fo_c = cv2.GaussianBlur(fo_s, (17, 17), 5.0)
        fg_c = cv2.GaussianBlur(fg_s, (17, 17), 5.0)
        rgb_c = np.mean(np.abs(fg_c - fo_c), axis=2)
        lab_oc = cv2.cvtColor(np.clip(fo_c, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_gc = cv2.cvtColor(np.clip(fg_c, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_c = np.sqrt(
            0.45 * (lab_gc[:, :, 0] - lab_oc[:, :, 0]) ** 2
            + 1.35 * (lab_gc[:, :, 1] - lab_oc[:, :, 1]) ** 2
            + 1.35 * (lab_gc[:, :, 2] - lab_oc[:, :, 2]) ** 2
        )
        score_c = 0.50 * rgb_c + 0.50 * lab_c

        fo_f = cv2.GaussianBlur(fo_s, (7, 7), 2.0)
        fg_f = cv2.GaussianBlur(fg_s, (7, 7), 2.0)
        rgb_f = np.mean(np.abs(fg_f - fo_f), axis=2)
        lab_of = cv2.cvtColor(np.clip(fo_f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_gf = cv2.cvtColor(np.clip(fg_f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_f = np.sqrt(
            0.45 * (lab_gf[:, :, 0] - lab_of[:, :, 0]) ** 2
            + 1.35 * (lab_gf[:, :, 1] - lab_of[:, :, 1]) ** 2
            + 1.35 * (lab_gf[:, :, 2] - lab_of[:, :, 2]) ** 2
        )
        score_f = 0.50 * rgb_f + 0.50 * lab_f

        score = cv2.GaussianBlur(np.maximum(score_c, 0.85 * score_f), (5, 5), 1.4)
        scores_small.append(score)

        # Robust lower-half background noise floor (remains accurate even when giant subjects cover >30% of frame)
        # scaled softly by scene dynamic range so both sunlit outdoor and low-key dark scenes segment cleanly.
        p60 = float(np.percentile(score, 60))
        lower = score[score <= p60]
        bg_med = float(np.median(lower)) if lower.size else float(np.median(score))
        sigma_bg = float(1.4826 * np.median(np.abs(lower - bg_med))) if lower.size else 1.5
        s_peak = float(np.percentile(score, 99.6))
        contrast_scale = float(np.clip(np.std(fo_s) / 45.0, 0.42, 1.30))

        thr_high = max(diff_threshold * 0.65 * contrast_scale, bg_med + 5.8 * sigma_bg, s_peak * 0.28)
        thr_low = max(diff_threshold * 0.30 * contrast_scale, bg_med + 3.4 * sigma_bg, s_peak * 0.14)

        # Hysteresis segmentation: keep every multi-scale component (giant monster + tiny human + tentacles + reflections)
        # whose own peak exceeds thr_high, without penalizing small subjects next to giant ones.
        bw_low = (score > thr_low).astype(np.uint8)
        bw_low = cv2.morphologyEx(bw_low, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bw_low = cv2.morphologyEx(bw_low, cv2.MORPH_CLOSE, close_k)

        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bw_low)
        clean = np.zeros_like(bw_low)
        min_comp_area = max(45, int(work_w * work_h * 0.0005))
        if num_lbl > 1:
            for lbl in range(1, num_lbl):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < min_comp_area:
                    continue
                comp_mask = labels == lbl
                peak95 = float(np.percentile(score[comp_mask], 95))
                mean_s = float(np.mean(score[comp_mask]))
                if peak95 >= thr_high and mean_s >= thr_low * 1.15:
                    clean[comp_mask] = 255

        clean_bins_small.append(clean)
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
        stack = np.stack(raw_masks_small, axis=0)
        weights = np.array([0.12, 0.22, 0.32, 0.22, 0.12], dtype=np.float32)
        padded = np.pad(stack, ((2, 2), (0, 0), (0, 0)), mode="edge")
        blur_k = max(21, int(round(work_w * 0.08)) | 1)
        for i in range(n):
            tm = np.tensordot(weights, padded[i : i + 5], axes=(0, 0))
            tm = np.clip(tm * 1.15, 0.0, 1.0)
            tm = cv2.GaussianBlur(tm, (blur_k, blur_k), blur_k * 0.28)
            final_masks.append(cv2.resize(tm, (w, h), interpolation=cv2.INTER_LINEAR))

    # --- Optional Local ONNX Refinement (SAM 2.1 Tiny Matting + Depth Anything V2 Occlusion Guard) ---
    onnx_meta: dict[str, Any] = {"use_sam2": use_sam2, "use_depth": use_depth}
    onnx_diag: dict[str, Any] = {}

    if use_sam2 or use_depth:
        from .onnx_matting import DepthAnythingV2Onnx, Sam2OnnxMatte, propagate_masks_optical_flow

        stride = onnx_stride if onnx_stride >= 1 else (1 if n <= 24 else (4 if n <= 120 else 6))
        kf_indices = sorted(set(list(range(0, n, stride)) + [n - 1]))
        onnx_meta["onnx_stride"] = stride
        onnx_meta["onnx_keyframes_evaluated"] = len(kf_indices)

        sam2_full: list[np.ndarray] | None = None
        if use_sam2:
            print(f"[vidrep] running SAM 2.1 Tiny ONNX on {len(kf_indices)}/{n} keyframes (stride={stride})...", file=sys.stderr, flush=True)
            sam = Sam2OnnxMatte(auto_download=True)
            sam_kfs: dict[int, np.ndarray] = {}
            ious: list[float] = []
            for gi in kf_indices:
                boxes, pts = _extract_subject_prompts(clean_bins_small[gi], scores_small[gi], w, h)
                m_sam, iou = sam.segment_boxes(gen_aligned[gi], boxes, anchor_points=pts)
                sam_kfs[gi] = m_sam
                if iou > 0:
                    ious.append(iou)
            sam2_full = propagate_masks_optical_flow(gen_aligned, sam_kfs)
            onnx_meta["sam2_mean_iou"] = round(float(np.mean(ious)), 4) if ious else 0.0
            onnx_diag["sam2_masks"] = sam2_full

            # Combine crisp SAM2 subject core (feather 9px) with softer outer erasure/shadow envelope
            for i in range(n):
                crisp_fg = cv2.GaussianBlur(sam2_full[i], (9, 9), 2.5)
                # Tighten outer diff envelope slightly when SAM2 gives high-confidence subject edges
                tight_env = np.clip((final_masks[i] - 0.12) / 0.88, 0.0, 1.0)
                final_masks[i] = np.maximum(crisp_fg, tight_env)

        if use_depth:
            print(f"[vidrep] running Depth Anything V2 ONNX on {len(kf_indices)}/{n} keyframes (stride={stride})...", file=sys.stderr, flush=True)
            dep = DepthAnythingV2Onnx(auto_download=True)
            occ_kfs: dict[int, np.ndarray] = {}
            depth_samples: dict[int, np.ndarray] = {}
            occ_ratios: list[float] = []
            for gi in kf_indices:
                core = sam2_full[gi] if sam2_full is not None else (final_masks[gi] > 0.55).astype(np.float32)
                occ_m, d_map, ratio = dep.detect_foreground_occluders(
                    orig_matched[gi], core, final_masks[gi], depth_margin=depth_margin
                )
                occ_kfs[gi] = occ_m
                depth_samples[gi] = d_map
                occ_ratios.append(ratio)

            occ_full = propagate_masks_optical_flow(orig_matched, occ_kfs)
            for i in range(n):
                # Subtract closer original foreground occluders so poles/trees stay IN FRONT of the replaced subject
                final_masks[i] = np.clip(final_masks[i] * (1.0 - occ_full[i]), 0.0, 1.0)

            onnx_meta["depth_margin"] = depth_margin
            onnx_meta["mean_occlusion_ratio"] = round(float(np.mean(occ_ratios)), 4)
            onnx_meta["max_occlusion_ratio"] = round(float(np.max(occ_ratios)), 4) if occ_ratios else 0.0
            onnx_diag["depth_samples"] = depth_samples
            onnx_diag["occluder_masks"] = occ_full

    mean_cov = float(np.mean([float(np.mean(m > 0.15)) for m in final_masks]))
    max_cov = float(np.max([float(np.mean(m > 0.15)) for m in final_masks])) if final_masks else 0.0
    recommend_graded_fallback = mean_cov > 0.62 or max_cov > 0.78

    meta = {
        "mask_mode": mode,
        "mean_mask_coverage": round(mean_cov, 4),
        "max_mask_coverage": round(max_cov, 4),
        "recommend_graded_fallback": recommend_graded_fallback,
        "fallback_reason": (
            "Over 62% of frame area changed (e.g., full-scene restyle, extreme camera drift, or heavy occlusion); "
            "review graded deliverable alongside composited deliverable."
            if recommend_graded_fallback
            else None
        ),
        "onnx": onnx_meta,
        "onnx_diag": onnx_diag,
        "centroids": centroids_s.tolist(),
    }
    return final_masks, meta
