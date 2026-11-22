"""
General-purpose temporal & spatial alignment (zero hardcoded top-half assumptions).
Supports arbitrary source FPS -> 24fps via monotonic DP, DIS optical-flow sub-frames,
and adaptive non-subject background ECC Euclidean registration.
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    if arr.ndim == 1:
        padded = np.pad(arr, radius, mode="edge")
        return np.convolve(padded, k, mode="valid")
    out = np.zeros_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        padded = np.pad(arr[:, col], radius, mode="edge")
        out[:, col] = np.convolve(padded, k, mode="valid")
    return out


def estimate_static_bg_weight_map(
    orig_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
    small_size: tuple[int, int] = (160, 90),
) -> np.ndarray:
    """
    Estimate where the background (unchanged area) lies across the shot by sampling
    8 proportional frame pairs. Returns a float32 weight map in [0.1, 1.0] where
    high-change subject regions are down-weighted automatically (works for top, center, or bottom subjects).
    """
    n_o, n_g = len(orig_frames), len(gen_frames)
    sw, sh = small_size
    acc_diff = np.zeros((sh, sw), dtype=np.float32)
    samples = min(8, n_g)
    for k in range(samples):
        frac = (k + 0.5) / samples
        gi = min(n_g - 1, int(round(frac * (n_g - 1))))
        oi = min(n_o - 1, int(round(frac * (n_o - 1))))
        fo = cv2.resize(orig_frames[oi], (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        fg = cv2.resize(gen_frames[gi], (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Normalize channel means so global tint shift does not hide localized subject diff
        for c in range(3):
            fg[:, :, c] += float(fo[:, :, c].mean() - fg[:, :, c].mean())
        acc_diff += np.mean(np.abs(fo - fg), axis=2)
    acc_diff /= max(1, samples)
    thr = float(np.percentile(acc_diff, 70))
    bg_weight = np.where(acc_diff <= thr, 1.0, 0.1).astype(np.float32)
    return bg_weight


def monotonic_temporal_match(
    orig_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
    fps_orig: float,
    fps_gen: float,
) -> tuple[list[int], np.ndarray]:
    """Monotonic dynamic programming alignment supporting any source/generated FPS ratio."""
    n_o, n_g = len(orig_frames), len(gen_frames)
    sw, sh = 160, 90
    bg_w = estimate_static_bg_weight_map(orig_frames, gen_frames, (sw, sh))
    bg_w3 = (bg_w / bg_w.mean())[:, :, None]

    orig_small = np.stack([
        cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) for f in orig_frames
    ])
    gen_small = np.stack([
        cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) for f in gen_frames
    ])

    cost = np.zeros((n_g, n_o), dtype=np.float32)
    for gi in range(n_g):
        diff = np.abs(orig_small - gen_small[gi][None]) * bg_w3[None]
        cost[gi] = np.mean(diff, axis=(1, 2, 3))

    expected_step = max(0.5, fps_orig / max(fps_gen, 1e-3))
    min_step = max(0, int(np.floor(expected_step)) - 1) if expected_step < 1.5 else max(1, int(np.floor(expected_step)) - 1)
    max_step = max(min_step + 2, int(np.ceil(expected_step)) + 2)
    steps = list(range(min_step, max_step + 1))

    INF = 1e12
    dp = np.full((n_g, n_o), INF, dtype=np.float64)
    parent = np.full((n_g, n_o), -1, dtype=np.int32)

    start_window = min(n_o, max(8, int(round(expected_step * 4))))
    for oi in range(start_window):
        dp[0, oi] = float(cost[0, oi])

    for gi in range(1, n_g):
        for oi in range(n_o):
            best_val = INF
            best_prev = -1
            for step in steps:
                po = oi - step
                if po < 0 or dp[gi - 1, po] >= INF:
                    continue
                pen = abs(step - expected_step) * 0.25
                val = dp[gi - 1, po] + float(cost[gi, oi]) + pen
                if val < best_val:
                    best_val = val
                    best_prev = po
            dp[gi, oi] = best_val
            parent[gi, oi] = best_prev

    last_oi = int(np.argmin(dp[n_g - 1]))
    path = [0] * n_g
    cur = last_oi
    for gi in range(n_g - 1, -1, -1):
        path[gi] = cur
        cur = int(parent[gi, cur]) if gi > 0 and parent[gi, cur] >= 0 else max(0, cur - 1)
    return path, bg_w


def build_subframe_orig(
    orig_frames: list[np.ndarray],
    match_idx: list[int],
    gen_frames: list[np.ndarray],
    bg_weight_small: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    """Synthesize crisp DIS optical-flow midpoints when generated frames fall halfway between source frames."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    n_o = len(orig_frames)
    h, w = orig_frames[0].shape[:2]
    sw, sh = bg_weight_small.shape[1], bg_weight_small.shape[0]
    w3 = (bg_weight_small / bg_weight_small.mean())[:, :, None]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    out: list[np.ndarray] = []
    half_count = 0
    for gi, oi in enumerate(match_idx):
        f0 = orig_frames[oi]
        fg = gen_frames[gi]
        g_s = cv2.resize(fg, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        f0_s = cv2.resize(f0, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        best_err = float(np.mean(np.abs(f0_s - g_s) * w3))
        best_pair: tuple[int, int] | None = None

        for a, b in ((oi - 1, oi), (oi, oi + 1)):
            if 0 <= a < n_o and 0 <= b < n_o:
                mid_s = 0.5 * (
                    cv2.resize(orig_frames[a], (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
                    + cv2.resize(orig_frames[b], (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
                )
                err = float(np.mean(np.abs(mid_s - g_s) * w3))
                if err < best_err - 0.15:
                    best_err = err
                    best_pair = (a, b)

        if best_pair is None:
            out.append(f0)
        else:
            a, b = best_pair
            fa, fb = orig_frames[a], orig_frames[b]
            hw, hh = max(16, w // 2), max(16, h // 2)
            fa_g = cv2.cvtColor(cv2.resize(fa, (hw, hh)), cv2.COLOR_BGR2GRAY)
            fb_g = cv2.cvtColor(cv2.resize(fb, (hw, hh)), cv2.COLOR_BGR2GRAY)
            flow_ab = cv2.resize(dis.calc(fa_g, fb_g, None), (w, h)) * 2.0
            flow_ba = cv2.resize(dis.calc(fb_g, fa_g, None), (w, h)) * 2.0
            wa = cv2.remap(fa, grid_x - 0.5 * flow_ab[:, :, 0], grid_y - 0.5 * flow_ab[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            wb = cv2.remap(fb, grid_x - 0.5 * flow_ba[:, :, 0], grid_y - 0.5 * flow_ba[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out.append(cv2.addWeighted(wa, 0.5, wb, 0.5, 0))
            half_count += 1
    return out, half_count


def align_frame_euclidean(
    orig: np.ndarray,
    gen: np.ndarray,
    bg_weight_small: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Sub-pixel Euclidean (translation + small roll) registration weighted to static background."""
    h, w = orig.shape[:2]
    s = 0.25
    sw, sh = max(32, int(round(w * s))), max(32, int(round(h * s)))
    g1 = cv2.cvtColor(cv2.resize(orig, (sw, sh)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(cv2.resize(gen, (sw, sh)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ecc_mask = (cv2.resize(bg_weight_small, (sw, sh)) > 0.5).astype(np.uint8) * 255

    (dx0, dy0), resp = cv2.phaseCorrelate(g2, g1)
    warp = np.array([[1.0, 0.0, dx0], [0.0, 1.0, dy0]], dtype=np.float32)
    if abs(dx0 / s) > 25 or abs(dy0 / s) > 25 or resp < 0.12:
        warp = np.eye(2, 3, dtype=np.float32)

    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
        _, warp = cv2.findTransformECC(g2, g1, warp, cv2.MOTION_EUCLIDEAN, criteria, inputMask=ecc_mask)
        dx, dy = float(warp[0, 2] / s), float(warp[1, 2] / s)
        angle = float(np.arctan2(warp[1, 0], warp[0, 0]))
        if abs(angle) > np.deg2rad(2.5) or abs(dx) > 22 or abs(dy) > 22:
            raise cv2.error("Clamped ECC warp")
        warp_full = warp.copy()
        warp_full[0, 2] = dx
        warp_full[1, 2] = dy
    except cv2.error:
        if abs(dx0 / s) <= 22 and abs(dy0 / s) <= 22 and resp >= 0.12:
            dx, dy = float(dx0 / s), float(dy0 / s)
        else:
            dx, dy = 0.0, 0.0
        warp_full = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    aligned = cv2.warpAffine(gen, warp_full, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned, (dx, dy)
