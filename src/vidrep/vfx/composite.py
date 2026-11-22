"""
High-fidelity post-production: temporally smooth global color grading,
local collar color matching, 5-level Burt & Adelson Laplacian pyramid blending,
and dynamic source color-space passthrough encoding.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from ..media.ffmpeg_io import (
    decode_video_frames,
    encode_video_with_audio,
    infer_color_profile,
    probe_video_info,
    require_ffmpeg,
)
from .align import (
    align_frame_euclidean,
    build_subframe_orig,
    gaussian_smooth_1d,
    monotonic_temporal_match,
)
from .segment import compute_replacement_masks


def laplacian_pyramid_blend(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray, levels: int = 5) -> np.ndarray:
    """Multi-band Laplacian pyramid blend (Burt & Adelson) for invisible lighting seams."""
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)
    m_f = mask.astype(np.float32)
    if m_f.ndim == 2:
        m_f = m_f[:, :, None]

    # Cap pyramid levels so smallest level stays >= 16px
    min_dim = min(fg.shape[0], fg.shape[1])
    max_levels = max(1, int(np.floor(np.log2(max(16, min_dim) / 16.0))))
    levels = min(levels, max_levels)

    gp_fg, gp_bg, gp_m = [fg_f], [bg_f], [m_f]
    for _ in range(levels):
        gp_fg.append(cv2.pyrDown(gp_fg[-1]))
        gp_bg.append(cv2.pyrDown(gp_bg[-1]))
        gp_m.append(cv2.pyrDown(gp_m[-1]))

    lp_fg, lp_bg = [gp_fg[-1]], [gp_bg[-1]]
    for i in range(levels, 0, -1):
        sz = (gp_fg[i - 1].shape[1], gp_fg[i - 1].shape[0])
        lp_fg.append(gp_fg[i - 1] - cv2.pyrUp(gp_fg[i], dstsize=sz))
        lp_bg.append(gp_bg[i - 1] - cv2.pyrUp(gp_bg[i], dstsize=sz))

    blended_pyr = []
    for lf, lb, gm in zip(lp_fg, lp_bg, reversed(gp_m)):
        if gm.ndim == 2:
            gm = gm[:, :, None]
        blended_pyr.append(lf * gm + lb * (1.0 - gm))

    out = blended_pyr[0]
    for i in range(1, len(blended_pyr)):
        sz = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        out = cv2.pyrUp(out, dstsize=sz) + blended_pyr[i]
    return np.clip(out, 0, 255).astype(np.uint8)


def run_composite_pipeline(
    orig_video: Path,
    gen_video: Path,
    out_dir: Path,
    mask_mode: str = "auto_contour",
    ffmpeg_bin: str = "",
    ffprobe_bin: str = "",
    tag: str = "final",
) -> dict[str, Any]:
    """
    Full automated finishing pipeline:
      1. Deterministic FFmpeg decode (orig + gen)
      2. Adaptive static-bg DP temporal match + DIS optical-flow half-frames + ECC alignment
      3. Temporally smooth global color grading (graded deliverable)
      4. General-purpose replacement mask + collar color match + 5-level Laplacian blend (composited deliverable)
      5. Dynamic original color-space passthrough encode + lossless original AAC audio mux
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = require_ffmpeg(ffmpeg_bin)

    orig_frames, orig_info = decode_video_frames(orig_video, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)
    gen_native, gen_info = decode_video_frames(gen_video, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)

    h_o, w_o = orig_info["height"], orig_info["width"]
    fps_o, fps_g = orig_info["fps"], gen_info["fps"]
    color_profile = infer_color_profile(orig_info)
    n_g = len(gen_native)

    gen_scaled = [
        f if (f.shape[1] == w_o and f.shape[0] == h_o) else cv2.resize(f, (w_o, h_o), interpolation=cv2.INTER_LANCZOS4)
        for f in gen_native
    ]

    match_idx, bg_w_small = monotonic_temporal_match(orig_frames, gen_scaled, fps_o, fps_g)
    orig_matched, half_frames_used = build_subframe_orig(orig_frames, match_idx, gen_scaled, bg_w_small)

    gen_aligned: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    for gi in range(n_g):
        al, sh = align_frame_euclidean(orig_matched[gi], gen_scaled[gi], bg_w_small)
        gen_aligned.append(al)
        shifts.append(sh)
    shifts_arr = np.array(shifts, dtype=np.float64)

    masks, seg_meta = compute_replacement_masks(orig_matched, gen_aligned, bg_w_small, mode=mask_mode)

    # Background pixels strictly outside the replacement mask for fair color grading & MAE
    gains = np.zeros((n_g, 3), dtype=np.float64)
    biases = np.zeros((n_g, 3), dtype=np.float64)
    raw_bg_maes: list[float] = []

    for gi in range(n_g):
        fo = orig_matched[gi]
        fg = gen_scaled[gi]
        bg_bool = masks[gi] < 0.05
        if int(bg_bool.sum()) < h_o * w_o * 0.15:
            bg_bool = cv2.resize(bg_w_small, (w_o, h_o)) > 0.5
        raw_bg_maes.append(float(np.mean(np.abs(fo[bg_bool].astype(np.float32) - fg[bg_bool].astype(np.float32)))))
        for c in range(3):
            mo, so = float(fo[:, :, c][bg_bool].mean()), float(fo[:, :, c][bg_bool].std()) + 1e-5
            mg, sg = float(fg[:, :, c][bg_bool].mean()), float(fg[:, :, c][bg_bool].std()) + 1e-5
            g_ratio = float(np.clip(so / sg, 0.85, 1.18))
            gains[gi, c] = g_ratio
            biases[gi, c] = mo - mg * g_ratio

    gains_s = gaussian_smooth_1d(gains, sigma=min(10.0, max(1.0, n_g / 6.0)))
    biases_s = gaussian_smooth_1d(biases, sigma=min(10.0, max(1.0, n_g / 6.0)))

    graded_frames: list[np.ndarray] = []
    comp_frames: list[np.ndarray] = []
    graded_bg_maes: list[float] = []
    comp_bg_maes: list[float] = []

    for gi in range(n_g):
        fo = orig_matched[gi]
        fg = gen_scaled[gi].astype(np.float32)
        bg_bool = masks[gi] < 0.05
        if int(bg_bool.sum()) < h_o * w_o * 0.15:
            bg_bool = cv2.resize(bg_w_small, (w_o, h_o)) > 0.5

        g_out = np.empty_like(fg)
        for c in range(3):
            g_out[:, :, c] = np.clip(fg[:, :, c] * gains_s[gi, c] + biases_s[gi, c], 0, 255)
        g_u8 = g_out.astype(np.uint8)
        graded_frames.append(g_u8)
        graded_bg_maes.append(float(np.mean(np.abs(fo[bg_bool].astype(np.float32) - g_u8[bg_bool].astype(np.float32)))))

        # Collar ring around soft mask transition for seamless local tone match
        m = masks[gi]
        ring = (m > 0.04) & (m < 0.55)
        fg_al = gen_aligned[gi].astype(np.float32)
        fg_local = np.empty_like(fg_al)
        if int(ring.sum()) > 400:
            for c in range(3):
                mo, so = float(fo[:, :, c][ring].mean()), float(fo[:, :, c][ring].std()) + 1e-5
                mg, sg = float(fg_al[:, :, c][ring].mean()), float(fg_al[:, :, c][ring].std()) + 1e-5
                g_local = float(np.clip(so / sg, 0.88, 1.14))
                g_use = 0.5 * gains_s[gi, c] + 0.5 * g_local
                b_use = mo - mg * g_use
                fg_local[:, :, c] = np.clip(fg_al[:, :, c] * g_use + b_use, 0, 255)
        else:
            for c in range(3):
                fg_local[:, :, c] = np.clip(fg_al[:, :, c] * gains_s[gi, c] + biases_s[gi, c], 0, 255)

        blended = laplacian_pyramid_blend(fg_local.astype(np.uint8), fo, m, levels=5)
        comp_frames.append(blended)
        comp_bg_maes.append(float(np.mean(np.abs(fo[bg_bool].astype(np.float32) - blended[bg_bool].astype(np.float32)))))

    out_graded = out_dir / f"{tag}-graded-with-audio.mp4"
    out_comp = out_dir / f"{tag}-composited-with-audio.mp4"
    out_raw_audio = out_dir / f"{tag}-raw-with-audio.mp4"

    encode_video_with_audio(graded_frames, fps_g, out_graded, color_profile, audio_source=orig_video, ffmpeg_bin=ff)
    encode_video_with_audio(comp_frames, fps_g, out_comp, color_profile, audio_source=orig_video, ffmpeg_bin=ff)

    # Also mux raw video + original audio if present
    if orig_info.get("has_audio"):
        subprocess.run(
            [
                ff, "-y", "-i", str(gen_video), "-i", str(orig_video),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
                "-shortest", "-movflags", "+faststart", str(out_raw_audio),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    # Verify actual encoded streams
    probe_grad = probe_video_info(out_graded, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)
    probe_comp = probe_video_info(out_comp, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)

    # Generate QA sheets right here using in-memory matched frames (zero redundant decode!)
    from ..qa.report import generate_qa_bundle

    qa_files = generate_qa_bundle(
        out_dir=out_dir,
        orig_matched=orig_matched,
        raw_scaled=gen_scaled,
        graded_frames=graded_frames,
        comp_frames=comp_frames,
        masks=masks,
        centroids=seg_meta["centroids"],
        fps=fps_g,
        tag=tag,
    )

    metrics: dict[str, Any] = {
        "source_probe": orig_info,
        "raw_probe": gen_info,
        "graded_probe": probe_grad,
        "composited_probe": probe_comp,
        "color_profile_used": color_profile.to_dict(),
        "half_frames_synthesized": half_frames_used,
        "total_generated_frames": n_g,
        "median_shift_px": [round(float(np.median(shifts_arr[:, 0])), 3), round(float(np.median(shifts_arr[:, 1])), 3)],
        "max_abs_shift_px": round(float(np.max(np.abs(shifts_arr))), 3),
        "bg_mae_raw_vs_orig": round(float(np.mean(raw_bg_maes)), 3),
        "bg_mae_graded_vs_orig": round(float(np.mean(graded_bg_maes)), 3),
        "bg_mae_composited_vs_orig": round(float(np.mean(comp_bg_maes)), 3),
        "segmentation": {k: v for k, v in seg_meta.items() if k != "centroids"},
        "deliverables": {
            "composited": str(out_comp),
            "graded": str(out_graded),
            "raw_with_audio": str(out_raw_audio) if out_raw_audio.is_file() else None,
        },
        "qa_files": qa_files,
    }
    metrics_path = out_dir / f"{tag}-metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics
