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
    """Multi-band Laplacian pyramid blend (Burt & Adelson) supporting both uint8 and 16-bit uint16 HDR."""
    is_16 = bg.dtype == np.uint16
    max_val = 65535.0 if is_16 else 255.0
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)
    m_f = mask.astype(np.float32)
    if m_f.ndim == 2:
        m_f = m_f[:, :, None]

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
    return np.clip(out, 0.0, max_val).astype(np.uint16 if is_16 else np.uint8)


def _to_u8(frame: np.ndarray) -> np.ndarray:
    return (frame >> 8).astype(np.uint8) if frame.dtype == np.uint16 else frame


def run_composite_pipeline(
    orig_video: Path,
    gen_video: Path,
    out_dir: Path,
    mask_mode: str = "auto_contour",
    use_sam2: bool = False,
    use_depth: bool = False,
    onnx_stride: int = 0,
    depth_margin: float = 0.14,
    audio_source_mode: str = "auto",
    ffmpeg_bin: str = "",
    ffprobe_bin: str = "",
    tag: str = "final",
) -> dict[str, Any]:
    """
    Full automated finishing pipeline (supports 8-bit SDR/HLG AND true 16-bit->10-bit PQ/HLG):
      1. Deterministic FFmpeg decode (16-bit bgr48le when source is 10-bit HDR/PQ, else 8-bit bgr24)
      2. Adaptive static-bg DP temporal match + DIS optical-flow half-frames + ECC alignment
      3. General-purpose replacement mask (+ optional SAM 2.1 Tiny matting & Depth Anything V2 occlusion subtraction)
      4. Temporally smooth global color grading + 5-level Laplacian blend in native bit depth
      5. Dynamic source color-matrix passthrough encode (`zscale`) + lossless original AAC audio mux
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = require_ffmpeg(ffmpeg_bin)

    orig_native, orig_info = decode_video_frames(orig_video, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin, preserve_16bit=True)
    gen_native, gen_info = decode_video_frames(gen_video, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin, preserve_16bit=False)

    h_o, w_o = orig_info["height"], orig_info["width"]
    fps_o, fps_g = orig_info["fps"], gen_info["fps"]
    color_profile = infer_color_profile(orig_info)
    is_10bit_pipe = color_profile.is_10bit and orig_native[0].dtype == np.uint16
    n_g = len(gen_native)

    # 8-bit proxy views for ultra-fast DP/DIS/ECC/ONNX tracking
    orig_u8 = [_to_u8(f) for f in orig_native]
    gen_scaled_u8 = [
        f if (f.shape[1] == w_o and f.shape[0] == h_o) else cv2.resize(f, (w_o, h_o), interpolation=cv2.INTER_LANCZOS4)
        for f in gen_native
    ]

    match_idx, bg_w_small = monotonic_temporal_match(orig_u8, gen_scaled_u8, fps_o, fps_g)
    orig_matched_u8, half_frames_used = build_subframe_orig(orig_u8, match_idx, gen_scaled_u8, bg_w_small)

    # For 10-bit pipelines keep untouched 16-bit original frames for highlight-preserving compositing
    orig_matched_native = [orig_native[oi] for oi in match_idx] if is_10bit_pipe else orig_matched_u8

    gen_aligned_u8: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    for gi in range(n_g):
        al, sh = align_frame_euclidean(orig_matched_u8[gi], gen_scaled_u8[gi], bg_w_small)
        gen_aligned_u8.append(al)
        shifts.append(sh)
    shifts_arr = np.array(shifts, dtype=np.float64)

    masks, seg_meta = compute_replacement_masks(
        orig_matched_u8,
        gen_aligned_u8,
        bg_w_small,
        mode=mask_mode,
        use_sam2=use_sam2,
        use_depth=use_depth,
        onnx_stride=onnx_stride,
        depth_margin=depth_margin,
    )

    gains = np.zeros((n_g, 3), dtype=np.float64)
    biases = np.zeros((n_g, 3), dtype=np.float64)
    raw_bg_maes: list[float] = []

    for gi in range(n_g):
        fo = orig_matched_u8[gi]
        fg = gen_scaled_u8[gi]
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

    graded_out_frames: list[np.ndarray] = []
    comp_out_frames: list[np.ndarray] = []
    graded_u8_qa: list[np.ndarray] = []
    comp_u8_qa: list[np.ndarray] = []
    graded_bg_maes: list[float] = []
    comp_bg_maes: list[float] = []
    scale_16 = 257.0 if is_10bit_pipe else 1.0

    for gi in range(n_g):
        fo_u8 = orig_matched_u8[gi]
        fo_nat = orig_matched_native[gi]
        fg_u8 = gen_scaled_u8[gi].astype(np.float32)
        bg_bool = masks[gi] < 0.05
        if int(bg_bool.sum()) < h_o * w_o * 0.15:
            bg_bool = cv2.resize(bg_w_small, (w_o, h_o)) > 0.5

        g_f = np.empty_like(fg_u8)
        for c in range(3):
            g_f[:, :, c] = np.clip(fg_u8[:, :, c] * gains_s[gi, c] + biases_s[gi, c], 0.0, 255.0)
        g_u8 = g_f.astype(np.uint8)
        graded_u8_qa.append(g_u8)
        graded_out_frames.append(np.clip(g_f * scale_16, 0.0, 65535.0).astype(np.uint16) if is_10bit_pipe else g_u8)
        graded_bg_maes.append(float(np.mean(np.abs(fo_u8[bg_bool].astype(np.float32) - g_u8[bg_bool].astype(np.float32)))))

        m = masks[gi]
        ring = (m > 0.04) & (m < 0.55)
        fg_al = gen_aligned_u8[gi].astype(np.float32)
        fg_local = np.empty_like(fg_al)
        if int(ring.sum()) > 400:
            for c in range(3):
                mo, so = float(fo_u8[:, :, c][ring].mean()), float(fo_u8[:, :, c][ring].std()) + 1e-5
                mg, sg = float(fg_al[:, :, c][ring].mean()), float(fg_al[:, :, c][ring].std()) + 1e-5
                g_local = float(np.clip(so / sg, 0.88, 1.14))
                g_use = 0.5 * gains_s[gi, c] + 0.5 * g_local
                b_use = mo - mg * g_use
                fg_local[:, :, c] = np.clip(fg_al[:, :, c] * g_use + b_use, 0.0, 255.0)
        else:
            for c in range(3):
                fg_local[:, :, c] = np.clip(fg_al[:, :, c] * gains_s[gi, c] + biases_s[gi, c], 0.0, 255.0)

        fg_nat = np.clip(fg_local * scale_16, 0.0, 65535.0).astype(np.uint16) if is_10bit_pipe else fg_local.astype(np.uint8)
        blended_nat = laplacian_pyramid_blend(fg_nat, fo_nat, m, levels=5)
        blended_u8 = _to_u8(blended_nat)
        comp_out_frames.append(blended_nat)
        comp_u8_qa.append(blended_u8)
        comp_bg_maes.append(float(np.mean(np.abs(fo_u8[bg_bool].astype(np.float32) - blended_u8[bg_bool].astype(np.float32)))))

    out_graded = out_dir / f"{tag}-graded-with-audio.mp4"
    out_comp = out_dir / f"{tag}-composited-with-audio.mp4"
    out_raw_audio = out_dir / f"{tag}-raw-with-audio.mp4"

    # Choose audio track: 'auto' prefers generated audio when present (e.g. generate_audio=True dialogue dubbing)
    if audio_source_mode == "gen":
        chosen_audio_src = gen_video if gen_info.get("has_audio") else None
    elif audio_source_mode == "orig":
        chosen_audio_src = orig_video if orig_info.get("has_audio") else None
    else:
        chosen_audio_src = gen_video if gen_info.get("has_audio") else (orig_video if orig_info.get("has_audio") else None)

    encode_video_with_audio(graded_out_frames, fps_g, out_graded, color_profile, audio_source=chosen_audio_src, ffmpeg_bin=ff)
    encode_video_with_audio(comp_out_frames, fps_g, out_comp, color_profile, audio_source=chosen_audio_src, ffmpeg_bin=ff)

    if chosen_audio_src is not None:
        if chosen_audio_src == gen_video:
            subprocess.run(
                [ff, "-y", "-i", str(gen_video), "-c", "copy", "-movflags", "+faststart", str(out_raw_audio)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        else:
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

    probe_grad = probe_video_info(out_graded, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)
    probe_comp = probe_video_info(out_comp, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)

    from ..qa.report import generate_qa_bundle

    qa_files = generate_qa_bundle(
        out_dir=out_dir,
        orig_matched=orig_matched_u8,
        raw_scaled=gen_scaled_u8,
        graded_frames=graded_u8_qa,
        comp_frames=comp_u8_qa,
        masks=masks,
        centroids=seg_meta["centroids"],
        fps=fps_g,
        tag=tag,
        onnx_diag=seg_meta.get("onnx_diag"),
    )

    metrics: dict[str, Any] = {
        "source_probe": orig_info,
        "raw_probe": gen_info,
        "graded_probe": probe_grad,
        "composited_probe": probe_comp,
        "color_profile_used": color_profile.to_dict(),
        "pipeline_bit_depth": 16 if is_10bit_pipe else 8,
        "half_frames_synthesized": half_frames_used,
        "total_generated_frames": n_g,
        "median_shift_px": [round(float(np.median(shifts_arr[:, 0])), 3), round(float(np.median(shifts_arr[:, 1])), 3)],
        "max_abs_shift_px": round(float(np.max(np.abs(shifts_arr))), 3),
        "bg_mae_raw_vs_orig": round(float(np.mean(raw_bg_maes)), 3),
        "bg_mae_graded_vs_orig": round(float(np.mean(graded_bg_maes)), 3),
        "bg_mae_composited_vs_orig": round(float(np.mean(comp_bg_maes)), 3),
        "segmentation": {k: v for k, v in seg_meta.items() if k not in ("centroids", "onnx_diag")},
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
