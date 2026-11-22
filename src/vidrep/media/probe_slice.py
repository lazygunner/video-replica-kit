"""Forensic video inspection, lightweight scene-cut detection, keyframe picking, and short preview slicing."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from .ffmpeg_io import decode_video_frames, infer_color_profile, probe_video_info, require_ffmpeg


def detect_scene_cuts(frames: list[np.ndarray], fps: float, threshold: float = 0.42, min_shot_sec: float = 0.6) -> list[dict[str, Any]]:
    """Lightweight HSV histogram + pixel-diff cut detector (zero external PySceneDetect dep)."""
    if not frames:
        return []
    n = len(frames)
    min_gap = max(1, int(round(min_shot_sec * fps)))
    hists = []
    smalls = []
    for f in frames:
        s = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
        smalls.append(s.astype(np.float32))
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist.flatten())

    cut_indices = [0]
    for i in range(1, n):
        hist_dist = float(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA))
        pix_diff = float(np.mean(np.abs(smalls[i] - smalls[i - 1])) / 255.0)
        score = 0.7 * hist_dist + 0.3 * pix_diff
        if score >= threshold and (i - cut_indices[-1]) >= min_gap:
            cut_indices.append(i)

    shots = []
    for idx, start_f in enumerate(cut_indices):
        end_f = cut_indices[idx + 1] if idx + 1 < len(cut_indices) else n
        shots.append({
            "shot_index": idx + 1,
            "start_frame": start_f,
            "end_frame": end_f - 1,
            "start_sec": round(start_f / fps, 3),
            "end_sec": round(end_f / fps, 3),
            "duration_sec": round((end_f - start_f) / fps, 3),
        })
    return shots


def save_contact_sheet(frames: list[np.ndarray], fps: float, out_path: Path, sample_fps: float = 2.0, cols: int = 4) -> Path:
    n = len(frames)
    dur = n / max(fps, 1e-3)
    count = max(1, min(24, int(round(dur * sample_fps))))
    tiles = []
    for k in range(count):
        t = (k + 0.5) / count * dur
        idx = min(n - 1, max(0, int(round(t * fps))))
        tile = cv2.resize(frames[idx], (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{t:.2f}s (f{idx})", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_path


def pick_sharp_keyframes(
    frames: list[np.ndarray],
    fps: float,
    out_dir: Path,
    fractions: tuple[float, ...] = (0.22, 0.68),
    window_sec: float = 0.35,
) -> list[dict[str, Any]]:
    """Pick sharpest (highest Laplacian variance) frames near target timeline fractions."""
    n = len(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    half_win = max(1, int(round(window_sec * fps)))
    picked = []
    for k, frac in enumerate(fractions, start=1):
        center = min(n - 1, max(0, int(round(frac * (n - 1)))))
        lo = max(0, center - half_win)
        hi = min(n, center + half_win + 1)
        best_i = center
        best_sharp = -1.0
        for i in range(lo, hi):
            gray = cv2.cvtColor(cv2.resize(frames[i], (640, 360)), cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharp > best_sharp:
                best_sharp = sharp
                best_i = i
        sec = round(best_i / fps, 2)
        fname = out_dir / f"orig-keyframe-{k}-{sec:.2f}s.png"
        cv2.imwrite(str(fname), frames[best_i])
        picked.append({
            "index": k,
            "frame": best_i,
            "sec": sec,
            "sharpness": round(best_sharp, 2),
            "path": str(fname),
        })
    return picked


def run_probe_bundle(
    video_path: Path,
    out_dir: Path,
    keyframes: tuple[float, ...] = (0.22, 0.68),
    ffmpeg_bin: str = "",
    ffprobe_bin: str = "",
) -> dict[str, Any]:
    frames, info = decode_video_frames(video_path, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
    fps = info["fps"]
    shots = detect_scene_cuts(frames, fps)
    cs_path = save_contact_sheet(frames, fps, out_dir / "reference-contact-sheet.jpg")
    kf = pick_sharp_keyframes(frames, fps, out_dir / "keyframes", fractions=keyframes)
    return {
        "info": info,
        "shot_count": len(shots),
        "shots": shots,
        "contact_sheet": str(cs_path),
        "keyframes": kf,
        "hdr_warning": (
            "HDR/BT.2020 source detected: do NOT run blind zscale+tonemap SDR preprocessing before editing; "
            "feed source unmodified and restore color tags via zscale on export."
            if info["color_profile"]["is_hdr"]
            else None
        ),
    }


def slice_preview_clip(
    video_path: Path,
    out_path: Path,
    start_sec: float = 0.0,
    duration_sec: float = 4.2,
    ffmpeg_bin: str = "",
    ffprobe_bin: str = "",
) -> dict[str, Any]:
    """
    Slice a keyframe-accurate short clip (default 4.2s) for low-cost 480p Flash validation.
    Ark Seedance 2.5 `edit` mode requires input video duration in [4.0, 30.0] seconds;
    if the source clip is slightly shorter than 4.1s, the final frame is cleanly held via tpad
    so zero adjacent-shot frames bleed in while satisfying the >=4.0s API requirement.
    """
    ff = require_ffmpeg(ffmpeg_bin)
    info = probe_video_info(video_path, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)
    total_dur = info["duration"]
    target_dur = max(4.1, duration_sec)
    dur = min(target_dur, total_dur)
    start = max(0.0, min(start_sec, max(0.0, total_dur - dur)))
    pad_sec = max(0.0, round(4.15 - dur, 3))
    out_dur = round(dur + pad_sec, 3)
    cp = infer_color_profile(info)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc_args = cp.ffmpeg_encode_args(input_is_rgb=False)
    if pad_sec > 0:
        # Inject tpad before zscale inside -vf so short (<4.0s) single shots cleanly satisfy Ark's >=4.0s rule
        vf_idx = enc_args.index("-vf") + 1
        enc_args[vf_idx] = f"tpad=stop_mode=clone:stop_duration={pad_sec:.3f},{enc_args[vf_idx]}"

    cmd = [
        ff, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{out_dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
    ]
    cmd += enc_args
    if info.get("has_audio"):
        if pad_sec > 0:
            cmd += ["-af", f"apad=pad_dur={pad_sec:.3f}"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    clip_info = probe_video_info(out_path, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)
    return {
        "path": str(out_path),
        "start_sec": round(start, 3),
        "requested_duration_sec": out_dur,
        "tail_pad_sec": pad_sec,
        "info": clip_info,
    }
