"""Deterministic FFmpeg I/O with dynamic color-space passthrough (Rec.709 / HLG / PQ)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np


@dataclass
class ColorProfile:
    pix_fmt: str = "yuv420p"
    color_range: str = "tv"
    color_space: str = "bt709"
    color_primaries: str = "bt709"
    color_transfer: str = "bt709"
    is_hdr: bool = False
    is_10bit: bool = False
    raw_tags: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pix_fmt": self.pix_fmt,
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_primaries": self.color_primaries,
            "color_transfer": self.color_transfer,
            "is_hdr": self.is_hdr,
            "is_10bit": self.is_10bit,
            "raw_tags": self.raw_tags or {},
        }

    def ffmpeg_encode_args(self) -> list[str]:
        """Return matched zscale matrix + VUI flags so round-trip color error stays minimal."""
        rng = "full" if self.color_range in ("pc", "full") else "limited"
        cr_flag = "pc" if rng == "full" else "tv"
        pix = "yuv420p10le" if self.is_10bit else "yuv420p"

        if self.color_space in ("bt2020nc", "bt2020ncl") or self.color_primaries == "bt2020":
            trc = self.color_transfer if self.color_transfer in ("arib-std-b67", "smpte2084") else "arib-std-b67"
            return [
                "-vf", f"zscale=matrix=2020_ncl:range={rng}",
                "-pix_fmt", pix,
                "-color_range", cr_flag,
                "-colorspace", "bt2020nc",
                "-color_primaries", "bt2020",
                "-color_trc", trc,
                "-x264-params", f"colorprim=bt2020:transfer={trc}:colormatrix=bt2020nc",
            ]

        return [
            "-vf", f"zscale=matrix=709:range={rng}",
            "-pix_fmt", pix,
            "-color_range", cr_flag,
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        ]


def find_binary(name: str, explicit: str = "") -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        cand = Path(prefix) / name
        if cand.is_file():
            return str(cand)
    return None


def require_ffmpeg(explicit: str = "") -> str:
    b = find_binary("ffmpeg", explicit)
    if not b:
        raise FileNotFoundError("ffmpeg not found on PATH (install via `brew install ffmpeg` or set VIDREP_FFMPEG_BIN).")
    return b


def _parse_fps(rate_str: str) -> float:
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        d = float(den)
        return float(num) / d if d else 0.0
    return float(rate_str)


def _probe_via_ffmpeg_stderr(video_path: Path, ffmpeg_bin: str) -> dict[str, Any]:
    """Fallback probe when ffprobe binary is missing on the host."""
    proc = subprocess.run([ffmpeg_bin, "-hide_banner", "-i", str(video_path)], capture_output=True, text=True)
    err = proc.stderr
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    duration = 0.0
    if m_dur:
        duration = int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))

    m_vid = re.search(r"Stream #\d+:\d+.*?Video:\s*([^\n]+)", err)
    if not m_vid:
        raise RuntimeError(f"Could not parse video stream from ffmpeg -i output for {video_path}")
    vline = m_vid.group(1)
    m_wh = re.search(r"(\d{2,5})x(\d{2,5})", vline)
    w, h = (int(m_wh.group(1)), int(m_wh.group(2))) if m_wh else (0, 0)
    m_fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", vline)
    fps = float(m_fps.group(1)) if m_fps else 24.0

    pix_fmt = "yuv420p10le" if "yuv420p10le" in vline else "yuv420p"
    color_range = "pc" if "(pc" in vline else "tv"
    color_space = "bt2020nc" if "bt2020nc" in vline else ("bt709" if "bt709" in vline else "unknown")
    color_primaries = "bt2020" if "bt2020" in vline else ("bt709" if "bt709" in vline else "unknown")
    if "arib-std-b67" in vline:
        color_transfer = "arib-std-b67"
    elif "smpte2084" in vline:
        color_transfer = "smpte2084"
    elif "bt709" in vline:
        color_transfer = "bt709"
    else:
        color_transfer = "unknown"

    has_audio = bool(re.search(r"Stream #\d+:\d+.*?Audio:", err))
    return {
        "width": w,
        "height": h,
        "fps": fps,
        "duration": round(duration, 6),
        "nb_frames": int(round(duration * fps)) if duration and fps else 0,
        "has_audio": has_audio,
        "codec": vline.split(",")[0].strip().split()[0],
        "pix_fmt": pix_fmt,
        "color_range": color_range,
        "color_space": color_space,
        "color_primaries": color_primaries,
        "color_transfer": color_transfer,
        "probe_backend": "ffmpeg-stderr-fallback",
    }


def probe_video_info(video_path: Path, ffmpeg_bin: str = "", ffprobe_bin: str = "") -> dict[str, Any]:
    vp = Path(video_path)
    if not vp.is_file():
        raise FileNotFoundError(f"Video file not found: {vp}")

    fp = find_binary("ffprobe", ffprobe_bin)
    if fp:
        out = subprocess.check_output(
            [fp, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(vp)],
            text=True,
        )
        raw = json.loads(out)
        streams = raw.get("streams") or []
        vs = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not vs:
            raise RuntimeError(f"No video stream in {vp}")
        aus = next((s for s in streams if s.get("codec_type") == "audio"), None)
        fps = _parse_fps(vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "24/1")
        if fps <= 0:
            fps = _parse_fps(vs.get("r_frame_rate") or "24/1")
        dur = float(vs.get("duration") or raw.get("format", {}).get("duration") or 0.0)
        nb = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else int(round(dur * fps))
        info = {
            "width": int(vs["width"]),
            "height": int(vs["height"]),
            "fps": round(fps, 6),
            "duration": round(dur, 6),
            "nb_frames": nb,
            "has_audio": aus is not None,
            "audio_codec": aus.get("codec_name") if aus else None,
            "audio_sample_rate": int(aus["sample_rate"]) if aus and aus.get("sample_rate") else None,
            "codec": vs.get("codec_name", "unknown"),
            "pix_fmt": vs.get("pix_fmt", "yuv420p"),
            "color_range": vs.get("color_range", "unknown"),
            "color_space": vs.get("color_space", "unknown"),
            "color_primaries": vs.get("color_primaries", "unknown"),
            "color_transfer": vs.get("color_transfer", "unknown"),
            "size_bytes": vp.stat().st_size,
            "probe_backend": "ffprobe",
        }
    else:
        ff = require_ffmpeg(ffmpeg_bin)
        info = _probe_via_ffmpeg_stderr(vp, ff)
        info["size_bytes"] = vp.stat().st_size

    info["color_profile"] = infer_color_profile(info).to_dict()
    return info


def infer_color_profile(info: dict[str, Any]) -> ColorProfile:
    pix = str(info.get("pix_fmt") or "yuv420p")
    cr = str(info.get("color_range") or "unknown")
    cs = str(info.get("color_space") or "unknown")
    cp = str(info.get("color_primaries") or "unknown")
    ct = str(info.get("color_transfer") or "unknown")

    is_10bit = "10" in pix or "12" in pix
    is_hdr = ct in ("arib-std-b67", "smpte2084") or cp == "bt2020" or cs in ("bt2020nc", "bt2020ncl")

    return ColorProfile(
        pix_fmt="yuv420p10le" if is_10bit else "yuv420p",
        color_range="pc" if cr == "pc" else "tv",
        color_space="bt2020nc" if is_hdr else "bt709",
        color_primaries="bt2020" if is_hdr else "bt709",
        color_transfer=ct if ct in ("arib-std-b67", "smpte2084") else "bt709",
        is_hdr=is_hdr,
        is_10bit=is_10bit,
        raw_tags={"pix_fmt": pix, "color_range": cr, "color_space": cs, "color_primaries": cp, "color_transfer": ct},
    )


def decode_video_frames(
    video_path: Path,
    ffmpeg_bin: str = "",
    ffprobe_bin: str = "",
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Deterministic FFmpeg rawvideo BGR24 decode.
    Never uses cv2.VideoCapture so macOS AVFoundation cannot apply implicit HDR->SDR tone-mapping.
    """
    ff = require_ffmpeg(ffmpeg_bin)
    info = probe_video_info(video_path, ffmpeg_bin=ff, ffprobe_bin=ffprobe_bin)
    w, h = info["width"], info["height"]
    cmd = [ff, "-v", "error", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.check_output(cmd)
    frame_bytes = w * h * 3
    n = len(raw) // frame_bytes
    if n == 0:
        raise RuntimeError(f"Decoded 0 frames from {video_path}")
    arr = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, h, w, 3)
    return [arr[i].copy() for i in range(n)], info


def encode_video_with_audio(
    frames: list[np.ndarray],
    fps: float,
    out_path: Path,
    color_profile: ColorProfile,
    audio_source: Path | None = None,
    crf: int = 15,
    preset: str = "slow",
    ffmpeg_bin: str = "",
) -> Path:
    """Single-pass rawvideo pipe encode with dynamic color-matrix passthrough and optional lossless AAC mux."""
    ff = require_ffmpeg(ffmpeg_bin)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

    has_audio = False
    if audio_source and Path(audio_source).is_file():
        src_info = probe_video_info(Path(audio_source), ffmpeg_bin=ff)
        has_audio = bool(src_info.get("has_audio"))

    cmd = [
        ff, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps:.6f}",
        "-i", "-",
    ]
    if has_audio and audio_source:
        cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0"]

    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    cmd += color_profile.ffmpeg_encode_args()
    if has_audio:
        cmd += ["-c:a", "copy", "-shortest"]
    cmd += ["-movflags", "+faststart", str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed (code {rc}): {err[-800:]}")
    return out_path
