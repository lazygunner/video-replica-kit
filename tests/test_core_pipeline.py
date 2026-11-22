"""End-to-end automated verification tests for vidrep core guardrails, media I/O, pixel-lock, and compositing."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from vidrep.config import DEFAULT_VIDEO_MODEL, VidrepConfig
from vidrep.media.ffmpeg_io import ColorProfile, decode_video_frames, encode_video_with_audio, probe_video_info
from vidrep.media.pixel_lock import pixel_lock_reference
from vidrep.media.probe_slice import run_probe_bundle, slice_preview_clip
from vidrep.state import JobState
from vidrep.vfx.composite import run_composite_pipeline


def _make_synthetic_frames(n: int = 12, w: int = 320, h: int = 180, replaced: bool = False) -> list[np.ndarray]:
    """Build textured moving background with either original blue box or replaced red circle."""
    frames = []
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        shift = i * 2
        # Rich spatial texture so phaseCorrelate/ECC & DP alignment have strong gradients
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :, 0] = np.clip(90 + 50 * np.sin((xx + shift) * 0.06) + 30 * np.cos(yy * 0.08), 0, 255)
        bg[:, :, 1] = np.clip(140 + 40 * np.cos((xx + shift) * 0.05) + 35 * np.sin(yy * 0.07), 0, 255)
        bg[:, :, 2] = np.clip(80 + 45 * np.sin((xx + shift + yy) * 0.04), 0, 255)

        cx = int(w * 0.42 + i * 2)
        cy = int(h * 0.62)
        if not replaced:
            cv2.rectangle(bg, (cx - 18, cy - 26), (cx + 18, cy + 26), (220, 60, 40), -1)
        else:
            # Simulate mild diffusion warmth shift (+8 red) + replaced foreground circle
            bg = np.clip(bg.astype(np.int16) + np.array([-4, 2, 10]), 0, 255).astype(np.uint8)
            cv2.circle(bg, (cx, cy + 4), 22, (35, 45, 225), -1)
        frames.append(bg)
    return frames


def test_video_model_allowlist_blocks_non_flash():
    cfg = VidrepConfig()
    cfg.assert_video_model_allowed(DEFAULT_VIDEO_MODEL)
    with pytest.raises(ValueError, match="Blocked video model"):
        cfg.assert_video_model_allowed("doubao-seedance-2-5-260628")
    # Explicit override allowed
    cfg.assert_video_model_allowed("doubao-seedance-2-5-260628", allow_non_flash=True)


def test_user_supplied_config_no_implicit_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from vidrep.config import load_config

    for k in ("ARK_API_KEY", "TOS_ACCESS_KEY_ID", "TOS_ACCESS_KEY_SECRET", "TOS_BUCKET", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    empty_cfg = load_config(cwd=tmp_path, config_path=tmp_path / "nonexistent.toml")
    assert empty_cfg.ark.api_key == ""
    assert empty_cfg.tos.access_key_id == ""
    assert empty_cfg.to_safe_dict()["ark"]["api_key_set"] is False

    (tmp_path / ".env").write_text('ARK_API_KEY="user-key-123"\nTOS_BUCKET="my-bucket"\n', encoding="utf-8")
    loaded = load_config(cwd=tmp_path, config_path=tmp_path / "nonexistent.toml")
    assert loaded.ark.api_key == "user-key-123"
    assert loaded.tos.bucket == "my-bucket"


def test_state_idempotency_and_preview_gate(tmp_path: Path):
    st = JobState(tmp_path / "job1")
    with pytest.raises(RuntimeError, match="Preview gate not passed"):
        st.require_preview_passed()

    st.acquire_submission_lock("preview", {"model": DEFAULT_VIDEO_MODEL})
    with pytest.raises(RuntimeError, match="Duplicate submission blocked"):
        st.acquire_submission_lock("preview", {"model": DEFAULT_VIDEO_MODEL})

    st.data["submissions"]["preview"] = {"status": "succeeded"}
    st.require_preview_passed()  # now passes


def test_pixel_lock_auto_detect_and_zero_bg_drift(tmp_path: Path):
    f_orig = _make_synthetic_frames(1, w=640, h=360, replaced=False)[0]
    f_edit = _make_synthetic_frames(1, w=640, h=360, replaced=True)[0]
    p_orig = tmp_path / "orig.png"
    p_edit = tmp_path / "edit.png"
    p_out = tmp_path / "locked.png"
    cv2.imwrite(str(p_orig), f_orig)
    cv2.imwrite(str(p_edit), f_edit)

    res = pixel_lock_reference(p_orig, p_edit, p_out)
    assert p_out.is_file()
    assert res["auto_detected"] is True
    assert res["outer_bg_mae_before"] > 3.0
    assert res["outer_bg_mae_after"] < 0.05


def test_ffmpeg_hlg_roundtrip_and_full_composite_pipeline(tmp_path: Path):
    orig_frames = _make_synthetic_frames(n=24, w=320, h=180, replaced=False)
    gen_frames = _make_synthetic_frames(n=12, w=320, h=180, replaced=True)
    # Match generated frames temporally to every 2nd original frame (48fps -> 24fps)
    gen_matched = [_make_synthetic_frames(n=24, w=320, h=180, replaced=True)[i * 2] for i in range(12)]

    p_orig = tmp_path / "orig_hlg.mp4"
    p_gen = tmp_path / "gen_raw.mp4"
    hlg_profile = ColorProfile(
        pix_fmt="yuv420p", color_range="tv", color_space="bt2020nc",
        color_primaries="bt2020", color_transfer="arib-std-b67", is_hdr=True,
    )
    sdr_profile = ColorProfile()

    encode_video_with_audio(orig_frames, 48.0, p_orig, hlg_profile, crf=12, preset="ultrafast")
    encode_video_with_audio(gen_matched, 24.0, p_gen, sdr_profile, crf=14, preset="ultrafast")

    info = probe_video_info(p_orig)
    assert info["width"] == 320 and info["height"] == 180
    assert info["color_profile"]["is_hdr"] is True
    assert info["color_space"] == "bt2020nc"
    assert info["color_transfer"] == "arib-std-b67"

    # Probe bundle + preview slice
    probe_res = run_probe_bundle(p_orig, tmp_path / "probe_out")
    assert probe_res["shot_count"] >= 1
    assert len(probe_res["keyframes"]) == 2

    # Full composite pipeline with auto_contour multi-subject mask
    comp_res = run_composite_pipeline(p_orig, p_gen, tmp_path / "comp_out", mask_mode="auto_contour", tag="test")
    assert Path(comp_res["deliverables"]["composited"]).is_file()
    assert Path(comp_res["deliverables"]["graded"]).is_file()
    assert comp_res["composited_probe"]["color_transfer"] == "arib-std-b67"
    assert comp_res["bg_mae_composited_vs_orig"] < comp_res["bg_mae_raw_vs_orig"]
    assert comp_res["bg_mae_composited_vs_orig"] < 1.5

    # Verify round-trip decode of composited MP4 stays color-exact
    dec_orig, _ = decode_video_frames(p_orig)
    dec_comp, _ = decode_video_frames(Path(comp_res["deliverables"]["composited"]))
    assert len(dec_comp) == len(gen_matched)
    assert len(dec_orig) == len(orig_frames)


def test_10bit_pq_end_to_end_pipeline(tmp_path: Path):
    """Verify genuine 10-bit PQ (yuv420p10le / bt2020nc / smpte2084) 16-bit decode -> composite -> 10-bit encode."""
    orig_u8 = _make_synthetic_frames(n=12, w=320, h=180, replaced=False)
    orig_u16 = [(f.astype(np.uint16) * 257) for f in orig_u8]
    gen_u8 = _make_synthetic_frames(n=12, w=320, h=180, replaced=True)[::2]

    p_orig = tmp_path / "orig_10bit_pq.mp4"
    p_gen = tmp_path / "gen_8bit.mp4"
    pq_profile = ColorProfile(
        pix_fmt="yuv420p10le",
        color_range="tv",
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="smpte2084",
        is_hdr=True,
        is_10bit=True,
    )
    encode_video_with_audio(orig_u16, 48.0, p_orig, pq_profile, crf=12, preset="ultrafast")
    encode_video_with_audio(gen_u8, 24.0, p_gen, ColorProfile(), crf=14, preset="ultrafast")

    info = probe_video_info(p_orig)
    assert info["color_profile"]["is_10bit"] is True
    assert info["color_transfer"] == "smpte2084"
    assert "10" in info["pix_fmt"]

    res = run_composite_pipeline(p_orig, p_gen, tmp_path / "pq_out", mask_mode="auto_contour", tag="pq10")
    assert res["pipeline_bit_depth"] == 16
    assert res["composited_probe"]["color_transfer"] == "smpte2084"
    assert "10" in res["composited_probe"]["pix_fmt"]
    assert res["bg_mae_composited_vs_orig"] < 1.5


def test_onnx_sam2_and_depth_local_pipeline(tmp_path: Path):
    """Verify local SAM 2.1 Tiny matting + Depth Anything V2 occlusion guard end-to-end."""
    from vidrep.vfx.onnx_models import is_model_cached

    if not (is_model_cached("depth") and is_model_cached("sam2")):
        pytest.skip("ONNX models not pre-cached locally; run `vidrep models download` to enable.")

    orig_frames = _make_synthetic_frames(n=6, w=320, h=180, replaced=False)
    gen_frames = _make_synthetic_frames(n=6, w=320, h=180, replaced=True)
    p_orig = tmp_path / "orig.mp4"
    p_gen = tmp_path / "gen.mp4"
    encode_video_with_audio(orig_frames, 24.0, p_orig, ColorProfile(), crf=14, preset="ultrafast")
    encode_video_with_audio(gen_frames, 24.0, p_gen, ColorProfile(), crf=14, preset="ultrafast")

    res = run_composite_pipeline(
        p_orig,
        p_gen,
        tmp_path / "onnx_out",
        mask_mode="auto_contour",
        use_sam2=True,
        use_depth=True,
        onnx_stride=2,
        tag="onnx",
    )
    assert res["segmentation"]["onnx"]["use_sam2"] is True
    assert res["segmentation"]["onnx"]["use_depth"] is True
    assert res["segmentation"]["onnx"]["sam2_mean_iou"] > 0.3
    assert Path(res["qa_files"]["onnx_diagnostics"]).is_file()
    assert res["bg_mae_composited_vs_orig"] < 1.5
