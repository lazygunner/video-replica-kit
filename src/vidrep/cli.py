"""Unified cross-agent CLI (`vidrep`) with deterministic JSON output and strict safety guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from .config import load_config
from .media.ffmpeg_io import find_binary
from .media.pixel_lock import pixel_lock_reference
from .media.probe_slice import run_probe_bundle, slice_preview_clip
from .providers.image_gen import generate_reference_image
from .providers.storage_tos import upload_files_to_tos
from .providers.video_edit_ark import submit_ark_edit, wait_ark_edit
from .state import JobState
from .vfx.composite import run_composite_pipeline
from .vfx.onnx_models import MODEL_REGISTRY, ensure_model_downloaded, get_models_status


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_pair(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    parts = [int(x.strip()) for x in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y' integer pair, got: {raw}")
    return parts[0], parts[1]


def cmd_preflight(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    ff = find_binary("ffmpeg", cfg.ffmpeg_bin)
    fp = find_binary("ffprobe", cfg.ffprobe_bin)
    missing: list[str] = []
    if not ff:
        missing.append("ffmpeg (install via brew/apt or set VIDREP_FFMPEG_BIN)")
    if not cfg.ark.api_key:
        missing.append("ARK_API_KEY (set in env, .env, or ~/.config/vidrep/config.toml)")
    if not (cfg.tos.access_key_id and cfg.tos.access_key_secret and cfg.tos.bucket):
        missing.append("TOS_ACCESS_KEY_ID / TOS_ACCESS_KEY_SECRET / TOS_BUCKET")
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "binaries": {
            "ffmpeg": ff,
            "ffprobe": fp or "missing (will use ffmpeg -i fallback)",
        },
        "config": cfg.to_safe_dict(),
    }


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    job_dir = Path(args.job_dir).resolve()
    src = Path(args.source).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source video not found: {src}")
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / "reference-original.mp4"
    if dest != src:
        shutil.copy2(src, dest)
    st = JobState(job_dir)
    st.data["source_video"] = str(dest)
    st.data["stage"] = "initialized"
    st.record_event("init", {"source": str(src)})
    st.save()
    return {"job_dir": str(job_dir), "source_video": str(dest), "state_file": str(st.path)}


def cmd_probe(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    src = Path(args.source).resolve() if args.source else Path(st.data["source_video"])
    fracs = tuple(float(x.strip()) for x in args.keyframes.split(",") if x.strip())
    res = run_probe_bundle(src, st.job_dir, keyframes=fracs, ffmpeg_bin=cfg.ffmpeg_bin, ffprobe_bin=cfg.ffprobe_bin)
    st.data["source_video"] = str(src)
    st.data["probe"] = res
    st.data["stage"] = "probed"
    st.record_event("probe", {"shot_count": res["shot_count"], "duration": res["info"]["duration"]})
    st.save()
    return res


def cmd_slice_preview(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    src = Path(st.data["source_video"])
    out = st.job_dir / "preview-source.mp4"
    res = slice_preview_clip(
        src, out, start_sec=args.start, duration_sec=args.duration,
        ffmpeg_bin=cfg.ffmpeg_bin, ffprobe_bin=cfg.ffprobe_bin,
    )
    st.data["preview_clip"] = res
    st.record_event("slice_preview", {"start_sec": res["start_sec"], "duration_sec": res["requested_duration_sec"]})
    st.save()
    return res


def cmd_ref_gen(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else (args.prompt or "")
    if not prompt and args.provider != "local":
        raise ValueError("Provide --prompt or --prompt-file.")
    out_path = st.job_dir / "refs" / f"{args.name}.png"
    refs = [Path(p) for p in (args.ref or [])]
    res = generate_reference_image(
        prompt=prompt,
        out_path=out_path,
        cfg=cfg,
        reference_images=refs,
        aspect_ratio=args.aspect,
        provider=args.provider,
        local_file=Path(args.local_file) if args.local_file else None,
    )
    st.data.setdefault("references", {})[args.name] = res
    st.record_event("ref_gen", {"name": args.name, "provider": args.provider, "model": res.get("model")})
    st.save()
    return res


def cmd_pixel_lock(args: argparse.Namespace) -> dict[str, Any]:
    st = JobState(Path(args.job_dir))
    out_path = Path(args.out) if args.out else (st.job_dir / "refs" / f"{args.name or 'scene-locked'}.png")
    res = pixel_lock_reference(
        orig_path=Path(args.orig),
        edited_path=Path(args.edited),
        out_path=out_path,
        center=_parse_pair(args.center),
        axes=_parse_pair(args.axes),
        pad_scale=args.pad_scale,
    )
    if args.name:
        st.data.setdefault("references", {})[args.name] = res
    st.record_event("pixel_lock", {"out": str(out_path), "outer_bg_mae_after": res["outer_bg_mae_after"]})
    st.save()
    return res


def cmd_upload(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    files: dict[str, Path] = {}
    if not args.refs_only:
        if args.stage == "preview":
            clip = (st.data.get("preview_clip") or {}).get("path")
            if not clip:
                raise RuntimeError("No preview clip found. Run `vidrep slice-preview` first.")
            files["video"] = Path(clip)
        else:
            files["video"] = Path(st.data["source_video"])

    ref_list = [x.strip() for x in (args.refs or "").split(",") if x.strip()]
    passthrough_refs: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(ref_list, start=1):
        if item.startswith(("asset://", "http://", "https://")):
            passthrough_refs[f"ref_{idx}"] = {"url": item, "key": "", "local_path": None}
            continue
        p = Path(item)
        if not p.is_file():
            # Allow referencing by alias saved in state.json or file in <job_dir>/refs/<alias>.png
            ref_entry = (st.data.get("references") or {}).get(item)
            if ref_entry and ref_entry.get("output_path"):
                p = Path(ref_entry["output_path"])
            elif (st.job_dir / "refs" / f"{item}.png").is_file():
                p = st.job_dir / "refs" / f"{item}.png"
        files[f"ref_{idx}"] = p

    uploaded = upload_files_to_tos(files, cfg.tos, run_tag=f"vidrep-{args.stage}") if files else {}
    uploaded.update(passthrough_refs)
    st.data.setdefault("uploads", {})[args.stage] = uploaded
    st.record_event("upload", {"stage": args.stage, "count": len(uploaded)})
    st.save()
    return uploaded


def cmd_edit_submit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    ups = (st.data.get("uploads") or {}).get(args.stage)
    if not ups or (args.mode == "edit" and "video" not in ups):
        raise RuntimeError(f"No uploaded assets for stage '{args.stage}'. Run `vidrep upload --stage {args.stage}` first.")
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else (args.prompt or "")
    if not prompt_text:
        raise ValueError("Provide --prompt or --prompt-file.")
    img_urls = [v["url"] for k, v in sorted(ups.items(), key=lambda kv: int(kv[0].split("_")[1]) if kv[0].startswith("ref_") else 999) if k.startswith("ref_")]
    return submit_ark_edit(
        state=st,
        cfg=cfg,
        stage=args.stage,
        video_url=ups["video"]["url"] if "video" in ups else None,
        image_urls=img_urls,
        prompt_text=prompt_text,
        resolution=args.resolution,
        model_id=args.model,
        allow_non_flash=args.allow_non_flash,
        skip_preview_gate=args.skip_preview_gate,
        force_resubmit=args.force_resubmit,
        generate_audio=args.generate_audio,
        mode=args.mode,
        ratio=args.ratio,
        duration=args.duration,
    )


def cmd_edit_wait(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    return wait_ark_edit(st, cfg, stage=args.stage, poll_interval_sec=args.interval, timeout_sec=args.timeout)


def cmd_composite(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    st = JobState(Path(args.job_dir))
    stage = args.stage
    if args.orig_video:
        orig_v = Path(args.orig_video)
    elif stage == "preview":
        orig_v = Path(st.data["preview_clip"]["path"])
    else:
        orig_v = Path(st.data["source_video"])

    if args.gen_video:
        gen_v = Path(args.gen_video)
    else:
        gen_v = Path(st.data["submissions"][stage]["raw_video"])

    out_dir = Path(args.out_dir) if args.out_dir else (st.job_dir / ("preview-out" if stage == "preview" else "deliverables"))
    res = run_composite_pipeline(
        orig_video=orig_v,
        gen_video=gen_v,
        out_dir=out_dir,
        mask_mode=args.mask_mode,
        use_sam2=bool(args.use_onnx or args.use_sam2),
        use_depth=bool(args.use_onnx or args.use_depth),
        onnx_stride=args.onnx_stride,
        depth_margin=args.depth_margin,
        audio_source_mode=args.audio_source,
        ffmpeg_bin=cfg.ffmpeg_bin,
        ffprobe_bin=cfg.ffprobe_bin,
        tag=stage,
    )
    st.data.setdefault("deliverables", {})[stage] = res["deliverables"]
    st.data.setdefault("qa", {})[stage] = res["qa_files"]
    st.data["stage"] = f"{stage}_composited"
    st.record_event("composite", {"stage": stage, "bg_mae_composited": res["bg_mae_composited_vs_orig"]})
    st.save()
    return res


def cmd_models(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "status":
        return get_models_status()
    keys = list(MODEL_REGISTRY.keys()) if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    downloaded = {k: {fn: str(p) for fn, p in ensure_model_downloaded(k, force=args.force).items()} for k in keys}
    return {"downloaded": downloaded, "status": get_models_status()}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    return JobState(Path(args.job_dir)).data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidrep", description="Cross-agent universal reference video subject-replacement CLI.")
    p.add_argument("--json", action="store_true", help="Emit JSON output (default).")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("preflight", help="Check ffmpeg/ffprobe, credentials, and model allowlist.")
    sp.set_defaults(func=cmd_preflight)

    sp = sub.add_parser("init", help="Initialize a job directory with a reference video.")
    sp.add_argument("job_dir")
    sp.add_argument("--source", required=True)
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("probe", help="Forensic probe, scene-cut detection, 2fps contact sheet, and sharp keyframes.")
    sp.add_argument("job_dir")
    sp.add_argument("--source")
    sp.add_argument("--keyframes", default="0.22,0.68", help="Comma-separated timeline fractions to extract sharpest keyframes.")
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("slice-preview", help="Slice a short ~4.2s clip (Ark edit min 4s) for low-cost 480p Flash validation.")
    sp.add_argument("job_dir")
    sp.add_argument("--start", type=float, default=0.0)
    sp.add_argument("--duration", type=float, default=4.2)
    sp.set_defaults(func=cmd_slice_preview)

    sp = sub.add_parser("ref-gen", help="Generate or edit a reference image via pluggable provider.")
    sp.add_argument("job_dir")
    sp.add_argument("--name", required=True)
    sp.add_argument("--prompt")
    sp.add_argument("--prompt-file")
    sp.add_argument("--ref", action="append", help="Reference image path (repeatable).")
    sp.add_argument("--aspect", default="1:1")
    sp.add_argument("--provider", default="seedream", choices=["seedream", "gemini", "local"])
    sp.add_argument("--local-file")
    sp.set_defaults(func=cmd_ref_gen)

    sp = sub.add_parser("pixel-lock", help="Lock background pixels of an edited scene keyframe onto the original frame.")
    sp.add_argument("job_dir")
    sp.add_argument("--orig", required=True)
    sp.add_argument("--edited", required=True)
    sp.add_argument("--out")
    sp.add_argument("--name")
    sp.add_argument("--center", help="Optional manual ellipse center x,y")
    sp.add_argument("--axes", help="Optional manual ellipse axes rx,ry")
    sp.add_argument("--pad-scale", type=float, default=1.45)
    sp.set_defaults(func=cmd_pixel_lock)

    sp = sub.add_parser("upload", help="Upload reference video + images to Volcengine TOS and verify HTTP 200.")
    sp.add_argument("job_dir")
    sp.add_argument("--stage", required=True, choices=["preview", "full"])
    sp.add_argument("--refs", required=True, help="Comma-separated reference image aliases or paths in exact prompt order.")
    sp.add_argument("--refs-only", action="store_true", help="Upload only reference images (for pure multi_ref generation without reference_video).")
    sp.set_defaults(func=cmd_upload)

    sp = sub.add_parser("edit-submit", help="Submit one idempotent Ark Seedance task (Flash allowlisted by default).")
    sp.add_argument("job_dir")
    sp.add_argument("--stage", required=True, choices=["preview", "full"])
    sp.add_argument("--prompt")
    sp.add_argument("--prompt-file")
    sp.add_argument("--resolution", choices=["480p", "720p", "1080p"])
    sp.add_argument("--model")
    sp.add_argument("--mode", default="edit", choices=["edit", "multi_ref"], help="edit (with reference_video) or multi_ref (pure AI reference_images).")
    sp.add_argument("--ratio", default="adaptive", help="Output aspect ratio (e.g. adaptive, 9:16, 16:9).")
    sp.add_argument("--duration", type=int, default=-1, help="Output duration in seconds (-1 = match reference_video in edit mode).")
    sp.add_argument("--allow-non-flash", action="store_true")
    sp.add_argument("--skip-preview-gate", action="store_true")
    sp.add_argument("--force-resubmit", action="store_true")
    sp.add_argument("--generate-audio", action="store_true")
    sp.set_defaults(func=cmd_edit_submit)

    sp = sub.add_parser("edit-wait", help="Poll submitted Ark task until finished and download output MP4.")
    sp.add_argument("job_dir")
    sp.add_argument("--stage", required=True, choices=["preview", "full"])
    sp.add_argument("--interval", type=int, default=8)
    sp.add_argument("--timeout", type=int, default=900)
    sp.set_defaults(func=cmd_edit_wait)

    sp = sub.add_parser("composite", help="Run alignment, color grading, original-bg Laplacian compositing, and QA.")
    sp.add_argument("job_dir")
    sp.add_argument("--stage", default="full", choices=["preview", "full"])
    sp.add_argument("--orig-video")
    sp.add_argument("--gen-video")
    sp.add_argument("--out-dir")
    sp.add_argument("--mask-mode", default="auto_contour", choices=["auto_contour", "ellipse"])
    sp.add_argument("--use-onnx", action="store_true", help="Enable both local SAM 2.1 Tiny matting and Depth Anything V2 occlusion guard (auto-downloads ONNX weights).")
    sp.add_argument("--use-sam2", action="store_true", help="Enable local SAM 2.1 Tiny ONNX subject matting.")
    sp.add_argument("--use-depth", action="store_true", help="Enable local Depth Anything V2 ONNX foreground occlusion detection.")
    sp.add_argument("--onnx-stride", type=int, default=0, help="Run ONNX every N frames and propagate via optical flow (0 = auto).")
    sp.add_argument("--depth-margin", type=float, default=0.14, help="Relative depth threshold above subject depth to classify closer foreground occluders.")
    sp.add_argument("--audio-source", default="auto", choices=["auto", "orig", "gen"], help="Audio track to mux: auto (prefers generated audio if present), orig, or gen.")
    sp.set_defaults(func=cmd_composite)

    sp = sub.add_parser("models", help="Check or pre-download local ONNX models (SAM 2.1 Tiny + Depth Anything V2 Small).")
    sp.add_argument("action", choices=["status", "download"])
    sp.add_argument("--models", default="all", help="all, sam2, depth, or comma-separated list.")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("status", help="Inspect job state.json.")
    sp.add_argument("job_dir")
    sp.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        _emit(result, as_json=True)
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
