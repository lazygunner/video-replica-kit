"""Automatic downloader & cache manager for local ONNX models (SAM 2.1 Tiny + Depth Anything V2 Small)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request


@dataclass(frozen=True)
class ModelFileSpec:
    remote_path: str
    local_name: str
    min_bytes: int = 1024


@dataclass(frozen=True)
class OnnxModelSpec:
    key: str
    repo_id: str
    description: str
    files: tuple[ModelFileSpec, ...]


MODEL_REGISTRY: dict[str, OnnxModelSpec] = {
    "depth": OnnxModelSpec(
        key="depth",
        repo_id="onnx-community/depth-anything-v2-small",
        description="Depth Anything V2 Small (FP32 ONNX, ~99 MB) for relative depth & foreground occlusion detection",
        files=(
            ModelFileSpec(remote_path="onnx/model.onnx", local_name="model.onnx", min_bytes=50_000_000),
        ),
    ),
    "sam2": OnnxModelSpec(
        key="sam2",
        repo_id="onnx-community/sam2.1-hiera-tiny-ONNX",
        description="SAM 2.1 Hiera Tiny (FP32 ONNX encoder + decoder, ~156 MB) for sharp subject matting",
        files=(
            ModelFileSpec(remote_path="onnx/vision_encoder.onnx", local_name="vision_encoder.onnx", min_bytes=100_000),
            ModelFileSpec(remote_path="onnx/vision_encoder.onnx_data", local_name="vision_encoder.onnx_data", min_bytes=50_000_000),
            ModelFileSpec(remote_path="onnx/prompt_encoder_mask_decoder.onnx", local_name="prompt_encoder_mask_decoder.onnx", min_bytes=50_000),
            ModelFileSpec(remote_path="onnx/prompt_encoder_mask_decoder.onnx_data", local_name="prompt_encoder_mask_decoder.onnx_data", min_bytes=10_000_000),
        ),
    ),
}


def get_model_cache_root() -> Path:
    custom = os.environ.get("VIDREP_MODEL_CACHE_DIR", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return (Path.home() / ".cache/vidrep/models").resolve()


def _candidate_endpoints() -> list[str]:
    endpoints: list[str] = []
    if env_ep := os.environ.get("HF_ENDPOINT", "").strip():
        endpoints.append(env_ep.rstrip("/"))
    for ep in ("https://huggingface.co", "https://hf-mirror.com"):
        if ep not in endpoints:
            endpoints.append(ep)
    return endpoints


def _download_file(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "video-replica-kit/0.2"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -10
        with part.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 20:
                        print(f"[vidrep] downloading {label}: {pct}% ({done/1e6:.1f}/{total/1e6:.1f} MB)", file=sys.stderr, flush=True)
                        last_pct = pct
    part.replace(dest)


def is_model_cached(model_key: str) -> bool:
    spec = MODEL_REGISTRY[model_key]
    base = get_model_cache_root() / spec.key
    for f in spec.files:
        p = base / f.local_name
        if not p.is_file() or p.stat().st_size < f.min_bytes:
            return False
    return True


def ensure_model_downloaded(model_key: str, force: bool = False) -> dict[str, Path]:
    """Ensure all files for an ONNX model exist locally, auto-downloading with mirror fallback."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown ONNX model '{model_key}'. Available: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_key]
    base = get_model_cache_root() / spec.key
    base.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    endpoints = _candidate_endpoints()
    for f in spec.files:
        dest = base / f.local_name
        if not force and dest.is_file() and dest.stat().st_size >= f.min_bytes:
            paths[f.local_name] = dest
            continue

        last_err = ""
        ok = False
        for ep in endpoints:
            url = f"{ep}/{spec.repo_id}/resolve/main/{f.remote_path}"
            try:
                print(f"[vidrep] fetching {spec.key}/{f.local_name} from {ep}...", file=sys.stderr, flush=True)
                _download_file(url, dest, f"{spec.key}/{f.local_name}")
                if dest.stat().st_size < f.min_bytes:
                    raise RuntimeError(f"Downloaded file too small ({dest.stat().st_size} bytes)")
                ok = True
                break
            except Exception as exc:
                last_err = f"{ep}: {exc}"
        if not ok:
            raise RuntimeError(f"Failed to download {spec.key}/{f.local_name}: {last_err}")
        paths[f.local_name] = dest
    return paths


def get_models_status() -> dict[str, Any]:
    root = get_model_cache_root()
    out: dict[str, Any] = {"cache_dir": str(root), "models": {}}
    for key, spec in MODEL_REGISTRY.items():
        base = root / key
        files_info = {}
        for f in spec.files:
            p = base / f.local_name
            files_info[f.local_name] = {
                "path": str(p),
                "exists": p.is_file() and p.stat().st_size >= f.min_bytes,
                "size_mb": round(p.stat().st_size / 1e6, 2) if p.is_file() else 0.0,
            }
        out["models"][key] = {
            "repo_id": spec.repo_id,
            "description": spec.description,
            "ready": is_model_cached(key),
            "files": files_info,
        }
    return out
