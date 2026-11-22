"""Pluggable image generation / editing provider (Seedream 5.5/5.0, Gemini, or local file)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
from typing import Any
import urllib.error
import urllib.request

from ..config import VidrepConfig


def _to_data_url(p: Path) -> str:
    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def generate_reference_image(
    prompt: str,
    out_path: Path,
    cfg: VidrepConfig,
    reference_images: list[Path] | None = None,
    aspect_ratio: str = "1:1",
    provider: str = "seedream",
    local_file: Path | None = None,
) -> dict[str, Any]:
    """Generate or edit a single reference image using a pluggable provider."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refs = [Path(p) for p in (reference_images or [])]

    if provider == "local":
        if not local_file or not Path(local_file).is_file():
            raise FileNotFoundError(f"local provider requires existing --local-file: {local_file}")
        shutil.copy2(local_file, out_path)
        return {"provider": "local", "model": "local-copy", "output_path": str(out_path)}

    if provider == "seedream":
        if not cfg.ark.api_key:
            raise ValueError("Missing ARK_API_KEY for Seedream image generation.")
        models = [cfg.ark.image_model]
        if cfg.ark.image_fallback_model and cfg.ark.image_fallback_model != cfg.ark.image_model:
            models.append(cfg.ark.image_fallback_model)

        img_field: Any = None
        if len(refs) == 1:
            img_field = _to_data_url(refs[0])
        elif len(refs) > 1:
            img_field = [_to_data_url(p) for p in refs]

        last_err = ""
        for model in models:
            size = "2K" if "5-5" in model else "2k"
            body: dict[str, Any] = {
                "model": model,
                "prompt": f"{prompt}，画面比例{aspect_ratio}" if "5-5" in model else prompt,
                "size": size,
                "response_format": "url",
                "watermark": False,
            }
            if img_field is not None:
                body["image"] = img_field

            req = urllib.request.Request(
                f"{cfg.ark.base_url}/images/generations",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {cfg.ark.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
                url = data["data"][0]["url"]
                urllib.request.urlretrieve(url, out_path)
                return {"provider": "seedream", "model": model, "output_path": str(out_path), "aspect_ratio": aspect_ratio}
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                last_err = f"HTTP {e.code} on {model}: {err_text[:400]}"
            except Exception as e:
                last_err = f"{model}: {e}"
        raise RuntimeError(f"Seedream image generation failed: {last_err}")

    if provider == "gemini":
        if not cfg.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY for Gemini image generation.")
        model = "gemini-3-pro-image-preview"
        parts: list[dict[str, Any]] = [{"text": f"{prompt}\nAspect ratio: {aspect_ratio}."}]
        for rp in refs:
            mime = "image/png" if rp.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(rp.read_bytes()).decode("ascii")}})
        body = {"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["IMAGE"]}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={cfg.gemini_api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    out_path.write_bytes(base64.b64decode(inline["data"]))
                    return {"provider": "gemini", "model": model, "output_path": str(out_path), "aspect_ratio": aspect_ratio}
        raise RuntimeError(f"Gemini returned no image payload: {json.dumps(data)[:400]}")

    raise ValueError(f"Unsupported image provider '{provider}'. Choose from: seedream, gemini, local.")
