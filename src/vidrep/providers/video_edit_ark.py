"""Volcengine Ark Seedance video editing provider with strict Flash whitelist and idempotency locks."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

from ..config import VidrepConfig
from ..state import JobState


def submit_ark_edit(
    state: JobState,
    cfg: VidrepConfig,
    stage: str,
    video_url: str | None,
    image_urls: list[str],
    prompt_text: str,
    resolution: str | None = None,
    model_id: str | None = None,
    allow_non_flash: bool = False,
    skip_preview_gate: bool = False,
    force_resubmit: bool = False,
    generate_audio: bool = False,
    mode: str = "edit",
    ratio: str = "adaptive",
    duration: int = -1,
) -> dict[str, Any]:
    """Submit exactly ONE Seedance task per stage ('preview' or 'full') in 'edit' or 'multi_ref' mode."""
    if stage not in ("preview", "full"):
        raise ValueError("stage must be 'preview' or 'full'")
    if mode not in ("edit", "multi_ref"):
        raise ValueError("mode must be 'edit' or 'multi_ref'")

    chosen_model = (model_id or cfg.ark.video_model).strip()
    cfg.assert_video_model_allowed(chosen_model, allow_non_flash=allow_non_flash)

    if stage == "full":
        state.require_preview_passed(skip_preview_gate=skip_preview_gate)

    chosen_res = resolution or ("480p" if stage == "preview" else "720p")
    if stage == "preview" and chosen_res != "480p":
        raise ValueError("Preview stage enforces resolution='480p' for low-cost validation.")
    if not cfg.ark.api_key:
        raise ValueError("Missing ARK_API_KEY.")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text.strip()}]
    if mode == "edit":
        if not video_url:
            raise ValueError("mode='edit' requires video_url.")
        content.append({"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"})
    for u in image_urls:
        content.append({"type": "image_url", "image_url": {"url": u}, "role": "reference_image"})

    body: dict[str, Any] = {
        "model": chosen_model,
        "content": content,
        "resolution": chosen_res,
        "ratio": ratio,
        "duration": duration,
        "generate_audio": generate_audio,
        "watermark": False,
    }
    if mode == "edit":
        body["omni_reference_task_type"] = "edit"

    state.acquire_submission_lock(stage, {"model": chosen_model, "resolution": chosen_res}, force=force_resubmit)
    req_file = state.job_dir / f"request-{stage}.json"
    req_file.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    url = f"{cfg.ark.base_url}/contents/generations/tasks"
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {cfg.ark.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            task_data = json.load(resp)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        if 400 <= exc.code < 500:
            state.mark_submission_rejected(stage, exc.code, err_body)
        raise RuntimeError(f"Ark HTTP {exc.code} Bad Request: {err_body}") from exc

    task_id = task_data["id"]
    state.data.setdefault("submissions", {})[stage] = {
        "task_id": task_id,
        "model": chosen_model,
        "resolution": chosen_res,
        "status": "submitted",
        "request_file": str(req_file),
    }
    state.data["stage"] = f"{stage}_submitted"
    state.record_event(f"edit_submit_{stage}", {"task_id": task_id, "model": chosen_model, "resolution": chosen_res})
    state.save()
    return state.data["submissions"][stage]


def wait_ark_edit(
    state: JobState,
    cfg: VidrepConfig,
    stage: str,
    poll_interval_sec: int = 8,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    sub = (state.data.get("submissions") or {}).get(stage)
    if not sub or not sub.get("task_id"):
        raise RuntimeError(f"No submitted task found for stage '{stage}' in {state.path}")

    task_id = sub["task_id"]
    url = f"{cfg.ark.base_url}/contents/generations/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {cfg.ark.api_key}"}

    start = time.time()
    while True:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        st = data.get("status", "unknown")
        if st in ("succeeded", "failed", "cancelled", "expired"):
            res_file = state.job_dir / f"result-{stage}.json"
            res_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            sub["status"] = st
            sub["usage"] = data.get("usage")
            sub["error"] = data.get("error")
            sub["result_file"] = str(res_file)
            if st == "succeeded":
                dest = state.job_dir / ("preview-raw.mp4" if stage == "preview" else "generated-raw.mp4")
                urllib.request.urlretrieve(data["content"]["video_url"], dest)
                sub["raw_video"] = str(dest)
                state.data["stage"] = f"{stage}_succeeded"
            else:
                state.data["stage"] = f"{stage}_{st}"
            state.record_event(f"edit_wait_{stage}", {"task_id": task_id, "status": st, "usage": data.get("usage")})
            state.save()
            return sub
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for Ark task {task_id} (last status: {st})")
        time.sleep(poll_interval_sec)
