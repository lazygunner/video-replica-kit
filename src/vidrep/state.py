"""Idempotent job state machine and anti-duplicate-billing locks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState:
    def __init__(self, job_dir: Path) -> None:
        self.job_dir = Path(job_dir).resolve()
        self.path = self.job_dir / STATE_FILE
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "version": 1,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "stage": "uninitialized",
            "source_video": None,
            "probe": None,
            "preview_clip": None,
            "references": {},
            "uploads": {},
            "submissions": {},
            "deliverables": {},
            "qa": {},
            "history": [],
        }

    def save(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = _now_iso()
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def record_event(self, event: str, detail: dict[str, Any] | None = None) -> None:
        self.data.setdefault("history", []).append(
            {"ts": _now_iso(), "event": event, "detail": detail or {}}
        )

    def lock_file_for_stage(self, stage: str) -> Path:
        return self.job_dir / f".submission-lock-{stage}"

    def has_submission_lock(self, stage: str) -> bool:
        return self.lock_file_for_stage(stage).exists() or stage in (self.data.get("submissions") or {})

    def acquire_submission_lock(self, stage: str, meta: dict[str, Any], force: bool = False) -> None:
        lock = self.lock_file_for_stage(stage)
        if not force and (lock.exists() or stage in (self.data.get("submissions") or {})):
            raise RuntimeError(
                f"Duplicate submission blocked for stage '{stage}'. Lock already exists at {lock}. "
                "Refusing duplicate billable POST (use --force-resubmit only if explicitly approved by the user)."
            )
        self.job_dir.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"locked_at": _now_iso(), **meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def require_preview_passed(self, skip_preview_gate: bool = False) -> None:
        if skip_preview_gate:
            return
        preview = (self.data.get("submissions") or {}).get("preview") or {}
        if preview.get("status") != "succeeded":
            raise RuntimeError(
                "Preview gate not passed: run a short 2-3s 480p preview first (`vidrep edit-submit --stage preview --resolution 480p`) "
                "and verify it before full 720p submission, or pass --skip-preview-gate explicitly."
            )
