"""Layered configuration, secret discovery, and model-safety guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any

DEFAULT_VIDEO_MODEL = "doubao-seedance-2-5-flash-261115"
DEFAULT_ALLOWED_VIDEO_MODELS = (DEFAULT_VIDEO_MODEL,)
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-5-261028"
DEFAULT_IMAGE_FALLBACK_MODEL = "doubao-seedream-5-0-260128"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


@dataclass
class ArkConfig:
    api_key: str = ""
    base_url: str = DEFAULT_ARK_BASE_URL
    video_model: str = DEFAULT_VIDEO_MODEL
    allowed_video_models: tuple[str, ...] = DEFAULT_ALLOWED_VIDEO_MODELS
    image_model: str = DEFAULT_IMAGE_MODEL
    image_fallback_model: str = DEFAULT_IMAGE_FALLBACK_MODEL


@dataclass
class TosConfig:
    access_key_id: str = ""
    access_key_secret: str = ""
    region: str = "cn-shanghai"
    endpoint: str = "https://tos-cn-shanghai.volces.com"
    bucket: str = ""
    path_prefix: str = "reference-videos/"


@dataclass
class VidrepConfig:
    ark: ArkConfig = field(default_factory=ArkConfig)
    tos: TosConfig = field(default_factory=TosConfig)
    gemini_api_key: str = ""
    openai_api_key: str = ""
    ffmpeg_bin: str = ""
    ffprobe_bin: str = ""
    sources: list[str] = field(default_factory=list)

    def assert_video_model_allowed(self, model_id: str, allow_non_flash: bool = False) -> None:
        if allow_non_flash:
            return
        if model_id not in self.ark.allowed_video_models:
            allowed = ", ".join(self.ark.allowed_video_models)
            raise ValueError(
                f"Blocked video model '{model_id}'. Allowed models: [{allowed}]. "
                "Pass --allow-non-flash or set VIDREP_ALLOWED_VIDEO_MODELS explicitly to override."
            )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "ark": {
                "base_url": self.ark.base_url,
                "api_key_set": bool(self.ark.api_key),
                "video_model": self.ark.video_model,
                "allowed_video_models": list(self.ark.allowed_video_models),
                "image_model": self.ark.image_model,
                "image_fallback_model": self.ark.image_fallback_model,
            },
            "tos": {
                "endpoint": self.tos.endpoint,
                "region": self.tos.region,
                "bucket": self.tos.bucket,
                "path_prefix": self.tos.path_prefix,
                "credentials_set": bool(self.tos.access_key_id and self.tos.access_key_secret),
            },
            "gemini_api_key_set": bool(self.gemini_api_key),
            "openai_api_key_set": bool(self.openai_api_key),
            "ffmpeg_bin": self.ffmpeg_bin or "auto",
            "ffprobe_bin": self.ffprobe_bin or "auto",
        }


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(cwd: Path | None = None, config_path: Path | None = None) -> VidrepConfig:
    """
    Load user-configured credentials and settings with strict precedence:
      1. Process environment variables (highest)
      2. Local .env in CWD
      3. ~/.config/vidrep/config.toml (or explicit config_path)

    Users must supply their own API keys and storage credentials; no external app
    databases or bundled keys are read implicitly.
    """
    cfg = VidrepConfig()

    # 3. TOML file
    toml_file = config_path or (Path.home() / ".config/vidrep/config.toml")
    td_toml = _load_toml(toml_file)
    if td_toml:
        cfg.sources.append(str(toml_file))
        ark_t = td_toml.get("ark") or {}
        tos_t = td_toml.get("tos") or {}
        if ark_t.get("api_key"):
            cfg.ark.api_key = str(ark_t["api_key"]).strip()
        if ark_t.get("base_url"):
            cfg.ark.base_url = str(ark_t["base_url"]).strip().rstrip("/")
        if ark_t.get("video_model"):
            cfg.ark.video_model = str(ark_t["video_model"]).strip()
        if ark_t.get("allowed_video_models"):
            cfg.ark.allowed_video_models = tuple(str(x).strip() for x in ark_t["allowed_video_models"] if str(x).strip())
        if ark_t.get("image_model"):
            cfg.ark.image_model = str(ark_t["image_model"]).strip()
        if ark_t.get("image_fallback_model"):
            cfg.ark.image_fallback_model = str(ark_t["image_fallback_model"]).strip()
        for k_attr in ("access_key_id", "access_key_secret", "region", "endpoint", "bucket", "path_prefix"):
            if tos_t.get(k_attr):
                setattr(cfg.tos, k_attr, str(tos_t[k_attr]).strip())
        if td_toml.get("gemini_api_key"):
            cfg.gemini_api_key = str(td_toml["gemini_api_key"]).strip()
        if td_toml.get("openai_api_key"):
            cfg.openai_api_key = str(td_toml["openai_api_key"]).strip()

    # 2. Local .env + 1. Environment variables
    dotenv = _parse_dotenv((cwd or Path.cwd()) / ".env")
    if dotenv:
        cfg.sources.append(".env")

    def env_get(key: str) -> str:
        val = os.environ.get(key)
        if val is not None and val.strip() != "":
            return val.strip()
        return dotenv.get(key, "").strip()

    if v := env_get("ARK_API_KEY"):
        cfg.ark.api_key = v
    if v := env_get("ARK_BASE_URL"):
        cfg.ark.base_url = v.rstrip("/")
    if v := env_get("VIDREP_VIDEO_MODEL"):
        cfg.ark.video_model = v
    if v := env_get("VIDREP_ALLOWED_VIDEO_MODELS"):
        cfg.ark.allowed_video_models = tuple(x.strip() for x in v.split(",") if x.strip())
    if v := env_get("VIDREP_IMAGE_MODEL"):
        cfg.ark.image_model = v
    if v := env_get("VIDREP_IMAGE_FALLBACK_MODEL"):
        cfg.ark.image_fallback_model = v

    if v := env_get("TOS_ACCESS_KEY_ID"):
        cfg.tos.access_key_id = v
    if v := env_get("TOS_ACCESS_KEY_SECRET"):
        cfg.tos.access_key_secret = v
    if v := env_get("TOS_REGION"):
        cfg.tos.region = v
    if v := env_get("TOS_ENDPOINT"):
        cfg.tos.endpoint = v
    if v := env_get("TOS_BUCKET"):
        cfg.tos.bucket = v
    if v := env_get("TOS_PATH_PREFIX"):
        cfg.tos.path_prefix = v

    if v := env_get("GEMINI_API_KEY"):
        cfg.gemini_api_key = v
    if v := env_get("OPENAI_API_KEY"):
        cfg.openai_api_key = v
    if v := env_get("VIDREP_FFMPEG_BIN"):
        cfg.ffmpeg_bin = v
    if v := env_get("VIDREP_FFPROBE_BIN"):
        cfg.ffprobe_bin = v

    if any(os.environ.get(k) for k in ("ARK_API_KEY", "TOS_ACCESS_KEY_ID", "VIDREP_VIDEO_MODEL")):
        cfg.sources.append("env")

    return cfg
