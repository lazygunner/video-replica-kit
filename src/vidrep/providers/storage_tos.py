"""Pure-Python Volcengine TOS uploader with mandatory HTTP 200 public-read verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import urllib.request
from urllib.parse import urlparse
import uuid
import warnings

from ..config import TosConfig


def _normalize_endpoint_host(endpoint: str, bucket: str) -> str:
    raw = endpoint if "://" in endpoint else f"https://{endpoint}"
    host = urlparse(raw).netloc
    prefix = f"{bucket.lower()}."
    if host.lower().startswith(prefix):
        host = host[len(prefix) :]
    return host


def upload_files_to_tos(
    files: dict[str, Path],
    tos_cfg: TosConfig,
    run_tag: str = "vidrep",
) -> dict[str, dict[str, str]]:
    """
    Upload named local files to Volcengine TOS and verify public HTTP 200 via HEAD.
    Zero Node.js dependencies.
    """
    if not (tos_cfg.access_key_id and tos_cfg.access_key_secret and tos_cfg.bucket and tos_cfg.endpoint):
        raise ValueError("Incomplete TOS config. Set TOS_ACCESS_KEY_ID, TOS_ACCESS_KEY_SECRET, TOS_ENDPOINT, TOS_BUCKET.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        import tos  # lazy import avoids noisy third-party SyntaxWarning on non-upload commands

    host = _normalize_endpoint_host(tos_cfg.endpoint, tos_cfg.bucket)
    client = tos.TosClientV2(
        ak=tos_cfg.access_key_id,
        sk=tos_cfg.access_key_secret,
        endpoint=host,
        region=tos_cfg.region,
    )

    prefix = tos_cfg.path_prefix.strip("/")
    results: dict[str, dict[str, str]] = {}
    for alias, local_path in files.items():
        lp = Path(local_path)
        if not lp.is_file():
            raise FileNotFoundError(f"Cannot upload missing file '{alias}': {lp}")
        key = f"{prefix}/{run_tag}-{uuid.uuid4()}-{lp.name}" if prefix else f"{run_tag}-{uuid.uuid4()}-{lp.name}"
        client.put_object_from_file(tos_cfg.bucket, key, str(lp))
        public_url = f"https://{tos_cfg.bucket}.{host}/{key}"

        req = urllib.request.Request(public_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Uploaded {alias} but public HEAD check returned HTTP {resp.status}: {public_url}")

        results[alias] = {
            "url": public_url,
            "key": key,
            "local_path": str(lp.resolve()),
        }
    return results
