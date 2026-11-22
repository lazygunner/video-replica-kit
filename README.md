# video-replica-kit (`vidrep`)

**Cross-agent universal reference video subject-replacement toolkit** — works seamlessly across **Antigravity**, **Claude Code**, **OpenAI Codex CLI**, **Cursor**, and **standalone Python CLI / CI**.

Built around one proven engineering principle:

> **Use `doubao-seedance-2-5-flash-261115` for what generative video excels at** (semantic subject/creature/vehicle replacement, 3D perspective adaptation, continuous biomechanics, dynamic shadow lighting), **and use deterministic engineering for everything beyond the model's boundary** (pixel-locked scene references, 2–3s 480p pre-validation gates, monotonic DP + DIS optical-flow half-frame alignment, ECC Euclidean registration, 5-level Laplacian multi-band original-background restoration, and exact Rec.709/HLG/PQ color-space passthrough).

---

## Architecture

```text
Antigravity / Claude Code / Codex / Cursor   ──▶  skills/video-subject-replica/SKILL.md  (Open Agent Skill)
                                                               │
                                                               ▼
Standalone CLI / Tapdance / CI               ──▶  vidrep CLI (Pure Python + uv, zero vendor lock-in)
                                                    ├── preflight / init / probe / slice-preview
                                                    ├── ref-gen (Seedream 5.5 / Gemini / local) + pixel-lock
                                                    ├── upload (pure Python TOS) + edit-submit/wait (Flash lock)
                                                    └── composite (DP + DIS + ECC + Laplacian + zscale) + QA
```

---

## Quickstart

### 1. Prerequisites & API Key Configuration (User-Supplied)
- Python `>= 3.10` + [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` (`brew install ffmpeg` on macOS; `ffprobe` optional — auto-falls back to `ffmpeg -i`)
- **No API keys are bundled.** Configure your own Volcengine Ark API Key + Volcengine TOS credentials using any of the 3 standard methods (precedence: `ENV > .env > ~/.config/vidrep/config.toml`):

**Option A — Project `.env` (recommended for local runs):**
```bash
cp .env.example .env
# Edit .env and fill in ARK_API_KEY, TOS_ACCESS_KEY_ID, TOS_ACCESS_KEY_SECRET, TOS_BUCKET
uv run vidrep preflight
```

**Option B — Global user config (`~/.config/vidrep/config.toml`, recommended when using the Skill across multiple projects):**
```bash
mkdir -p ~/.config/vidrep
cp config.example.toml ~/.config/vidrep/config.toml
# Edit ~/.config/vidrep/config.toml and fill in your own keys
uv run vidrep preflight
```

**Option C — Shell environment variables:**
```bash
export ARK_API_KEY="your-ark-api-key"
export TOS_ACCESS_KEY_ID="your-tos-ak"
export TOS_ACCESS_KEY_SECRET="your-tos-sk"
export TOS_BUCKET="your-public-read-bucket"
uv run vidrep preflight
```

> `vidrep preflight` returns `"ok": true` once `ffmpeg`, `ARK_API_KEY`, and `TOS_*` are all configured (and never prints secret values).

### 2. Install the Skill into Your AI Coding Agent(s)

```bash
# Install globally for Antigravity, Claude Code, Codex, and Cursor at once:
./install-skill.sh --target all --scope global

# Or install into a specific project workspace:
./install-skill.sh --target all --scope project --project-dir /path/to/your/project
```

Supported targets:
- `antigravity` (`~/.gemini/config/skills/` or `<project>/.agents/skills/`)
- `claude` (`~/.claude/skills/` or `<project>/.claude/skills/`)
- `codex` (`~/.codex/skills/` or `<project>/.agents/skills/`)
- `cursor` (`~/.cursor/skills/` or `<project>/.agents/skills/`)

---

## End-to-End CLI Workflow

```bash
# 0. Initialize & forensic probe (cuts, HDR check, 2fps contact sheet, sharpest keyframes)
uv run vidrep init ./jobs/demo --source /path/to/reference.mp4
uv run vidrep probe ./jobs/demo --keyframes 0.22,0.68

# 1. Generate studio subject references + pixel-locked scene keyframes
uv run vidrep ref-gen ./jobs/demo --name char-main --aspect 1:1 --prompt "..."
uv run vidrep ref-gen ./jobs/demo --name char-3q --aspect 1:1 --ref ./jobs/demo/refs/char-main.png --prompt "..."
uv run vidrep ref-gen ./jobs/demo --name scene-1-raw --aspect 16:9 \
  --ref ./jobs/demo/keyframes/orig-keyframe-1-*.png \
  --ref ./jobs/demo/refs/char-main.png --prompt "..."
uv run vidrep pixel-lock ./jobs/demo \
  --orig ./jobs/demo/keyframes/orig-keyframe-1-*.png \
  --edited ./jobs/demo/refs/scene-1-raw.png --name scene-1-locked

# 2. Mandatory low-cost short 480p preview validation (Gate 2)
uv run vidrep slice-preview ./jobs/demo --start 2.0 --duration 2.5
uv run vidrep upload ./jobs/demo --stage preview --refs char-main,char-3q,scene-1-locked
uv run vidrep edit-submit ./jobs/demo --stage preview --resolution 480p --prompt-file ./jobs/demo/prompt.txt
uv run vidrep edit-wait ./jobs/demo --stage preview
uv run vidrep composite ./jobs/demo --stage preview
# -> Inspect ./jobs/demo/preview-out/preview-qa-comparison.jpg & preview-qa-motion-5fps.jpg

# 3. Full-length 720p Flash edit + 1080p original-background restoration
uv run vidrep upload ./jobs/demo --stage full --refs char-main,char-3q,scene-1-locked
uv run vidrep edit-submit ./jobs/demo --stage full --resolution 720p --prompt-file ./jobs/demo/prompt.txt
uv run vidrep edit-wait ./jobs/demo --stage full
uv run vidrep composite ./jobs/demo --stage full --mask-mode auto_contour
```

---

## Built-in Safety Guardrails

- **Flash-Only Model Allowlist**: Defaults strictly to `doubao-seedance-2-5-flash-261115`. Attempting to submit any other video model fails fast unless `--allow-non-flash` or `VIDREP_ALLOWED_VIDEO_MODELS` is explicitly set.
- **Preview Gate**: `--stage full` refuses submission until `--stage preview` (`480p` short clip) succeeds (override only via explicit `--skip-preview-gate`).
- **Anti-Duplicate Billing Lock**: Every stage writes a submission lockfile before calling Ark POST, preventing accidental duplicate charges on retries.
- **Deterministic FFmpeg Color Pipeline**: Never relies on `cv2.VideoCapture` (avoiding macOS `AVFoundation` double HDR tone-mapping), and pairs `zscale` matrix encoding with source-matched VUI tags (`Rec.709`, `BT.2020/HLG`, or `BT.2020/PQ`).

---

## Development & Tests

```bash
uv run --extra dev pytest -v
```

## License

MIT
