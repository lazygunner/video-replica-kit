# video-replica-kit (`vidrep`)

[简体中文](README.md) | English

**Cross-agent universal reference video subject-replacement & automated color-grading toolkit** — works seamlessly across **Antigravity**, **Claude Code**, **OpenAI Codex**, **Cursor**, and standalone Python CLI (`vidrep`).

> **Core design**: Use `doubao-seedance-2-5-flash-261115` for semantic subject/creature/product replacement, perspective lighting, and synced speech — and use deterministic engineering for everything beyond the model's boundary (pixel-locked scene references, 480p pre-validation gates, DP/DIS/ECC alignment, local auto-downloaded SAM 2.1 + Depth Anything V2 ONNX matting, 5-level Laplacian original-background blending, and true Rec.709 / HLG / 10-bit PQ `zscale` color passthrough).

---

## One-Line Prompts (Copy Directly Into Your AI Agent)

### 1. Install & Configure
```text
Clone https://github.com/lazygunner/video-replica-kit.git, run ./install-skill.sh --target all --scope global to install the skill, and run uv run vidrep preflight to verify my environment and guide me through API key setup.
```

### 2. Recreate a Reference Video
```text
Use the video-subject-replica skill on [VIDEO_PATH_OR_URL] to replace [ORIGINAL_SUBJECT] with [NEW_SUBJECT_AND_STYLE] while preserving camera motion, background, and lighting. Start with reference images and a short 480p preview; once I approve, run 720p + local ONNX compositing.
```

---

## 30-Second Setup

- Requires Python `>= 3.10`, [`uv`](https://docs.astral.sh/uv/), and `ffmpeg`.
- **No API keys are bundled.** Configure your own `ARK_API_KEY` and Volcengine `TOS_*` credentials via `ENV`, `.env`, or `~/.config/vidrep/config.toml`:

```bash
git clone https://github.com/lazygunner/video-replica-kit.git
cd video-replica-kit
mkdir -p ~/.config/vidrep && cp config.example.toml ~/.config/vidrep/config.toml
# Fill in your own keys, then verify:
uv run vidrep preflight
./install-skill.sh --target all --scope global
```

---

## Quick CLI Cheatsheet

```bash
# Mode A: Reference video subject replacement (`edit` + local ONNX compositing)
uv run vidrep init ./jobs/demo --source /path/to/reference.mp4
uv run vidrep probe ./jobs/demo --keyframes 0.22,0.68
uv run vidrep slice-preview ./jobs/demo --start 0.0 --duration 4.2
uv run vidrep upload ./jobs/demo --stage preview --refs subject,scene-locked
uv run vidrep edit-submit ./jobs/demo --stage preview --resolution 480p --prompt-file ./jobs/demo/prompt.txt
uv run vidrep edit-wait ./jobs/demo --stage preview
uv run vidrep composite ./jobs/demo --stage preview --use-onnx

# Mode B: Zero-face multi-reference TVC / short-drama with synced native speech (`multi_ref`)
uv run vidrep upload ./jobs/tvc --stage preview --refs-only --refs ref_product,ref_room,ref_outfit
uv run vidrep edit-submit ./jobs/tvc --stage preview --mode multi_ref --resolution 480p \
  --ratio 9:16 --duration 12 --generate-audio --prompt-file ./jobs/tvc/prompt.txt
uv run vidrep edit-wait ./jobs/tvc --stage preview
```

---

## Tests & Docs

- Skill Spec: [skills/video-subject-replica/SKILL.md](skills/video-subject-replica/SKILL.md)
- Deep-Dive Tutorial (ZH): [docs/tutorial-seedance-2.5-flash.md](docs/tutorial-seedance-2.5-flash.md)
- Run Tests: `uv run --extra dev pytest -v`
- License: MIT
