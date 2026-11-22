---
name: video-subject-replica
description: >-
  End-to-end reference video subject-replacement workflow using Seedance 2.5 Flash
  (doubao-seedance-2-5-flash-261115) with pixel-locked reference images, mandatory
  short 480p pre-validation, and automated high-resolution original-background
  compositing + color-space passthrough. Works identically across Antigravity,
  Claude Code, OpenAI Codex, Cursor, and standalone CLI via `vidrep`.
---

# Reference Video Subject Replacement (`video-subject-replica`)

Use this skill whenever the user wants to recreate a reference video by replacing one or more foreground subjects (characters, creatures/monsters, vehicles, props, products) while preserving the original video's camera motion, framing, pacing, background textures, lighting, and audio.

## Non-Negotiable Guardrails

1. **Flash-Only Default**: Always use `doubao-seedance-2-5-flash-261115` unless the user explicitly approves another model (`--allow-non-flash`).
2. **Short Low-Res Pre-Validation First**: Before running a full-length `720p` edit, ALWAYS slice a representative `~4.2s` clip (`vidrep slice-preview`, auto-pads `<4.0s` single shots via `tpad` to satisfy Ark `edit` mode's `[4, 30]s` rule) and validate with `480p` Flash (`--stage preview`). Only run `--stage full --resolution 720p` after the user approves the preview QA frames.
3. **Single Billable POST Discipline**: Never retry a submitted video generation task blindly. `vidrep edit-submit` enforces a lockfile per stage to prevent accidental double billing (synchronous HTTP 4xx rejections with 0 tokens billed automatically mark the lock `rejected_4xx` for safe retry after fixing parameters).
4. **Never Blindly Tone-Map HDR/HLG Inputs**: If `vidrep probe` reports `is_hdr: true` (e.g., `bt2020nc / arib-std-b67 HLG` or `smpte2084` 10-bit PQ), feed the unmodified source video to the edit API and let `vidrep composite` restore exact color signaling via `zscale` on export.
5. **Visual Inspection Before Every Gate**: Never pass an uninspected generated image or video frame to the next stage. Visually inspect every reference image and QA sheet first.
6. **Zero-Face Compliant Fallback (`--mode multi_ref`)**: If the source video or photorealistic human references contain identifiable real faces and trigger platform privacy checks (`InputVideoSensitiveContentDetected.PrivacyInformation`), NEVER bypass checks — switch to `--mode multi_ref --refs-only` using zero-human product/creature, empty-scene, and wardrobe flat-lay references with text-driven characters and `--generate-audio`.

## Running the `vidrep` CLI

Invoke `vidrep` via `uv` from the `video-replica-kit` repository root (or directly if installed globally via `uv tool install`):

```bash
VIDREP="uv run --project /path/to/video-replica-kit vidrep"
$VIDREP preflight
```

Every subcommand prints structured JSON to stdout and updates `<job_dir>/state.json`.

---

## Standard 5-Phase Workflow

### Phase 0 — Preflight & Forensic Source Probe
1. Run `vidrep preflight` to verify `ffmpeg`, credentials (`ARK_API_KEY`, `TOS_*`), and the Flash-only allowlist.
2. Initialize and probe the reference video:
   ```bash
   $VIDREP init ./jobs/<name> --source /path/to/reference.mp4
   $VIDREP probe ./jobs/<name> --keyframes 0.22,0.68
   ```
3. Visually inspect `<job_dir>/reference-contact-sheet.jpg` and the extracted keyframes.
   - Check `shot_count`: if multi-shot (`> 1`), confirm with the user which shot(s) to replace and work shot-by-shot.
   - Check camera angle(s), subject proportions, lighting direction, cast shadows, and any foreground occluders passing in front of the subject.
   - Read `references/color-pipelines.md` if `color_profile.is_hdr` is `true`.

### Phase 1 — Reference Image Engineering + Pixel-Lock (Gate 1)
Consult `references/prompt-patterns.md` to choose the right reference set for the replacement archetype (single subject, character+vehicle, character+creature/monster, or product/prop).

Rules:
- **Match the dominant camera angle first**: Generate clean studio single-view subject reference(s) matching the main shot angle (e.g., straight-behind or 3/4 view) on neutral studio background. Never use split multi-view collage sheets (they induce split-screen artifacts in video models).
- **Consistent secondary structural angle**: When mechanical/anatomy details are occluded in the primary angle, generate a second studio angle referencing the first image (`--ref`).
- **1–2 Pixel-Locked Scene Keyframes**: Edit the sharp original keyframes extracted in Phase 0 (`--ref orig_kf --ref subject_ref`), varying action phase across keyframes so the model understands motion. Then ALWAYS pixel-lock each edited scene keyframe back onto untouched original pixels:
  ```bash
  $VIDREP pixel-lock ./jobs/<name> \
    --orig ./jobs/<name>/keyframes/orig-keyframe-1-*.png \
    --edited ./jobs/<name>/refs/scene-1-raw.png \
    --name scene-1-locked
  ```
  Verify `outer_bg_mae_after < 0.05` and visually inspect the locked frame.

> **GATE 1 — Pause & Confirm with User**: Show the generated character/creature reference(s) and pixel-locked scene keyframe(s). Confirm visual design, scale, and posture before spending any video tokens.

### Phase 2 — Short Clip + 480p Flash Pre-Validation (Gate 2)
1. Pick the hardest / most informative `~4.2s` segment (peak motion, lighting change, or subject interaction) and slice it:
   ```bash
   $VIDREP slice-preview ./jobs/<name> --start <sec> --duration 4.2
   ```
2. Write `<job_dir>/prompt.txt` following `references/prompt-patterns.md`:
   - Explicitly assign roles to `视频1` and every `图片1..N` (`图片N` scene refs are explicitly marked "不作为首帧，不复制为静止画面").
   - **Explicitly release conflicting original subject pose/silhouette constraints** ("允许彻底改变原主体的身形、姿态与动作轨迹...").
   - Describe concrete observable mechanics rather than vague adjectives.
   - Specify conflict resolution priority (background/camera/color -> `视频1`; new subject design/anatomy -> studio `图片`).
3. Upload preview clip + references in exact prompt order, submit ONCE at `480p`, wait, and run preview QA:
   ```bash
   $VIDREP upload ./jobs/<name> --stage preview --refs char-main,char-struct,scene-1-locked,scene-2-locked
   $VIDREP edit-submit ./jobs/<name> --stage preview --resolution 480p --prompt-file ./jobs/<name>/prompt.txt
   $VIDREP edit-wait ./jobs/<name> --stage preview
   $VIDREP composite ./jobs/<name> --stage preview --use-onnx
   ```
   > **Spoken Dialogue / Commercial TVC / Zero-Face Multi-Ref Mode**: Add `--generate-audio` for synchronized native speech & SFX (`vidrep composite --audio-source auto` automatically preserves generated audio). When recreating human-face TVC/drama scenes without uploading real-person media, use `--refs-only` + `--mode multi_ref --ratio 9:16 --duration 12 --generate-audio` with zero-face product/creature, empty-room, and wardrobe flat-lay references.
4. Visually inspect `preview-out/preview-qa-comparison.jpg` and `preview-out/preview-qa-motion-5fps.jpg`.

> **GATE 2 — Pause & Confirm with User**: Report preview posture, mechanical motion, silhouette stability, and lighting integration. If flawed, fix references/prompt first; only proceed to Phase 3 once preview passes.

### Phase 3 — Full-Length 720p Flash Edit
1. Upload the full reference video + approved reference images (exact same order as prompt):
   ```bash
   $VIDREP upload ./jobs/<name> --stage full --refs char-main,char-struct,scene-1-locked,scene-2-locked
   ```
2. Submit ONE single full-length `720p` Flash edit and wait:
   ```bash
   $VIDREP edit-submit ./jobs/<name> --stage full --resolution 720p --prompt-file ./jobs/<name>/prompt.txt
   $VIDREP edit-wait ./jobs/<name> --stage full
   ```

### Phase 4 — Automated High-Resolution Background Restoration & Color Passthrough
Run deterministic finishing (zero additional API cost; safe to re-run with different `--mask-mode` or `--use-onnx` if needed):
```bash
# Fast pure-CV mode:
$VIDREP composite ./jobs/<name> --stage full --mask-mode auto_contour

# High-precision local ONNX mode (auto-downloads SAM 2.1 Tiny matting + Depth Anything V2 occlusion guard):
$VIDREP composite ./jobs/<name> --stage full --mask-mode auto_contour --use-onnx
```
This automatically executes:
- Deterministic FFmpeg rawvideo decode (16-bit `bgr48le` automatically for 10-bit PQ/HLG sources; 8-bit `bgr24` for 8-bit sources — zero macOS AVFoundation HDR corruption)
- Adaptive non-subject background DP temporal alignment + DIS optical-flow half-frame synthesis + ECC Euclidean registration
- Multi-subject + shadow replacement mask (`auto_contour`, or `--mask-mode ellipse`), optionally refined with **SAM 2.1 Tiny ONNX** semantic edges and **Depth Anything V2 ONNX** foreground occluder subtraction (`--use-onnx`)
- Temporally smooth global color grading (`full-graded-with-audio.mp4`)
- Local collar tone match + 5-level Laplacian pyramid blend onto untouched original resolution background (`full-composited-with-audio.mp4`)
- Dynamic source color-matrix passthrough (`zscale`: Rec.709 / HLG / 10-bit PQ) + lossless original AAC audio mux

### Phase 5 — Quantitative & Visual QA
Follow `references/qa-checklist.md`:
1. Inspect `deliverables/full-qa-comparison.jpg`, `deliverables/full-qa-motion-5fps.jpg`, `deliverables/full-qa-heatmaps.jpg`, and `deliverables/full-metrics.json`.
2. Check `segmentation.recommend_graded_fallback`: if `true` (e.g., >62% of frame changed or heavy foreground occlusion), explicitly recommend `full-graded-with-audio.mp4` alongside or instead of `full-composited-with-audio.mp4`.
3. Report actual measured `ffprobe` specs, background MAE progression (`raw -> graded -> composited`), and honest remaining limitations.
