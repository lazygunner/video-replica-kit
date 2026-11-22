---
name: video-subject-replica
description: 基于 Seedance 2.5 Flash 与本地 ONNX（SAM 2.1 + Depth Anything V2）的参考视频主体复刻与自动化调色合成工作流，支持像素锁定参考图、480p 预验证、10-bit PQ/HLG 色彩无损透传与零人像多图音画同步生成。
---

# 视频复刻调色（`video-subject-replica`）

当用户需要复刻参考视频、替换其中的前景主体（人物、神话生物/怪物、载具、商品或道具）并保留原片运镜、构图、背景纹理、光影色调与音频时使用本 Skill。跨平台兼容 Antigravity、Claude Code、OpenAI Codex、Cursor 及独立 CLI（`vidrep`）。

## 不可妥协的护栏

1. **默认仅用 Flash 模型**：始终使用 `doubao-seedance-2-5-flash-261115`，除非用户明确授权切换（`--allow-non-flash`）。
2. **先做短片 480p 预验证**：全片 `720p` 前必须先切 `~4.2s` 片段（`vidrep slice-preview`，不足 4s 自动 `tpad` 补齐以满足 Ark `edit` 模式 `[4, 30]s` 限制）跑 `480p` 预验证，经用户确认后再跑全片。
3. **单次计费幂等锁**：严禁盲目重试提交；`vidrep edit-submit` 每阶段写锁防重复扣费（同步 HTTP 4xx 零扣费驳回自动解锁）。
4. **禁止盲目 HDR→SDR 色调映射**：检测到 HDR/HLG/10-bit PQ 源片时直接原片送编辑，由 `vidrep composite` 通过 `zscale` 无损恢复色彩矩阵与 VUI 标签。
5. **每道门禁必须人工审图/审片**：绝不跳过视觉检查直接进入下一阶段。
6. **零人像合规降级（`--mode multi_ref`）**：若原视频或写实人像触发平台真人隐私检测（`PrivacyInformation`），禁止任何规避/打码绕过，改为 `--refs-only --mode multi_ref --generate-audio`，仅传商品/生物、纯空景与服装平铺静物图，由文字驱动人物与同步配音。

## 调用方式

```bash
VIDREP="uv run --project /path/to/video-replica-kit vidrep"
$VIDREP preflight
```

---

## 五阶段标准工作流

### 阶段 0：环境自检与原片拆解
```bash
$VIDREP init ./jobs/<name> --source /path/to/reference.mp4
$VIDREP probe ./jobs/<name> --keyframes 0.22,0.68
```
检查 `reference-contact-sheet.jpg`、`shot_count`（多镜头时与用户确认分镜策略）及 `color_profile.is_hdr`（HDR 详见 `references/color-pipelines.md`）。

### 阶段 1：参考图工程与背景像素锁定（门禁 1）
参考 `references/prompt-patterns.md` 准备参考图：
- 生成匹配主镜头机位的灰底棚拍主体定妆图（单视角，禁用多视角拼图）；
- 编辑 1–2 张关键帧并用 `pixel-lock` 锁回原片未改动背景像素（`outer_bg_mae_after < 0.05`）：
  ```bash
  $VIDREP pixel-lock ./jobs/<name> --orig ./jobs/<name>/keyframes/orig-keyframe-1-*.png --edited ./jobs/<name>/refs/scene-1-raw.png --name scene-1-locked
  ```
> **门禁 1**：展示参考图给用户确认造型、比例与质感后，再消耗视频 Token。

### 阶段 2：短片 480p Flash 预验证（门禁 2）
```bash
$VIDREP slice-preview ./jobs/<name> --start <sec> --duration 4.2
$VIDREP upload ./jobs/<name> --stage preview --refs ref1,ref2,scene-1-locked
$VIDREP edit-submit ./jobs/<name> --stage preview --resolution 480p --prompt-file ./jobs/<name>/prompt.txt
$VIDREP edit-wait ./jobs/<name> --stage preview
$VIDREP composite ./jobs/<name> --stage preview --use-onnx
```
- 含中文口播/广告场景加 `--generate-audio`（`composite --audio-source auto` 自动保留生成音轨）；
- 零人像多图模式加 `--refs-only` 与 `--mode multi_ref --ratio 9:16 --duration 12 --generate-audio`。
> **门禁 2**：审阅 `preview-out/preview-qa-comparison.jpg` 与 `preview-qa-motion-5fps.jpg`，确认无误再升全片。

### 阶段 3：全片 720p Flash 生成
```bash
$VIDREP upload ./jobs/<name> --stage full --refs ref1,ref2,scene-1-locked
$VIDREP edit-submit ./jobs/<name> --stage full --resolution 720p --prompt-file ./jobs/<name>/prompt.txt
$VIDREP edit-wait ./jobs/<name> --stage full
```

### 阶段 4：本地 ONNX 精抠、调色与原背景拉普拉斯融合
```bash
$VIDREP composite ./jobs/<name> --stage full --mask-mode auto_contour --use-onnx
```
自动执行：FFmpeg 确定性解码（10-bit 源片自动启用 16-bit 内部管线）→ DP 时序对齐 + DIS 光流半帧 + ECC 亚像素配准 → 双尺度多主体分割 + 自动下载 **SAM 2.1 Tiny** 精抠与 **Depth Anything V2** 前景遮挡扣除 → 时序平滑调色 + 5 层拉普拉斯融合 → `zscale` 原色彩矩阵回写与无损音轨封装。

### 阶段 5：量化与视觉审片交付
按 `references/qa-checklist.md` 核验 `full-qa-comparison.jpg`、`full-qa-motion-5fps.jpg`、`full-qa-heatmaps.jpg` 与 `full-metrics.json`（关注 `bg_mae_composited_vs_orig` 与 `recommend_graded_fallback`），如实汇报实测指标与交付路径。
