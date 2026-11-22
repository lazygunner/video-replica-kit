# video-replica-kit（视频复刻调色 Skill & CLI）

简体中文 | [English](README.en.md)

**跨平台通用的参考视频主体复刻与自动化调色合成工具包**，一套 Skill + Pure Python CLI（`vidrep`）无缝适配 **Antigravity、Claude Code、OpenAI Codex、Cursor** 与命令行。

核心思路：**让 `doubao-seedance-2-5-flash-261115` 专注它最擅长的语义主体替换、透视光影与音画生成，把模型边界之外的稳定性全部交给确定性工程**（像素锁定参考图、480p 低成本预验证、DP/DIS/ECC 对齐、本地 SAM 2.1 + Depth Anything V2 ONNX 精抠、5 层拉普拉斯原背景融合、10-bit PQ/HLG/Rec.709 `zscale` 无损色彩透传）。

---

## 一句话直接用（复制到 Agent 客户端即可）

### 1. 一句话安装 Skill 到你的 Agent 客户端
直接把下面这句话发给 **Antigravity / Claude Code / Codex / Cursor**：

```text
请克隆 https://github.com/lazygunner/video-replica-kit.git 到本地，运行 ./install-skill.sh --target all --scope global 安装视频复刻调色 Skill，并执行 uv run vidrep preflight 检查环境与引导我配置 API Key。
```

### 2. 一句话发起视频复刻（安装好 Skill 后直接复制使用）
把方括号内容换成你的视频和需求，直接发给 Agent：

```text
使用 video-subject-replica skill 复刻参考视频【视频路径或链接】，把原视频中的【原主体】替换为【目标主体与风格要求】，严格保留原片运镜、场景和色调；先出参考图和 480p 短片预验证，我确认后再出 720p 并用本地 ONNX 精抠调色合成。
```

> **广告 / 短剧口播场景（含真人出镜、需换中文配音）可直接复制**：
> ```text
> 使用 video-subject-replica skill，参考视频【视频路径】的镜头节奏与剧情，把主角换成【新角色】，产品/元素换成【新元素】；若原视频含真人正脸，自动切换零人像 multi_ref 模式 + 原生中文配音，先出 480p 预验证给我审片。
> ```

---

## 30 秒快速配置

### 1. 前置依赖
- Python `>= 3.10` + [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg`（macOS：`brew install ffmpeg`）

### 2. 配置自己的 API Key（仓库不含任何密钥）
支持环境变量、项目 `.env` 或全局 `~/.config/vidrep/config.toml`（优先级：`ENV > .env > config.toml`）：

```bash
git clone https://github.com/lazygunner/video-replica-kit.git
cd video-replica-kit
mkdir -p ~/.config/vidrep && cp config.example.toml ~/.config/vidrep/config.toml
# 编辑 ~/.config/vidrep/config.toml，填入你自己的 ARK_API_KEY 与 TOS 凭证
uv run vidrep preflight
```

`vidrep preflight` 返回 `"ok": true` 即准备就绪。

### 3. 一键安装 Skill 到主流 Agent 客户端
```bash
# 全局安装到 Antigravity / Claude Code / Codex / Cursor
./install-skill.sh --target all --scope global

# 或仅安装到指定项目目录
./install-skill.sh --target all --scope project --project-dir /path/to/your/project
```

---

## 核心能力一览

| 模块 | 解决什么问题 |
|---|---|
| **Flash 专属白名单 + 计费锁** | 默认锁定 `doubao-seedance-2-5-flash-261115`，强制先跑 `~4.2s` `480p` 预验证（不足 4s 自动 `tpad` 补齐），单阶段幂等防重复扣费 |
| **参考图工程 + 像素锁定** | 内置 Seedream 5.5 / Gemini / 本地图生图，自动把编辑后关键帧的外围背景像素锁回原片（`MAE < 0.05`） |
| **零人像 `multi_ref` + 原生配音** | 支持 `--refs-only --mode multi_ref --generate-audio`，纯商品/空景/服装静物图合规生成音画同步的中文口播 TVC 与短剧 |
| **本地 ONNX 精抠与前景遮挡** | `--use-onnx` 首次运行自动下载 **SAM 2.1 Tiny**（精抠）+ **Depth Anything V2 Small**（前景遮挡保护），支持多尺度双主体 |
| **真 10-bit PQ/HLG 无损透传** | 绕开 macOS `cv2.VideoCapture` 的 AVFoundation 双重 HDR 偏色，16-bit 内部合成 + `zscale` 矩阵回写 Rec.709 / HLG / 10-bit PQ |

---

## CLI 常用命令速查

### 模式 A：参考视频局部主体替换（`edit` + 本地 ONNX 原背景合成）
```bash
# 1. 初始化与原片拆解（分镜、HDR 检测、2fps 总览图、清晰关键帧）
uv run vidrep init ./jobs/demo --source /path/to/reference.mp4
uv run vidrep probe ./jobs/demo --keyframes 0.22,0.68

# 2. 生成定妆图并像素锁定关键帧
uv run vidrep ref-gen ./jobs/demo --name subject --aspect 1:1 --prompt "..."
uv run vidrep pixel-lock ./jobs/demo --orig ./jobs/demo/keyframes/orig-keyframe-1-*.png \
  --edited ./jobs/demo/refs/scene-raw.png --name scene-locked

# 3. 低成本 480p 预验证（审片通过后再跑 --stage full --resolution 720p）
uv run vidrep slice-preview ./jobs/demo --start 0.0 --duration 4.2
uv run vidrep upload ./jobs/demo --stage preview --refs subject,scene-locked
uv run vidrep edit-submit ./jobs/demo --stage preview --resolution 480p --prompt-file ./jobs/demo/prompt.txt
uv run vidrep edit-wait ./jobs/demo --stage preview
uv run vidrep composite ./jobs/demo --stage preview --use-onnx
```

### 模式 B：零人像多图参考 + 中文音画同步（`multi_ref`）
```bash
uv run vidrep upload ./jobs/tvc --stage preview --refs-only --refs ref_product,ref_room,ref_outfit
uv run vidrep edit-submit ./jobs/tvc --stage preview --mode multi_ref --resolution 480p \
  --ratio 9:16 --duration 12 --generate-audio --prompt-file ./jobs/tvc/prompt.txt
uv run vidrep edit-wait ./jobs/tvc --stage preview
```

---

## 目录与进阶文档

- Skill 规范：[skills/video-subject-replica/SKILL.md](skills/video-subject-replica/SKILL.md)
- 实战深度教程：[docs/tutorial-seedance-2.5-flash.md](docs/tutorial-seedance-2.5-flash.md)
- 自动化测试：`uv run --extra dev pytest -v`
- License：MIT
