#!/usr/bin/env bash
set -euo pipefail

# Cross-agent Skill installer for Antigravity, Claude Code, OpenAI Codex, and Cursor.
# Usage:
#   ./install-skill.sh --target all --scope global
#   ./install-skill.sh --target claude --scope project --project-dir /path/to/repo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_NAME="video-subject-replica"
SKILL_SRC="${SCRIPT_DIR}/skills/${SKILL_NAME}"

TARGET="all"
SCOPE="global"
PROJECT_DIR="$(pwd)"
MODE="symlink"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --copy) MODE="copy"; shift ;;
    --symlink) MODE="symlink"; shift ;;
    -h|--help)
      echo "Usage: $0 [--target antigravity|claude|codex|cursor|all] [--scope global|project] [--project-dir PATH] [--symlink|--copy]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "${SKILL_SRC}/SKILL.md" ]]; then
  echo "Error: skill source not found at ${SKILL_SRC}/SKILL.md" >&2
  exit 1
fi

install_one() {
  local label="$1"
  local dest_parent="$2"
  local dest="${dest_parent}/${SKILL_NAME}"
  mkdir -p "${dest_parent}"

  if [[ -L "${dest}" || -e "${dest}" ]]; then
    echo "[SKIP] ${label}: ${dest} already exists (remove manually to replace)"
    return 0
  fi

  if [[ "${MODE}" == "symlink" ]]; then
    ln -s "${SKILL_SRC}" "${dest}"
    echo "[OK] ${label}: symlinked ${dest} -> ${SKILL_SRC}"
  else
    cp -R "${SKILL_SRC}" "${dest}"
    echo "[OK] ${label}: copied ${SKILL_SRC} -> ${dest}"
  fi
}

do_target() {
  local t="$1"
  if [[ "${SCOPE}" == "global" ]]; then
    case "${t}" in
      antigravity) install_one "Antigravity (global)" "${HOME}/.gemini/config/skills" ;;
      claude)      install_one "Claude Code (global)" "${HOME}/.claude/skills" ;;
      codex)       install_one "OpenAI Codex (global)" "${HOME}/.codex/skills" ;;
      cursor)      install_one "Cursor (global)" "${HOME}/.cursor/skills" ;;
      *) echo "Unknown target: ${t}" >&2; exit 1 ;;
    esac
  elif [[ "${SCOPE}" == "project" ]]; then
    case "${t}" in
      antigravity|codex|cursor)
        install_one "Open Agent Skills (${t} project)" "${PROJECT_DIR}/.agents/skills"
        ;;
      claude)
        install_one "Claude Code (project)" "${PROJECT_DIR}/.claude/skills"
        ;;
      *) echo "Unknown target: ${t}" >&2; exit 1 ;;
    esac
  else
    echo "Unknown scope: ${SCOPE} (use global or project)" >&2
    exit 1
  fi
}

if [[ "${TARGET}" == "all" ]]; then
  if [[ "${SCOPE}" == "project" ]]; then
    do_target antigravity
    do_target claude
  else
    do_target antigravity
    do_target claude
    do_target codex
    do_target cursor
  fi
else
  do_target "${TARGET}"
fi

echo ""
echo "Done! Verify CLI availability with:"
echo "  uv run --project \"${SCRIPT_DIR}\" vidrep preflight"
