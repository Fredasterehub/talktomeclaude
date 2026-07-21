#!/usr/bin/env bash
# Claude Code Stop hook — speaks Claude's final reply via talktomeclaude.
# Registered async: it must never block Claude Code, so every failure
# (CLI missing, TTS unavailable) exits 0 silently.
ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

for candidate in "$ROOT/.venv/bin/talktomeclaude" "$(command -v talktomeclaude)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    exec "$candidate" hook stop "$@"
  fi
done

exit 0
