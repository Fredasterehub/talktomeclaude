"""Persistent configuration state shared by the CLI and the Claude Code hook.

Settings live in a single JSON file under ``CLAUDE_PLUGIN_DATA`` when Claude
Code provides it (that directory survives plugin updates), falling back to
the user's XDG config directory for standalone CLI use.
"""

import json
import os
from pathlib import Path

_CONFIG_FILE = "config.json"

RECORDING_MODES = ("always-on", "push-to-talk", "push-toggle")
DEFAULT_RECORDING_MODE = "push-to-talk"


def config_dir() -> Path:
    """Directory holding persistent state."""
    plugin_data = os.environ.get("CLAUDE_PLUGIN_DATA")
    if plugin_data:
        return Path(plugin_data)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "talktomeclaude"


def config_path() -> Path:
    return config_dir() / _CONFIG_FILE


def load() -> dict:
    try:
        with config_path().open(encoding="utf-8") as handle:
            settings = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def save(settings: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    swap = path.with_name(path.name + ".tmp")
    swap.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    swap.replace(path)


def get_value(key: str, default=None):
    return load().get(key, default)


def set_value(key: str, value) -> None:
    settings = load()
    settings[key] = value
    save(settings)


def recording_mode() -> str:
    """The persisted recording mode; push-to-talk is the reliable default."""
    value = load().get("recording-mode")
    return value if value in RECORDING_MODES else DEFAULT_RECORDING_MODE


def set_recording_mode(mode: str) -> None:
    if mode not in RECORDING_MODES:
        raise ValueError(
            f"unknown recording mode {mode!r}: expected one of {', '.join(RECORDING_MODES)}"
        )
    set_value("recording-mode", mode)


def voice_assist_enabled() -> bool:
    """The full-mute switch the Stop hook consults before speaking."""
    return load().get("voice-assist", "on") == "on"


def set_voice_assist(enabled: bool) -> None:
    set_value("voice-assist", "on" if enabled else "off")


def remote() -> str | None:
    """The persisted SSH target (``user@host``) Claude Code runs on, or None
    for a fully local install."""
    value = load().get("remote")
    return value if isinstance(value, str) and value.strip() else None


def set_remote(value: str | None) -> None:
    """Persist the SSH target, or clear it (local) when value is empty."""
    if value and value.strip():
        set_value("remote", value.strip())
    else:
        settings = load()
        settings.pop("remote", None)
        save(settings)


def remote_cwd() -> str | None:
    """The persisted project directory for remote Claude sessions, or None
    to use the remote login shell's home directory."""
    value = load().get("remote-cwd")
    return value if isinstance(value, str) and value.strip() else None


def set_remote_cwd(value: str | None) -> None:
    """Persist the remote project directory, or clear it when empty."""
    if value and value.strip():
        set_value("remote-cwd", value)
    else:
        settings = load()
        settings.pop("remote-cwd", None)
        save(settings)
