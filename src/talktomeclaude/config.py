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
DEFAULT_WAKE_PHRASE = "yo claude"
CLAUDE_PERMISSIONS = ("off", "skip", "acceptEdits", "bypassPermissions")


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


def onboarding_version() -> int:
    """The persisted onboarding version; zero means onboarding is incomplete."""
    value = get_value("onboarding-version", 0)
    return value if type(value) is int else 0


def set_onboarding_version(version: int) -> None:
    set_value("onboarding-version", version)


def onboarding_needed(current: int) -> bool:
    return onboarding_version() < current


def claude_permissions() -> str:
    """The persisted Claude Code permission posture; off is the safe default."""
    value = load().get("claude-permissions")
    return value if value in CLAUDE_PERMISSIONS else "off"


def set_claude_permissions(value: str) -> None:
    if value not in CLAUDE_PERMISSIONS:
        raise ValueError(
            f"unknown Claude permission posture {value!r}: expected one of "
            f"{', '.join(CLAUDE_PERMISSIONS)}"
        )
    set_value("claude-permissions", value)


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


def barge_in_enabled() -> bool:
    """Whether the listen loop may be interrupted while Claude is still
    speaking. Off by default: half-duplex is safe on every machine, and
    full-duplex barge-in is opt-in and gated on capable audio hardware."""
    return load().get("barge-in", "off") == "on"


def set_barge_in(enabled: bool) -> None:
    set_value("barge-in", "on" if enabled else "off")


def wake_word_enabled() -> bool:
    """Whether always-on listening waits for a wake word before recording."""
    return load().get("wake-word", "off") == "on"


def set_wake_word(enabled: bool) -> None:
    set_value("wake-word", "on" if enabled else "off")


def wake_phrase() -> str:
    """The phrase associated with the user's configured wake-word model."""
    value = load().get("wake-phrase")
    return value if isinstance(value, str) and value.strip() else DEFAULT_WAKE_PHRASE


def set_wake_phrase(value: str) -> None:
    set_value("wake-phrase", value)


def wake_model_path() -> str | None:
    """Path to the trained wake-word model for the configured phrase, or None
    when no detector model has been installed yet."""
    value = load().get("wake-model")
    return value if isinstance(value, str) and value.strip() else None


def set_wake_model_path(value: str | None) -> None:
    """Persist the wake-word model path, or clear it when empty."""
    if value and value.strip():
        set_value("wake-model", value.strip())
    else:
        settings = load()
        settings.pop("wake-model", None)
        save(settings)


def onboarding_completed_at() -> str | None:
    """ISO timestamp of the last completed onboarding run, or None."""
    value = load().get("onboarding-completed-at")
    return value if isinstance(value, str) and value.strip() else None


def set_onboarding_completed_at(value: str) -> None:
    set_value("onboarding-completed-at", value)


def default_voice_name() -> str | None:
    """The user's chosen default voice, or None to auto-select the best
    available voice. The name is validated against the registry when it is
    used, not here, so a removed voice degrades gracefully to auto-select."""
    value = load().get("default-voice")
    return value.strip() if isinstance(value, str) and value.strip() else None


def set_default_voice(value: str | None) -> None:
    """Persist the default voice name, or clear it (auto-select) when empty."""
    if value and value.strip():
        set_value("default-voice", value.strip())
    else:
        settings = load()
        settings.pop("default-voice", None)
        save(settings)
