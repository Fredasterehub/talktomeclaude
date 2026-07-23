"""Voice-fireable commands discovered from a Claude Code session.

The live session supplies command metadata through its system/init event,
while user-owned flags persist separately so refreshed metadata never erases
the operator's choices or command history.
"""

import json
from pathlib import Path

from talktomeclaude import config

CATALOG_FILE = "command_catalog.json"

_BUILTIN_INTERACTIVE_COMMANDS = frozenset(
    {
        "add-dir",
        "agents",
        "clear",
        "compact",
        "config",
        "cost",
        "help",
        "init",
        "mcp",
        "memory",
        "migrate-installer",
        "model",
        "permissions",
        "pr-comments",
        "review",
        "status",
        "terminal-setup",
        "vim",
    }
)


def parse_init_event(event: dict) -> list[dict]:
    """Return the voice-fireable commands advertised by a system/init event."""
    if not isinstance(event, dict):
        return []

    records = []
    for field in ("slash_commands", "skills"):
        entries = event.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, str):
                command_id = entry
                namespace = ""
                description = ""
                mutating = True
            elif isinstance(entry, dict):
                command_id = entry.get("name", entry.get("id"))
                namespace = entry.get("namespace", "")
                description = entry.get("description", "")
                mutating = not entry.get("read_only", False)
            else:
                continue

            if command_id in _BUILTIN_INTERACTIVE_COMMANDS:
                continue
            records.append(
                {
                    "id": command_id,
                    "namespace": namespace,
                    "description": description,
                    "mutating": mutating,
                    "enabled": True,
                    "favorite": False,
                    "fire_count": 0,
                }
            )
    return records


def qualified_id(record: dict) -> str:
    """The catalog identity contract: ``namespace:id``, bare id at top level."""
    namespace = record.get("namespace") or ""
    return f"{namespace}:{record['id']}" if namespace else str(record["id"])


def merge_with_saved(init_records: list[dict], saved_flags: dict) -> list[dict]:
    """Refresh session metadata while preserving each command's saved flags.

    Flags are keyed by qualified identity; a legacy bare-id key is honored
    only while the bare id names exactly one command, and a qualified key
    always wins over it.
    """
    counts: dict = {}
    for record in init_records:
        counts[record["id"]] = counts.get(record["id"], 0) + 1
    merged = []
    for record in init_records:
        flags = saved_flags.get(qualified_id(record))
        if not isinstance(flags, dict) and counts[record["id"]] == 1:
            flags = saved_flags.get(record["id"])
        if not isinstance(flags, dict):
            flags = {}
        merged.append(
            {
                "id": record["id"],
                "namespace": record["namespace"],
                "description": record["description"],
                "mutating": record["mutating"],
                "enabled": flags.get("enabled", record["enabled"]),
                "favorite": flags.get("favorite", record["favorite"]),
                "fire_count": flags.get("fire_count", record["fire_count"]),
            }
        )
    return merged


def catalog_path() -> Path:
    """Path to the user-owned command catalog flags."""
    return config.config_dir() / CATALOG_FILE


def load_saved_flags() -> dict:
    """Load persisted command flags, returning empty state when unavailable."""
    try:
        with catalog_path().open(encoding="utf-8") as handle:
            saved_flags = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return saved_flags if isinstance(saved_flags, dict) else {}


def save_flags(records: list[dict]) -> None:
    """Persist the user-owned flags for every discovered command."""
    saved_flags = {
        qualified_id(record): {
            "enabled": record["enabled"],
            "favorite": record["favorite"],
            "fire_count": record["fire_count"],
        }
        for record in records
    }
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    swap = path.with_name(path.name + ".tmp")
    swap.write_text(json.dumps(saved_flags, indent=2) + "\n", encoding="utf-8")
    swap.replace(path)


def load_catalog(event: dict) -> list[dict]:
    """Load the event's current commands with the user's saved flags."""
    return merge_with_saved(parse_init_event(event), load_saved_flags())
