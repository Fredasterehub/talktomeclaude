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
    """Return the voice-fireable commands advertised by a system/init event.

    Records are deduped by case-folded qualified identity (first-wins), so a
    session that advertises the same command twice never yields the impossible
    ``('foo', 'foo')`` ambiguity or silently collapses on persistence.
    """
    if not isinstance(event, dict):
        return []

    records = []
    seen: set = set()
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
                arg_schema = None
            elif isinstance(entry, dict):
                command_id = entry.get("name", entry.get("id"))
                namespace = entry.get("namespace", "")
                description = entry.get("description", "")
                mutating = not entry.get("read_only", False)
                arg_schema = entry.get("arg_schema")
            else:
                continue

            if not command_id or command_id in _BUILTIN_INTERACTIVE_COMMANDS:
                continue
            identity = (
                f"{namespace}:{command_id}" if namespace else str(command_id)
            ).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            records.append(
                {
                    "id": command_id,
                    "namespace": namespace,
                    "description": description,
                    "mutating": mutating,
                    "enabled": True,
                    "favorite": False,
                    "fire_count": 0,
                    "arg_schema": arg_schema,
                }
            )
    return records


def qualified_id(record: dict) -> str:
    """The catalog identity contract: ``namespace:id``, bare id at top level."""
    namespace = record.get("namespace") or ""
    return f"{namespace}:{record['id']}" if namespace else str(record["id"])


def _persist_key(record: dict) -> str:
    """Unambiguous persistence key: always ``namespace:id`` — with a leading
    colon for a top-level command — so a legacy bare ``id`` key can never be
    mistaken for a canonical top-level entry on a namespace collision."""
    namespace = record.get("namespace") or ""
    return f"{namespace}:{record['id']}"


def required_slots(record: dict) -> list[str]:
    """The argument slots an init event marks required for a command.

    ``arg_schema`` items are required when they are a bare slot name or a dict
    with a truthy ``required``; anything else (or an absent/unknown schema) is
    treated as no required slots, so ordinary no-arg commands keep firing.
    """
    schema = record.get("arg_schema")
    slots: list[str] = []
    if isinstance(schema, list):
        for item in schema:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict) and item.get("required"):
                name = str(item.get("name", "")).strip()
            else:
                continue
            if name and name not in slots:
                slots.append(name)
    return slots


def merge_with_saved(init_records: list[dict], saved_flags: dict) -> list[dict]:
    """Refresh session metadata while preserving each command's saved flags.

    Flags are keyed by the unambiguous persist key (``namespace:id``, with a
    leading colon at top level). A legacy bare-id key is honored only as a
    fallback and only while the bare id names exactly one command, so legacy
    ``deploy`` flags never leak onto a top-level ``deploy`` when a ``web:deploy``
    also exists; the persist key always wins over it.
    """
    counts: dict = {}
    for record in init_records:
        counts[record["id"]] = counts.get(record["id"], 0) + 1
    merged = []
    for record in init_records:
        flags = saved_flags.get(_persist_key(record))
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
                "arg_schema": record.get("arg_schema"),
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


def roster_from_saved(saved_flags: dict | None = None) -> list[dict]:
    """Rebuild an initial fireable roster from the persisted flags so commands
    resolve on the very first utterance, before the session's system/init event
    arrives to refresh the authoritative metadata.

    Metadata unknown until that event degrades safely: descriptions are blank,
    there is no arg schema, and commands are treated as mutating so a spoken
    confirmation is still required. A missing/unreadable file yields no roster.
    """
    flags = load_saved_flags() if saved_flags is None else saved_flags
    if not isinstance(flags, dict):
        return []
    records: list[dict] = []
    seen: set = set()
    for key, state in flags.items():
        if not isinstance(state, dict):
            continue
        persist_key = str(key)
        if ":" in persist_key:
            namespace, _, command_id = persist_key.partition(":")
        else:
            namespace, command_id = "", persist_key
        if not command_id:
            continue
        identity = (
            f"{namespace}:{command_id}" if namespace else command_id
        ).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "id": command_id,
                "namespace": namespace,
                "description": "",
                "mutating": True,
                "enabled": state.get("enabled", True),
                "favorite": state.get("favorite", False),
                "fire_count": state.get("fire_count", 0),
                "arg_schema": None,
            }
        )
    return records


def save_flags(records: list[dict]) -> None:
    """Persist the user-owned flags for every discovered command."""
    saved_flags = {
        _persist_key(record): {
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
