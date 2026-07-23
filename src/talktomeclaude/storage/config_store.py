"""Versioned configuration storage built on atomic JSON transactions."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .atomic import AtomicJsonTransaction, AtomicStorageError

SCHEMA_KEY = "_schema-version"
CURRENT_SCHEMA_VERSION = 1
Migration = Callable[[dict[str, Any]], Mapping[str, Any]]


class ConfigMigrationError(AtomicStorageError):
    """Configuration could not be migrated without risking existing state."""


def _schema_one(settings: dict[str, Any]) -> Mapping[str, Any]:
    migrated = dict(settings)
    migrated[SCHEMA_KEY] = 1
    return migrated


DEFAULT_MIGRATIONS: dict[int, Migration] = {0: _schema_one}


class ConfigStore:
    """Cross-process transactional access to one additive config document."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        timeout: float = 5.0,
        recovery_dir: str | os.PathLike[str] | None = None,
        current_schema: int = CURRENT_SCHEMA_VERSION,
        migrations: Mapping[int, Migration] | None = None,
        on_event: Callable[[str], None] | None = None,
        phase_hook: Callable[[str], None] | None = None,
        purpose: str = "config",
    ) -> None:
        self.atomic = AtomicJsonTransaction(
            path,
            timeout=timeout,
            purpose=purpose,
            on_event=on_event,
            phase_hook=phase_hook,
        )
        self.path = self.atomic.path
        self.lock_identity = self.atomic.lock_identity
        self.recovery_dir = (
            type(self.path)(recovery_dir)
            if recovery_dir is not None
            else self.path.parent / "recovery"
        )
        self.current_schema = current_schema
        self.migrations = dict(DEFAULT_MIGRATIONS if migrations is None else migrations)
        self._on_event = on_event

    def _emit(self, code: str) -> None:
        if self._on_event is not None:
            self._on_event(code)

    @staticmethod
    def _version(settings: Mapping[str, Any]) -> int:
        value = settings.get(SCHEMA_KEY, 0)
        if type(value) is not int or value < 0:
            raise ConfigMigrationError("configuration schema version is invalid")
        return value

    def _recovery_copy(self) -> Path | None:
        try:
            original = self.path.read_bytes()
        except FileNotFoundError:
            return None
        digest = hashlib.sha256(original).hexdigest()
        destination = self.recovery_dir / f"config-{digest}.json"
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != original:
                raise ConfigMigrationError("content-addressed recovery copy mismatch")
            return destination
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.recovery_dir
        )
        temporary = type(self.path)(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(original)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return destination

    def _migrate(self, settings: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        try:
            version = self._version(settings)
        except ConfigMigrationError:
            # Invalid durable metadata is itself a failed migration boundary:
            # preserve the exact source before surfacing the error.
            self._recovery_copy()
            raise
        if version > self.current_schema:
            return dict(settings), False
        if version == self.current_schema:
            return dict(settings), False

        self._recovery_copy()
        migrated = dict(settings)
        while version < self.current_schema:
            migration = self.migrations.get(version)
            if migration is None:
                raise ConfigMigrationError(f"no migration from schema version {version}")
            try:
                candidate = migration(dict(migrated))
            except Exception as exc:
                raise ConfigMigrationError(
                    f"migration from schema version {version} failed"
                ) from exc
            if not isinstance(candidate, Mapping):
                raise ConfigMigrationError("migration result must be an object")
            migrated = dict(candidate)
            next_version = self._version(migrated)
            if next_version != version + 1:
                raise ConfigMigrationError(
                    f"migration must advance schema from {version} to {version + 1}"
                )
            version = next_version
        self._emit("config_migrated")
        return migrated, True

    def load(self) -> dict[str, Any]:
        # Reads are deliberately side-effect free.  The first subsequent
        # save/update performs and persists any required additive migration.
        return self.atomic.read()

    def save(self, settings: Mapping[str, Any]) -> None:
        replacement = dict(settings)

        def replace(current: dict[str, Any]) -> Mapping[str, Any]:
            current_version = self._version(current)
            migrated, _changed = self._migrate(replacement)
            migrated_version = self._version(migrated)
            if (
                current_version > self.current_schema
                and migrated_version != current_version
            ):
                raise ConfigMigrationError(
                    "configuration schema is owned by the store and cannot be downgraded"
                )
            return migrated

        self.atomic.update(replace)

    def update(
        self, mutator: Callable[[dict[str, Any]], Mapping[str, Any] | None]
    ) -> dict[str, Any]:
        def migrate_and_update(settings: dict[str, Any]) -> Mapping[str, Any]:
            migrated, _changed = self._migrate(settings)
            working = dict(migrated)
            result = mutator(working)
            updated = working if result is None else dict(result)
            owned_version = self._version(migrated)
            if self._version(updated) != owned_version:
                raise ConfigMigrationError(
                    "configuration schema is owned by the store and cannot be changed"
                )
            return updated

        return self.atomic.update(migrate_and_update)
