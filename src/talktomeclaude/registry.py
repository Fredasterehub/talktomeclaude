"""User voice registry: bring-your-own Piper voices and cloned voices.

The three bundled public-domain voices live in :mod:`catalog`; this registry
holds every voice the user adds beyond them. Records live in a single
``voices.json`` under the configuration directory, and cloned-voice reference
clips are copied into a ``voice-refs/`` subdirectory (stored by basename, so the
registry keeps working if the config directory itself is moved).

Design notes:
- Stdlib only, and it imports the *catalog* (leaf data), never the synthesis
  engines — so :mod:`tts` can build on the registry without a circular import.
- Reads never crash on a corrupt/absent file; writes quarantine a corrupt file
  before overwriting so a single damaged byte can never silently wipe the set.
- Writes go through a unique ``mkstemp`` temp with ``fsync`` + ``os.replace``.
  This is a single-user, manual-CLI tool, so it does not add cross-process
  locking — the same deliberate simplicity as :mod:`config`.
"""

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Callable
from typing import Mapping, TypeVar

from talktomeclaude.catalog import BUNDLED_VOICE_NAMES
from talktomeclaude.config import config_dir
from talktomeclaude.storage import AtomicJsonTransaction, AtomicStorageError

ENGINES = ("piper", "clone", "f5")
_RESERVED = frozenset({"default", "none", "auto"})
# A voice name is a JSON key, a CLI argument and the stem of a copied reference
# file, so it is restricted to a filesystem- and shell-safe charset (validated
# with fullmatch: no leading dot, no separators, and crucially no trailing
# newline, which `$` would have allowed).
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
_AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aiff", ".aif"}
)
_REGISTRY_FILE = "voices.json"
_SCHEMA_VERSION = 1


class RegistryError(RuntimeError):
    """Raised when a registry operation is invalid or a record is corrupt."""


_Result = TypeVar("_Result")


def _under_registry_lock(operation: Callable[[], _Result]) -> _Result:
    """Serialize registry plus reference-asset mutations across processes."""

    result: list[_Result] = []
    callback_error: list[BaseException] = []
    transaction = AtomicJsonTransaction(
        config_dir() / ".voice-registry-guard.json",
        purpose="voice-registry",
    )

    def run(state: dict) -> dict:
        try:
            result.append(operation())
        except BaseException as exc:
            callback_error.append(exc)
            raise
        return state

    try:
        transaction.update(run)
    except RegistryError:
        raise
    except (OSError, AtomicStorageError) as exc:
        if callback_error:
            raise callback_error[0]
        raise RegistryError("voice registry lock or transaction failed") from exc
    return result[0]


@dataclass(frozen=True)
class RegisteredVoice:
    """A user-registered voice: an engine plus the parameters that drive it.

    ``params`` is a read-only mapping (mutating it would be silently discarded on
    the next load). For a clone voice, ``params['reference']`` is the resolved
    absolute path to the copied clip.
    """

    name: str
    engine: str
    params: Mapping
    language: str
    license: str
    provenance: str


def registry_path() -> Path:
    return config_dir() / _REGISTRY_FILE


def refs_dir() -> Path:
    """Directory holding reference clips copied in for cloned voices."""
    return config_dir() / "voice-refs"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _empty() -> dict:
    return {"version": _SCHEMA_VERSION, "voices": {}}


def _coerce(data) -> dict | None:
    if not isinstance(data, dict) or not isinstance(data.get("voices"), dict):
        return None
    return data


def _load_for_read() -> dict:
    """Read the registry, tolerating an absent or corrupt file as empty.
    Reads (list/get) must never crash on a damaged file — including invalid
    UTF-8, which decodes as a ValueError rather than an OSError."""
    try:
        raw = registry_path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return _empty()
    try:
        parsed = _coerce(json.loads(raw))
    except json.JSONDecodeError:
        return _empty()
    return parsed if parsed is not None else _empty()


def _quarantine(path: Path) -> None:
    """Move a corrupt registry aside before it is overwritten. Failure to do so
    aborts the mutation, so recoverable data is never silently discarded."""
    backup = path.with_name(path.name + ".corrupt")
    counter = 0
    while backup.exists():
        counter += 1
        backup = path.with_name(f"{path.name}.corrupt.{counter}")
    try:
        path.replace(backup)
    except OSError as exc:
        raise RegistryError(
            f"the registry at {path} is corrupt and could not be quarantined ({exc}); "
            "refusing to overwrite it — move it aside by hand"
        ) from exc


def _load_for_write() -> dict:
    """Read the registry for a mutation. An absent file is empty; a present but
    corrupt file (bad JSON, wrong shape, or invalid UTF-8) is quarantined first
    so the imminent save cannot silently drop recoverable data; a transient read
    error aborts the mutation."""
    path = registry_path()
    corrupt = False
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _empty()
    except UnicodeDecodeError:
        corrupt, raw = True, ""
    except OSError as exc:
        raise RegistryError(f"cannot read the voice registry at {path}: {exc}") from exc
    if not corrupt:
        try:
            parsed = _coerce(json.loads(raw))
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return parsed
    _quarantine(path)  # raises if it cannot preserve the corrupt bytes
    return _empty()


def _save(data: dict) -> None:
    path = registry_path()
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(directory))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RegistryError(f"failed to write the voice registry at {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Record <-> value object
# --------------------------------------------------------------------------- #
def _to_voice(name: str, record: dict) -> RegisteredVoice:
    raw_engine = record.get("engine")
    engine: str = raw_engine if isinstance(raw_engine, str) else ""
    raw_params = record.get("params")
    params = dict(raw_params) if isinstance(raw_params, dict) else {}
    # Storage keeps the reference by basename; runtime exposes the absolute path.
    if engine in {"clone", "f5"} and isinstance(params.get("reference"), str):
        params["reference"] = str(refs_dir() / Path(params["reference"]).name)
    return RegisteredVoice(
        name=name,
        engine=engine,
        params=MappingProxyType(params),
        language=str(record.get("language", "")),
        license=str(record.get("license", "")),
        provenance=str(record.get("provenance", "")),
    )


def list_voices() -> list[RegisteredVoice]:
    """All well-formed registered voices, name-sorted. Records with an unknown
    engine or a malformed shape are skipped rather than crashing the listing."""
    voices = _load_for_read()["voices"]
    return [
        _to_voice(name, record)
        for name, record in sorted(voices.items())
        if isinstance(record, dict) and record.get("engine") in ENGINES
    ]


def get(name: str) -> RegisteredVoice | None:
    """Return the registered voice, or None if no such name exists. A present
    but corrupt record raises RegistryError (it is a real problem, not absence)."""
    record = _load_for_read()["voices"].get(name)
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("engine") not in ENGINES:
        raise RegistryError(f"registered voice {name!r} has a corrupt or unsupported record")
    return _to_voice(name, record)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_new_name(name: str, existing: dict) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise RegistryError(
            f"invalid voice name {name!r}: use letters, digits, '.', '-' or '_' "
            "(1-64 chars, starting with a letter or digit, single line)"
        )
    if name != name.rstrip(". "):
        raise RegistryError(f"voice name {name!r} may not end with a dot or space")
    folded = name.casefold()
    # A Windows device name is reserved even with an extension (CON.wav, NUL.voice),
    # so validate the segment before the first dot.
    if folded.split(".", 1)[0] in _WINDOWS_DEVICE_NAMES:
        raise RegistryError(f"{name!r} is a reserved device name")
    if folded in {r.casefold() for r in _RESERVED}:
        raise RegistryError(f"{name!r} is a reserved name")
    if folded in {b.casefold() for b in BUNDLED_VOICE_NAMES}:
        raise RegistryError(f"{name!r} collides with a bundled voice; choose a different name")
    if folded in {n.casefold() for n in existing}:
        raise RegistryError(
            f"a voice named {name!r} already exists (names are case-insensitive); "
            "remove it first"
        )


def _resolve(path: str | Path) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except OSError as exc:
        raise RegistryError(f"cannot resolve path {str(path)!r}: {exc}") from exc


def _validate_weight(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"{label} must be a number") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RegistryError(f"{label} must be a finite number between 0.0 and 1.0")
    return number


def _safe_refs_dir() -> Path:
    directory = refs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise RegistryError(
            f"{directory} is a symlink; refusing to write clone references through it"
        )
    return directory


def _copy_into_refs(source: Path, destination: Path) -> None:
    """Copy *source* to *destination* safely: never destroy the source when it
    already is the destination, never truncate an unrelated file through a
    planted symlink, and never leave a half-written clip on failure.

    The copy is staged into a unique temp file (mode 0600 — a personal voice
    clip) and atomically moved into place.
    """
    # Re-registering a clip that already lives in voice-refs: source == dest.
    # Just ensure the private mode; unlinking here would delete the source.
    try:
        if destination.exists() and source.samefile(destination):
            os.chmod(destination, 0o600)
            return
    except OSError:
        pass
    directory = destination.parent
    handle_fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(directory))
    tmp = Path(tmp_name)
    try:
        os.close(handle_fd)
        shutil.copyfile(source, tmp)  # tmp is a fresh regular file, never a symlink
        os.chmod(tmp, 0o600)
        os.replace(tmp, destination)  # atomic; no destroy-before-copy
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RegistryError(f"cannot copy reference clip into {destination}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Mutations
# --------------------------------------------------------------------------- #
def _add_piper_unlocked(
    name: str,
    model_path: str | Path,
    config_path: str | Path | None = None,
    *,
    language: str = "",
    license_name: str = "user-supplied",
    provenance: str = "user-supplied Piper voice",
) -> RegisteredVoice:
    """Register an existing Piper voice by path. The ``.onnx`` is referenced in
    place (not copied); its ``.onnx.json`` config must sit alongside it unless
    given explicitly."""
    data = _load_for_write()
    _validate_new_name(name, data["voices"])
    model = _resolve(model_path)
    if model.suffix != ".onnx" or not model.is_file():
        raise RegistryError(f"Piper model not found or not an .onnx file: {model}")
    config = _resolve(config_path) if config_path is not None else model.with_name(model.name + ".json")
    if not config.is_file():
        raise RegistryError(
            f"Piper voice config not found: {config} "
            "(expected the .onnx.json beside the model, or pass it explicitly)"
        )
    record = {
        "engine": "piper",
        "params": {"model": str(model), "config": str(config)},
        "language": language,
        "license": license_name,
        "provenance": provenance,
    }
    data["voices"][name] = record
    _save(data)
    return _to_voice(name, record)


def _add_clone_unlocked(
    name: str,
    reference_path: str | Path,
    *,
    language: str = "en",
    license_name: str = "personal / non-distributed",
    provenance: str = "voice clone (timbre only)",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> RegisteredVoice:
    """Register a cloned voice from a reference clip. The clip is copied into
    ``voice-refs/`` (stored by basename) so the voice survives the original
    being moved."""
    data = _load_for_write()
    _validate_new_name(name, data["voices"])
    exaggeration = _validate_weight(exaggeration, "exaggeration")
    cfg_weight = _validate_weight(cfg_weight, "cfg_weight")
    reference = _resolve(reference_path)
    if not reference.is_file():
        raise RegistryError(f"reference clip not found: {reference}")
    suffix = reference.suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        raise RegistryError(
            f"reference must be an audio file ({', '.join(sorted(_AUDIO_SUFFIXES))}); "
            f"got {suffix or 'no extension'}"
        )
    stored = _safe_refs_dir() / f"{name}{suffix}"
    _copy_into_refs(reference, stored)
    record = {
        "engine": "clone",
        "params": {
            "reference": stored.name,  # basename; runtime resolves under refs_dir()
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        },
        "language": language,
        "license": license_name,
        "provenance": provenance,
    }
    try:
        data["voices"][name] = record
        _save(data)
    except Exception:
        # Roll back the copied clip so a failed save never orphans a personal clip.
        try:
            if stored.is_file() and not stored.is_symlink():
                stored.unlink()
        except OSError:
            pass
        raise
    return _to_voice(name, record)


def _add_f5_unlocked(
    name: str,
    reference_path: str | Path,
    ref_text: str,
    *,
    language: str = "en",
    license_name: str = "personal / non-distributed",
    provenance: str = "F5 voice clone (timbre only)",
) -> RegisteredVoice:
    """Register an F5-TTS voice and its transcribed reference clip."""
    data = _load_for_write()
    _validate_new_name(name, data["voices"])
    if not isinstance(ref_text, str) or not ref_text.strip():
        raise RegistryError("F5 reference text must be a non-empty string")
    reference = _resolve(reference_path)
    if not reference.is_file():
        raise RegistryError(f"reference clip not found: {reference}")
    suffix = reference.suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        raise RegistryError(
            f"reference must be an audio file ({', '.join(sorted(_AUDIO_SUFFIXES))}); "
            f"got {suffix or 'no extension'}"
        )
    stored = _safe_refs_dir() / f"{name}{suffix}"
    _copy_into_refs(reference, stored)
    record = {
        "engine": "f5",
        "params": {
            "reference": stored.name,
            "ref_text": ref_text,
        },
        "language": language,
        "license": license_name,
        "provenance": provenance,
    }
    try:
        data["voices"][name] = record
        _save(data)
    except Exception:
        try:
            if stored.is_file() and not stored.is_symlink():
                stored.unlink()
        except OSError:
            pass
        raise
    return _to_voice(name, record)


def _remove_unlocked(name: str) -> None:
    """Remove a registered voice. The registry entry is committed first; the
    copied clone clip (if any) is then deleted, and a cleanup failure is
    reported so an orphaned personal clip cannot go unnoticed."""
    data = _load_for_write()
    record = data["voices"].get(name)
    if not isinstance(record, dict):
        raise RegistryError(f"no registered voice named {name!r}")
    reference_basename = None
    if record.get("engine") in {"clone", "f5"} and isinstance(record.get("params"), dict):
        reference = record["params"].get("reference")
        if isinstance(reference, str):
            reference_basename = Path(reference).name  # defend against legacy absolute values
    del data["voices"][name]
    _save(data)  # commit the registry change before touching the filesystem
    if reference_basename:
        directory = refs_dir()
        if directory.is_symlink():
            raise RegistryError(
                f"voice {name!r} was removed, but {directory} is a symlink; refusing to "
                "delete through it — remove the clone clip by hand"
            )
        stored = directory / reference_basename
        try:
            if stored.is_symlink() or not stored.is_file():
                return  # not a regular clip we own; leave it alone
            stored.unlink()
        except OSError as exc:
            raise RegistryError(
                f"voice {name!r} was removed, but its reference clip {stored} could not "
                f"be deleted ({exc}); please remove it by hand"
            ) from exc


def add_piper(
    name: str,
    model_path: str | Path,
    config_path: str | Path | None = None,
    *,
    language: str = "",
    license_name: str = "user-supplied",
    provenance: str = "user-supplied Piper voice",
) -> RegisteredVoice:
    return _under_registry_lock(
        lambda: _add_piper_unlocked(
            name,
            model_path,
            config_path,
            language=language,
            license_name=license_name,
            provenance=provenance,
        )
    )


def add_clone(
    name: str,
    reference_path: str | Path,
    *,
    language: str = "en",
    license_name: str = "personal / non-distributed",
    provenance: str = "voice clone (timbre only)",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> RegisteredVoice:
    return _under_registry_lock(
        lambda: _add_clone_unlocked(
            name,
            reference_path,
            language=language,
            license_name=license_name,
            provenance=provenance,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    )


def add_f5(
    name: str,
    reference_path: str | Path,
    ref_text: str,
    *,
    language: str = "en",
    license_name: str = "personal / non-distributed",
    provenance: str = "F5 voice clone (timbre only)",
) -> RegisteredVoice:
    return _under_registry_lock(
        lambda: _add_f5_unlocked(
            name,
            reference_path,
            ref_text,
            language=language,
            license_name=license_name,
            provenance=provenance,
        )
    )


def remove(name: str) -> None:
    _under_registry_lock(lambda: _remove_unlocked(name))
