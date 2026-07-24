"""Assistant-specific integration contracts and Claude Code adapter."""

from .claude import (
    AssistantAdapter,
    AssistantEventCode,
    AssistantEventResult,
    ClaudeCodeAdapter,
    ValidatedAssistantEvent,
)
from talktomeclaude.reply_protocol import canonical_reply_digest
from .hooks import (
    CLAUDE_STOP_HOOK_COMMAND,
    OWNED_HOOK_MARKER,
    ClaudeHookManager,
    HookInspection,
    HookStatus,
)
from .suppression import (
    DirectorEventGate,
    DirectorLaunchGuard,
    DirectorLease,
    ManagedDirectorProcess,
    SuppressionRegistry,
)

__all__ = [
    "AssistantAdapter",
    "AssistantEventCode",
    "AssistantEventResult",
    "CLAUDE_STOP_HOOK_COMMAND",
    "ClaudeCodeAdapter",
    "ClaudeHookManager",
    "DirectorEventGate",
    "DirectorLaunchGuard",
    "DirectorLease",
    "ManagedDirectorProcess",
    "HookInspection",
    "HookStatus",
    "OWNED_HOOK_MARKER",
    "SuppressionRegistry",
    "ValidatedAssistantEvent",
    "canonical_reply_digest",
]
