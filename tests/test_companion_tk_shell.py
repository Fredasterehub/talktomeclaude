from __future__ import annotations

import unittest
from typing import Any

from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.tk_shell import (
    TkCompanionShell,
    WindowsNonActivatingPolicy,
)
from talktomeclaude.core import RuntimePhase, RuntimeState


class _Variable:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _Widget:
    def __init__(self, _parent: object = None, **options: Any) -> None:
        self.options = options
        self.grid_options: dict[str, object] = {}

    def grid(self, **options: object) -> None:
        self.grid_options = dict(options)

    def configure(self, **options: object) -> None:
        self.options.update(options)

    def invoke(self) -> None:
        command = self.options.get("command")
        if callable(command):
            command()


class _Root(_Widget):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, object]] = []
        self.protocols: dict[str, object] = {}
        self.after_callbacks: dict[str, object] = {}
        self.destroyed = False
        self.focus_calls = 0

    def withdraw(self) -> None:
        self.calls.append(("withdraw", None))

    def title(self, value: str) -> None:
        self.calls.append(("title", value))

    def geometry(self, value: str) -> None:
        self.calls.append(("geometry", value))

    def resizable(self, width: bool, height: bool) -> None:
        self.calls.append(("resizable", (width, height)))

    def protocol(self, name: str, callback: object) -> None:
        self.protocols[name] = callback

    def update_idletasks(self) -> None:
        self.calls.append(("update", None))

    def winfo_id(self) -> int:
        return 41

    def after(self, _milliseconds: int, callback: object) -> str:
        handle = f"after-{len(self.after_callbacks) + 1}"
        self.after_callbacks[handle] = callback
        return handle

    def after_cancel(self, handle: object) -> None:
        self.calls.append(("after_cancel", handle))
        self.after_callbacks.pop(str(handle), None)

    def destroy(self) -> None:
        self.destroyed = True

    def mainloop(self) -> None:
        self.calls.append(("mainloop", None))

    def focus_force(self) -> None:
        self.focus_calls += 1

    def lift(self) -> None:
        self.focus_calls += 1


class _Tk:
    NORMAL = "normal"
    DISABLED = "disabled"
    Frame = _Widget
    Label = _Widget
    Button = _Widget
    StringVar = _Variable

    def __init__(self) -> None:
        self.root = _Root()

    def Tk(self) -> _Root:
        return self.root


class _Policy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def apply(self, widget_handle: int) -> int:
        self.calls.append(("apply", widget_handle))
        return 99

    def show_without_activation(self, window_handle: int) -> None:
        self.calls.append(("show", window_handle))

    def release(self, window_handle: int) -> None:
        self.calls.append(("release", window_handle))


class TkCompanionShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tk = _Tk()
        self.policy = _Policy()
        self.intents: list[CompanionIntent] = []
        self.muted = False

        def dispatch(intent: CompanionIntent) -> CompanionSnapshot:
            self.intents.append(intent)
            if intent.kind is IntentKind.START_RECORDING:
                return CompanionSnapshot(RuntimeState(RuntimePhase.RECORDING))
            if intent.kind is IntentKind.FINISH_RECORDING:
                return CompanionSnapshot(RuntimeState(RuntimePhase.TRANSCRIBING))
            if intent.kind is IntentKind.TOGGLE_OUTPUT_MUTE:
                self.muted = not self.muted
            return CompanionSnapshot(RuntimeState(), output_muted=self.muted)

        self.shell = TkCompanionShell(
            dispatch,
            CompanionSnapshot(RuntimeState(), "ready detail"),
            tk_module=self.tk,
            window_policy=self.policy,
            poll_milliseconds=10,
        )

    def tearDown(self) -> None:
        self.shell.close()

    def test_launch_is_compact_semantic_and_native_nonactivating(self) -> None:
        self.assertIn(("geometry", "360x190"), self.tk.root.calls)
        self.assertIn(("resizable", (False, False)), self.tk.root.calls)
        self.assertEqual(self.policy.calls[:2], [("apply", 41), ("show", 99)])
        self.assertEqual(self.shell._cue.get(), "[IDLE]")
        self.assertEqual(self.shell._status.get(), "Companion ready")
        self.assertEqual(self.shell._detail.get(), "ready detail")
        self.assertEqual(self.tk.root.focus_calls, 0)

    def test_workflow_mute_and_surface_controls_dispatch_focus_permissions(self) -> None:
        self.shell.buttons[IntentKind.START_RECORDING].invoke()
        self.assertEqual(self.intents[-1], CompanionIntent(IntentKind.START_RECORDING))
        self.assertEqual(
            self.shell.buttons[IntentKind.FINISH_RECORDING].options["state"],
            self.tk.NORMAL,
        )
        self.shell.buttons[IntentKind.FINISH_RECORDING].invoke()
        self.shell.buttons[IntentKind.TOGGLE_OUTPUT_MUTE].invoke()
        self.assertEqual(self.shell._mute_text.get(), "Unmute output")

        for kind in (
            IntentKind.OPEN_SETTINGS,
            IntentKind.OPEN_VOICE,
            IntentKind.OPEN_REVIEW,
            IntentKind.OPEN_DIAGNOSTICS,
        ):
            self.shell.buttons[kind].invoke()
            self.assertEqual(self.intents[-1], CompanionIntent(kind, allow_focus=True))
        self.assertTrue(
            all(
                not intent.allow_focus
                for intent in self.intents
                if intent.kind
                not in {
                    IntentKind.OPEN_SETTINGS,
                    IntentKind.OPEN_VOICE,
                    IntentKind.OPEN_REVIEW,
                    IntentKind.OPEN_DIAGNOSTICS,
                }
            )
        )

    def test_unsolicited_updates_change_text_and_controls_without_focus(self) -> None:
        self.shell.publish(
            CompanionSnapshot(
                RuntimeState(RuntimePhase.RECORDING),
                "microphone active",
            )
        )

        self.shell._drain_updates()

        self.assertEqual(self.shell._cue.get(), "[RECORDING]")
        self.assertEqual(self.shell._status.get(), "Recording")
        self.assertEqual(self.shell._detail.get(), "microphone active")
        self.assertEqual(self.tk.root.focus_calls, 0)
        self.assertEqual(self.policy.calls.count(("show", 99)), 1)

    def test_quit_returns_to_application_owned_shutdown_without_dispatch(self) -> None:
        self.shell.buttons[IntentKind.QUIT].invoke()
        self.shell.close()

        self.assertNotIn(IntentKind.QUIT, [intent.kind for intent in self.intents])
        self.assertTrue(self.tk.root.destroyed)
        self.assertTrue(any(call[0] == "after_cancel" for call in self.tk.root.calls))
        self.assertEqual(self.policy.calls[-1], ("release", 99))

    def test_workflow_intent_cannot_be_constructed_with_focus_permission(self) -> None:
        with self.assertRaises(ValueError):
            CompanionIntent(IntentKind.START_RECORDING, allow_focus=True)


class WindowsPolicyTests(unittest.TestCase):
    def test_policy_sets_noactivate_toolwindow_and_shows_without_activation(self) -> None:
        class Api:
            def __init__(self) -> None:
                self.style: tuple[int, int] | None = None
                self.calls: list[tuple[str, int]] = []

            def top_level(self, widget_handle: int) -> int:
                self.calls.append(("top", widget_handle))
                return 73

            def extended_style(self, window_handle: int) -> int:
                self.calls.append(("style", window_handle))
                return 0x20

            def set_extended_style(self, window_handle: int, style: int) -> None:
                self.style = (window_handle, style)

            def set_topmost_without_activation(self, window_handle: int) -> None:
                self.calls.append(("topmost", window_handle))

            def show_without_activation(self, window_handle: int) -> None:
                self.calls.append(("show", window_handle))

        api = Api()
        policy = WindowsNonActivatingPolicy(api)

        handle = policy.apply(41)
        policy.show_without_activation(handle)

        self.assertEqual(handle, 73)
        assert api.style is not None
        self.assertTrue(api.style[1] & policy.WS_EX_NOACTIVATE)
        self.assertTrue(api.style[1] & policy.WS_EX_TOOLWINDOW)
        self.assertEqual(api.calls[-2:], [("topmost", 73), ("show", 73)])


if __name__ == "__main__":
    unittest.main()
