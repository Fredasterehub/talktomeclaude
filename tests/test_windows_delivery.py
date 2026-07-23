from __future__ import annotations

import ctypes
import inspect
import os
import unittest
from collections import deque

from talktomeclaude.platform import (
    DeliveryCode,
    DeliveryMode,
    RestoreStatus,
)
from talktomeclaude.platform.windows.capabilities import (
    resolve_terminal_capability,
)
from talktomeclaude.platform.windows.clipboard import (
    ClipboardCode,
    ClipboardOperation,
    ClipboardSnapshot,
    ClipboardTransaction,
)
from talktomeclaude.platform.windows.hotkeys import (
    MOD_CONTROL,
    MOD_NOREPEAT,
    WM_HOTKEY,
    GlobalHotkeyAdapter,
)
from talktomeclaude.platform.windows.injector import (
    CtypesKeyboardFacade,
    KeySendCode,
    KeySendOutcome,
    TextInjector,
)
from talktomeclaude.platform.windows.target import (
    TargetCode,
    TargetEvidence,
    TargetResolution,
    TargetValidation,
    WindowsTargetResolver,
)


class _TargetFacade:
    def __init__(self) -> None:
        self.hwnd = 100
        self.pid = 200
        self.process = "WindowsTerminal.exe"
        self.window_class = "CASCADIA_HOSTING_WINDOW_CLASS"
        self.exists = True

    def get_foreground_window(self) -> int:
        return self.hwnd

    def get_window_process_id(self, _hwnd: int) -> int:
        return self.pid

    def get_process_name(self, _pid: int) -> str:
        return self.process

    def get_window_class(self, _hwnd: int) -> str:
        return self.window_class

    def is_window(self, _hwnd: int) -> bool:
        return self.exists


class TerminalCapabilityTests(unittest.TestCase):
    def test_supported_process_and_class_pairs_are_case_insensitive(self) -> None:
        capability = resolve_terminal_capability(
            "WINDOWSTERMINAL.EXE", "cascadia_hosting_window_class"
        )
        self.assertIsNotNone(capability)
        self.assertEqual(capability.kind, "windows_terminal")

    def test_process_or_class_alone_is_not_eligibility(self) -> None:
        self.assertIsNone(
            resolve_terminal_capability("WindowsTerminal.exe", "NotATerminal")
        )
        self.assertIsNone(
            resolve_terminal_capability("notepad.exe", "CASCADIA_HOSTING_WINDOW_CLASS")
        )


class TargetResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.facade = _TargetFacade()
        self.resolver = WindowsTargetResolver(self.facade)

    def test_snapshot_captures_ephemeral_supported_evidence(self) -> None:
        result = self.resolver.snapshot()
        self.assertEqual(result.code, TargetCode.VALID)
        self.assertEqual(
            result.evidence,
            TargetEvidence(
                100,
                200,
                "WindowsTerminal.exe",
                "CASCADIA_HOSTING_WINDOW_CLASS",
                "windows_terminal",
            ),
        )
        self.assertNotIn("100", repr(result.evidence))
        self.assertNotIn("200", repr(result.evidence))
        self.assertNotIn("WindowsTerminal", repr(result.evidence))

    def test_snapshot_fails_closed_for_each_unavailable_boundary(self) -> None:
        cases = (
            ("hwnd", 0, TargetCode.NO_FOREGROUND_WINDOW),
            ("pid", 0, TargetCode.PID_UNAVAILABLE),
            ("process", "", TargetCode.PROCESS_UNAVAILABLE),
            ("window_class", "", TargetCode.CLASS_UNAVAILABLE),
            ("process", "notepad.exe", TargetCode.UNSUPPORTED),
        )
        for attribute, value, expected in cases:
            with self.subTest(expected=expected):
                facade = _TargetFacade()
                setattr(facade, attribute, value)
                result = WindowsTargetResolver(facade).snapshot()
                self.assertIsNone(result.evidence)
                self.assertEqual(result.code, expected)

    def test_process_query_denial_is_fail_closed(self) -> None:
        class _Denied(_TargetFacade):
            def get_process_name(self, _pid: int) -> str:
                raise PermissionError("query denied")

        result = WindowsTargetResolver(_Denied()).snapshot()
        self.assertIsNone(result.evidence)
        self.assertEqual(result.code, TargetCode.PROCESS_UNAVAILABLE)

    def test_validation_requires_same_live_foreground_evidence(self) -> None:
        evidence = self.resolver.snapshot().evidence
        assert evidence is not None
        self.assertTrue(self.resolver.validate(evidence).valid)

        mutations = (
            ("exists", False, TargetCode.WINDOW_GONE),
            ("hwnd", 101, TargetCode.FOREGROUND_CHANGED),
            ("pid", 201, TargetCode.EVIDENCE_CHANGED),
            ("process", "conhost.exe", TargetCode.EVIDENCE_CHANGED),
            ("window_class", "ConsoleWindowClass", TargetCode.EVIDENCE_CHANGED),
        )
        for attribute, value, expected in mutations:
            with self.subTest(expected=expected):
                facade = _TargetFacade()
                setattr(facade, attribute, value)
                validation = WindowsTargetResolver(facade).validate(evidence)
                self.assertFalse(validation.valid)
                self.assertEqual(validation.code, expected)


class _Resolver:
    def __init__(self, validations: list[TargetValidation], events: list[str]) -> None:
        self.validations = deque(validations)
        self.events = events
        self.snapshot_calls = 0
        self.seen_evidence: list[TargetEvidence] = []

    def snapshot(self) -> TargetResolution:
        self.snapshot_calls += 1
        self.events.append("target.snapshot")
        return TargetResolution(EVIDENCE, TargetCode.VALID)

    def validate(self, evidence: TargetEvidence) -> TargetValidation:
        self.events.append("target.validate")
        self.seen_evidence.append(evidence)
        return self.validations.popleft()


class _Transaction:
    def __init__(
        self,
        events: list[str],
        *,
        snapshot_code: ClipboardCode = ClipboardCode.OK,
        set_code: ClipboardCode = ClipboardCode.OK,
        restore: RestoreStatus = RestoreStatus.RESTORED,
    ) -> None:
        self.events = events
        self.snapshot_code = snapshot_code
        self.set_code = set_code
        self.restore_status = restore
        self.text: str | None = None

    def snapshot(self) -> ClipboardOperation:
        self.events.append("clipboard.snapshot")
        return ClipboardOperation(self.snapshot_code)

    def set_text(self, text: str) -> ClipboardOperation:
        self.events.append("clipboard.set")
        self.text = text
        return ClipboardOperation(self.set_code)

    def restore(self) -> RestoreStatus:
        self.events.append("clipboard.restore")
        return self.restore_status


class _Keyboard:
    def __init__(
        self,
        events: list[str],
        *,
        paste: KeySendCode = KeySendCode.SENT,
        enter: KeySendCode = KeySendCode.SENT,
    ) -> None:
        self.events = events
        self.paste = paste
        self.enter = enter

    def send_paste(self) -> KeySendOutcome:
        self.events.append("keyboard.paste")
        inserted = 4 if self.paste in {KeySendCode.SENT, KeySendCode.TIMEOUT} else 0
        return KeySendOutcome(self.paste, inserted, 4, 0.2 if self.paste is KeySendCode.TIMEOUT else 0.0)

    def send_enter(self) -> KeySendOutcome:
        self.events.append("keyboard.enter")
        inserted = 2 if self.enter in {KeySendCode.SENT, KeySendCode.TIMEOUT} else 0
        return KeySendOutcome(self.enter, inserted, 2, 0.2 if self.enter is KeySendCode.TIMEOUT else 0.0)


EVIDENCE = TargetEvidence(
    100,
    200,
    "WindowsTerminal.exe",
    "CASCADIA_HOSTING_WINDOW_CLASS",
    "windows_terminal",
)
VALID = TargetValidation(True, TargetCode.VALID)
CHANGED = TargetValidation(False, TargetCode.FOREGROUND_CHANGED)


class TextInjectorTests(unittest.TestCase):
    def make_injector(
        self,
        validations: list[TargetValidation],
        *,
        snapshot_code: ClipboardCode = ClipboardCode.OK,
        set_code: ClipboardCode = ClipboardCode.OK,
        restore: RestoreStatus = RestoreStatus.RESTORED,
        paste: KeySendCode = KeySendCode.SENT,
        enter: KeySendCode = KeySendCode.SENT,
    ) -> tuple[TextInjector, _Resolver, _Transaction, list[str]]:
        events: list[str] = []
        resolver = _Resolver(validations, events)
        transaction = _Transaction(
            events,
            snapshot_code=snapshot_code,
            set_code=set_code,
            restore=restore,
        )
        injector = TextInjector(
            resolver=resolver,  # type: ignore[arg-type]
            keyboard=_Keyboard(events, paste=paste, enter=enter),
            clipboard_factory=lambda: transaction,
        )
        return injector, resolver, transaction, events

    def test_empty_or_missing_target_has_zero_side_effects(self) -> None:
        injector, _, _, events = self.make_injector([])
        self.assertEqual(
            injector.deliver(
                "  ", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
            ).code,
            DeliveryCode.EMPTY_TRANSCRIPT,
        )
        self.assertEqual(
            injector.deliver(
                "text", None, mode=DeliveryMode.ASSISTANT, auto_submit=True
            ).code,
            DeliveryCode.INVALID_TARGET,
        )
        self.assertEqual(events, [])

    def test_generic_and_assistant_off_paste_once_without_enter(self) -> None:
        for mode, auto_submit in (
            (DeliveryMode.GENERIC, True),
            (DeliveryMode.ASSISTANT, False),
        ):
            with self.subTest(mode=mode, auto_submit=auto_submit):
                injector, resolver, _, events = self.make_injector([VALID, VALID])
                result = injector.deliver(
                    "hello", EVIDENCE, mode=mode, auto_submit=auto_submit
                )
                self.assertEqual(result.code, DeliveryCode.DELIVERED)
                self.assertTrue(result.pasted)
                self.assertFalse(result.submitted)
                self.assertEqual(events.count("keyboard.paste"), 1)
                self.assertNotIn("keyboard.enter", events)
                self.assertEqual(resolver.snapshot_calls, 0)

    def test_assistant_on_validates_and_submits_exactly_once(self) -> None:
        injector, resolver, transaction, events = self.make_injector(
            [VALID, VALID, VALID]
        )
        text = "🙂 e\u0301\nمرحبا"
        result = injector.deliver(
            text, EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.DELIVERED)
        self.assertTrue(result.pasted)
        self.assertTrue(result.submitted)
        self.assertEqual(transaction.text, text)
        self.assertEqual(
            events,
            [
                "target.validate",
                "clipboard.snapshot",
                "clipboard.set",
                "target.validate",
                "keyboard.paste",
                "target.validate",
                "keyboard.enter",
                "clipboard.restore",
            ],
        )
        self.assertEqual(resolver.seen_evidence, [EVIDENCE, EVIDENCE, EVIDENCE])
        self.assertTrue(all(item is EVIDENCE for item in resolver.seen_evidence))
        self.assertEqual(resolver.snapshot_calls, 0)

    def test_finish_snapshot_is_one_resolution_then_delivery_never_retargets(self) -> None:
        injector, resolver, _, events = self.make_injector([VALID, VALID, VALID])
        snapshot = injector.snapshot_target()
        assert snapshot.evidence is not None
        injector.deliver(
            "hello",
            snapshot.evidence,
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )
        self.assertEqual(resolver.snapshot_calls, 1)
        self.assertEqual(events.count("target.snapshot"), 1)

    def test_target_change_pre_clipboard_does_nothing_else(self) -> None:
        injector, _, _, events = self.make_injector([CHANGED])
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.TARGET_CHANGED_PRE_CLIPBOARD)
        self.assertEqual(events, ["target.validate"])

    def test_clipboard_failures_have_distinct_codes_and_no_keys(self) -> None:
        cases = (
            (ClipboardCode.OPEN_TIMEOUT, ClipboardCode.OK, DeliveryCode.CLIPBOARD_OPEN_TIMEOUT),
            (
                ClipboardCode.UNSUPPORTED_FORMAT,
                ClipboardCode.OK,
                DeliveryCode.CLIPBOARD_UNSUPPORTED_FORMAT,
            ),
            (ClipboardCode.READ_FAILED, ClipboardCode.OK, DeliveryCode.CLIPBOARD_READ_FAILED),
            (ClipboardCode.OK, ClipboardCode.OPEN_TIMEOUT, DeliveryCode.CLIPBOARD_OPEN_TIMEOUT),
            (ClipboardCode.OK, ClipboardCode.SET_FAILED, DeliveryCode.CLIPBOARD_SET_FAILED),
        )
        for snapshot_code, set_code, expected in cases:
            with self.subTest(expected=expected):
                injector, _, _, events = self.make_injector(
                    [VALID], snapshot_code=snapshot_code, set_code=set_code
                )
                result = injector.deliver(
                    "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
                )
                self.assertEqual(result.code, expected)
                self.assertFalse(any(event.startswith("keyboard.") for event in events))
                if snapshot_code is ClipboardCode.UNSUPPORTED_FORMAT:
                    self.assertNotIn("clipboard.set", events)
                    self.assertNotIn("clipboard.restore", events)
                if snapshot_code is ClipboardCode.OK and set_code is not ClipboardCode.OK:
                    self.assertIn("clipboard.restore", events)

    def test_target_change_after_set_restores_with_zero_keys(self) -> None:
        injector, _, _, events = self.make_injector([VALID, CHANGED])
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.TARGET_CHANGED_PRE_PASTE)
        self.assertEqual(result.restore_status, RestoreStatus.RESTORED)
        self.assertNotIn("keyboard.paste", events)
        self.assertNotIn("keyboard.enter", events)
        self.assertEqual(events[-1], "clipboard.restore")

    def test_paste_failure_restores_and_sends_no_enter(self) -> None:
        injector, _, _, events = self.make_injector(
            [VALID, VALID], paste=KeySendCode.FAILED
        )
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.PASTE_FAILED)
        self.assertNotIn("keyboard.enter", events)
        self.assertEqual(events[-1], "clipboard.restore")

    def test_paste_timeout_is_distinct_restores_and_never_sends_enter(self) -> None:
        injector, _, _, events = self.make_injector(
            [VALID, VALID, VALID], paste=KeySendCode.TIMEOUT
        )
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.PASTE_TIMEOUT)
        self.assertTrue(result.pasted)
        self.assertFalse(result.submitted)
        self.assertEqual(events.count("keyboard.paste"), 1)
        self.assertNotIn("keyboard.enter", events)
        self.assertEqual(events[-1], "clipboard.restore")

    def test_target_change_after_paste_is_explicit_partial_without_enter(self) -> None:
        injector, _, _, events = self.make_injector([VALID, VALID, CHANGED])
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.PASTED_NOT_SUBMITTED)
        self.assertTrue(result.pasted)
        self.assertFalse(result.submitted)
        self.assertEqual(events.count("keyboard.paste"), 1)
        self.assertNotIn("keyboard.enter", events)

    def test_enter_failure_stops_and_restores(self) -> None:
        injector, _, _, events = self.make_injector(
            [VALID, VALID, VALID], enter=KeySendCode.FAILED
        )
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.ASSISTANT, auto_submit=True
        )
        self.assertEqual(result.code, DeliveryCode.ENTER_FAILED)
        self.assertTrue(result.pasted)
        self.assertFalse(result.submitted)
        self.assertEqual(events.count("keyboard.enter"), 1)
        self.assertEqual(events[-1], "clipboard.restore")

    def test_restore_conflict_is_visible_without_hiding_delivery(self) -> None:
        injector, _, _, _ = self.make_injector(
            [VALID, VALID], restore=RestoreStatus.CONFLICT
        )
        result = injector.deliver(
            "hello", EVIDENCE, mode=DeliveryMode.GENERIC, auto_submit=False
        )
        self.assertEqual(result.code, DeliveryCode.DELIVERED)
        self.assertEqual(result.restore_status, RestoreStatus.CONFLICT)


class _ClipboardFacade:
    def __init__(self, opens: list[bool] | None = None) -> None:
        self.opens = deque(opens or [True])
        self.text: str | None = "original"
        self.sequence = 10
        self.open_count = 0
        self.close_count = 0
        self.write_ok = True
        self.write_exception = False
        self.safe_formats = True

    def open(self) -> bool:
        self.open_count += 1
        return self.opens.popleft() if self.opens else True

    def close(self) -> None:
        self.close_count += 1

    def read_unicode(self) -> str | None:
        return self.text

    def is_unicode_only_or_empty(self) -> bool:
        return self.safe_formats

    def write_unicode(self, text: str) -> bool:
        if self.write_exception:
            self.text = None
            self.sequence += 1
            self.write_exception = False
            raise OSError("failure after EmptyClipboard")
        if not self.write_ok:
            return False
        self.text = text
        self.sequence += 1
        return True

    def clear(self) -> bool:
        if not self.write_ok:
            return False
        self.text = None
        self.sequence += 1
        return True

    def sequence_number(self) -> int:
        return self.sequence


class ClipboardTransactionTests(unittest.TestCase):
    def test_bounded_retry_snapshot_set_and_owned_restore(self) -> None:
        facade = _ClipboardFacade([False, False, True, True, True])
        ticks = iter([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        transaction = ClipboardTransaction(
            facade,
            open_timeout=0.25,
            retry_interval=0,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(ticks),
        )
        self.assertTrue(transaction.snapshot().succeeded)
        self.assertTrue(transaction.set_text("replacement").succeeded)
        self.assertEqual(facade.text, "replacement")
        self.assertEqual(transaction.restore(), RestoreStatus.RESTORED)
        self.assertEqual(facade.text, "original")
        self.assertEqual(facade.close_count, 3)

    def test_restore_detects_newer_clipboard_owner_without_opening(self) -> None:
        facade = _ClipboardFacade()
        transaction = ClipboardTransaction(facade)
        self.assertTrue(transaction.snapshot().succeeded)
        self.assertTrue(transaction.set_text("ours").succeeded)
        open_count = facade.open_count
        facade.sequence += 1
        facade.text = "someone else's"
        self.assertEqual(transaction.restore(), RestoreStatus.CONFLICT)
        self.assertEqual(facade.open_count, open_count)
        self.assertEqual(facade.text, "someone else's")

    def test_open_timeout_is_bounded_and_never_closed_without_ownership(self) -> None:
        facade = _ClipboardFacade([False, False, False])
        ticks = iter([0.0, 0.1, 0.3])
        transaction = ClipboardTransaction(
            facade,
            open_timeout=0.2,
            retry_interval=0,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(ticks),
        )
        self.assertEqual(transaction.snapshot().code, ClipboardCode.OPEN_TIMEOUT)
        self.assertEqual(facade.close_count, 0)

    def test_nontext_or_rich_clipboard_fails_closed_without_mutation(self) -> None:
        facade = _ClipboardFacade()
        facade.safe_formats = False
        transaction = ClipboardTransaction(facade)
        self.assertEqual(
            transaction.snapshot().code, ClipboardCode.UNSUPPORTED_FORMAT
        )
        self.assertEqual(facade.text, "original")
        with self.assertRaises(RuntimeError):
            transaction.set_text("replacement")

    def test_set_exception_after_mutation_is_restorable(self) -> None:
        facade = _ClipboardFacade()
        transaction = ClipboardTransaction(facade)
        self.assertTrue(transaction.snapshot().succeeded)
        facade.write_exception = True
        self.assertEqual(transaction.set_text("replacement").code, ClipboardCode.SET_FAILED)
        self.assertIsNone(facade.text)
        self.assertEqual(transaction.restore(), RestoreStatus.RESTORED)
        self.assertEqual(facade.text, "original")


class _HotkeyFacade:
    def __init__(self) -> None:
        self.register_calls: list[tuple[int, int, int, int]] = []
        self.unregister_calls: list[tuple[int, int]] = []
        self.register_ok = True
        self.unregister_ok = True

    def register_hotkey(self, hwnd: int, hotkey_id: int, modifiers: int, vk: int) -> bool:
        self.register_calls.append((hwnd, hotkey_id, modifiers, vk))
        return self.register_ok

    def unregister_hotkey(self, hwnd: int, hotkey_id: int) -> bool:
        self.unregister_calls.append((hwnd, hotkey_id))
        return self.unregister_ok


class _IntentQueue:
    def __init__(self) -> None:
        self.items: list[int] = []

    def put_nowait(self, hotkey_id: int) -> None:
        self.items.append(hotkey_id)


class HotkeyTests(unittest.TestCase):
    def test_registration_always_adds_no_repeat_and_close_releases(self) -> None:
        facade = _HotkeyFacade()
        adapter = GlobalHotkeyAdapter(facade, hwnd=123)
        adapter.register(7, MOD_CONTROL, 0x78)
        self.assertEqual(
            facade.register_calls,
            [(123, 7, MOD_CONTROL | MOD_NOREPEAT, 0x78)],
        )
        adapter.close()
        self.assertEqual(facade.unregister_calls, [(123, 7)])

    def test_duplicate_id_is_rejected_without_second_os_call(self) -> None:
        facade = _HotkeyFacade()
        adapter = GlobalHotkeyAdapter(facade)
        adapter.register(1, 0, 0x78)
        with self.assertRaises(ValueError):
            adapter.register(1, 0, 0x79)
        self.assertEqual(len(facade.register_calls), 1)
        adapter.close()

    def test_registration_failure_is_not_tracked_for_cleanup(self) -> None:
        facade = _HotkeyFacade()
        facade.register_ok = False
        adapter = GlobalHotkeyAdapter(facade)
        with self.assertRaises(OSError):
            adapter.register(1, 0, 0x78)
        adapter.close()
        self.assertEqual(facade.unregister_calls, [])

    def test_shell_owned_dispatch_only_queues_registered_hotkey_intent(self) -> None:
        facade = _HotkeyFacade()
        intent_queue = _IntentQueue()
        adapter = GlobalHotkeyAdapter(facade, intent_queue=intent_queue)
        adapter.register(9, MOD_CONTROL, 0x78)
        self.assertFalse(adapter.dispatch_message(0x000F, 9))
        self.assertFalse(adapter.dispatch_message(WM_HOTKEY, 10))
        self.assertTrue(adapter.dispatch_message(WM_HOTKEY, 9))
        self.assertEqual(intent_queue.items, [9])
        adapter.close()

    def test_close_failure_keeps_id_observable_and_retryable(self) -> None:
        facade = _HotkeyFacade()
        adapter = GlobalHotkeyAdapter(facade)
        adapter.register(3, MOD_CONTROL, 0x78)
        facade.unregister_ok = False
        with self.assertRaises(OSError):
            adapter.close()
        facade.unregister_ok = True
        adapter.close()
        self.assertEqual(facade.unregister_calls, [(0, 3), (0, 3)])

    def test_one_hundred_dispatches_queue_exactly_once_and_reregister_cleanly(self) -> None:
        class _StatefulHotkeyFacade(_HotkeyFacade):
            def __init__(self) -> None:
                super().__init__()
                self.active: set[tuple[int, int]] = set()

            def register_hotkey(
                self, hwnd: int, hotkey_id: int, modifiers: int, vk: int
            ) -> bool:
                super().register_hotkey(hwnd, hotkey_id, modifiers, vk)
                key = (hwnd, hotkey_id)
                if key in self.active:
                    return False
                self.active.add(key)
                return True

            def unregister_hotkey(self, hwnd: int, hotkey_id: int) -> bool:
                super().unregister_hotkey(hwnd, hotkey_id)
                key = (hwnd, hotkey_id)
                if key not in self.active:
                    return False
                self.active.remove(key)
                return True

        facade = _StatefulHotkeyFacade()
        intent_queue = _IntentQueue()
        adapter = GlobalHotkeyAdapter(facade, intent_queue=intent_queue)
        adapter.register(11, MOD_CONTROL, 0x78)
        for _ in range(100):
            self.assertTrue(adapter.dispatch_message(WM_HOTKEY, 11))
        self.assertEqual(intent_queue.items, [11] * 100)
        adapter.close()
        self.assertEqual(facade.active, set())

        replacement = GlobalHotkeyAdapter(facade, intent_queue=intent_queue)
        replacement.register(11, MOD_CONTROL, 0x78)
        replacement.close()
        self.assertEqual(facade.active, set())
        self.assertEqual(len(facade.register_calls), 2)
        self.assertEqual(len(facade.unregister_calls), 2)


class NativeLayoutTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Win32 ABI layout")
    def test_send_input_structure_matches_64_bit_windows_abi(self) -> None:
        expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(CtypesKeyboardFacade._INPUT), expected)

    def test_send_input_checks_exact_count_and_cleans_partial_modifiers(self) -> None:
        class _User32:
            def __init__(self, returns: list[int]) -> None:
                self.returns = deque(returns)
                self.counts: list[int] = []

            def SendInput(self, count: int, _array: object, _size: int) -> int:
                self.counts.append(count)
                return self.returns.popleft()

        facade = CtypesKeyboardFacade.__new__(CtypesKeyboardFacade)
        user32 = _User32([2, 2])
        facade._user32 = user32  # type: ignore[attr-defined]
        facade._paste_deadline = 0.1
        facade._monotonic = iter([0.0, 0.01, 0.02]).__next__
        self.assertEqual(facade.send_paste().code, KeySendCode.FAILED)
        self.assertEqual(user32.counts, [4, 2])

        facade = CtypesKeyboardFacade.__new__(CtypesKeyboardFacade)
        user32 = _User32([4])
        facade._user32 = user32  # type: ignore[attr-defined]
        facade._paste_deadline = 0.1
        facade._monotonic = iter([0.0, 0.01]).__next__
        self.assertEqual(facade.send_paste().code, KeySendCode.SENT)
        self.assertEqual(user32.counts, [4])

    def test_send_input_timeout_is_measured_synchronously(self) -> None:
        class _User32:
            def SendInput(self, count: int, _array: object, _size: int) -> int:
                return count

        facade = CtypesKeyboardFacade.__new__(CtypesKeyboardFacade)
        facade._user32 = _User32()  # type: ignore[attr-defined]
        facade._paste_deadline = 0.1
        facade._monotonic = iter([10.0, 10.2]).__next__
        outcome = facade.send_paste()
        self.assertEqual(outcome.code, KeySendCode.TIMEOUT)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.elapsed_seconds, 0.2)


class SafetySurfaceTests(unittest.TestCase):
    def test_windows_delivery_has_no_focus_or_terminal_inspection_api(self) -> None:
        from talktomeclaude.platform.windows import injector, target

        source = inspect.getsource(target) + inspect.getsource(injector)
        forbidden = (
            "SetForegroundWindow",
            "SetFocus",
            "GetWindowText",
            "WM_GETTEXT",
            "GetGUIThreadInfo",
            "UIAutomation",
        )
        for name in forbidden:
            with self.subTest(name=name):
                self.assertNotIn(name, source)

    def test_lower_platform_reprs_hide_content_and_native_identity(self) -> None:
        transcript_secret = "SYNTHETIC_TRANSCRIPT_SECRET"
        process_secret = "SYNTHETIC_PROCESS_SECRET.exe"
        class_secret = "SYNTHETIC_CLASS_SECRET"
        evidence = TargetEvidence(
            987654321,
            123456789,
            process_secret,
            class_secret,
            "windows_terminal",
        )
        objects = (
            ClipboardSnapshot(transcript_secret, 42),
            evidence,
            TargetResolution(evidence, TargetCode.VALID),
            TargetValidation(True, TargetCode.VALID),
            KeySendOutcome(KeySendCode.SENT, 4, 4, 0.01),
        )
        rendered = repr(objects)
        for forbidden in (
            transcript_secret,
            process_secret,
            class_secret,
            "987654321",
            "123456789",
        ):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
