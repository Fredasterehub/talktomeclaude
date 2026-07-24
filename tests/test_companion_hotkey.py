from __future__ import annotations

import queue
import threading
import unittest

from talktomeclaude.companion.hotkey import ThreadHotkeyListener
from talktomeclaude.platform.windows.hotkeys import MOD_NOREPEAT, WM_HOTKEY


class _Pump:
    def __init__(self) -> None:
        self.messages: queue.Queue[tuple[int, int] | None] = queue.Queue()
        self.prepared = False
        self.quit_thread: int | None = None

    def prepare(self) -> int:
        self.prepared = True
        return 77

    def next_message(self) -> tuple[int, int] | None:
        return self.messages.get(timeout=1)

    def post_quit(self, thread_id: int) -> bool:
        self.quit_thread = thread_id
        self.messages.put(None)
        return True


class _Facade:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int, int]] = []
        self.unregistered: list[tuple[int, int]] = []

    def register_hotkey(
        self, hwnd: int, hotkey_id: int, modifiers: int, vk: int
    ) -> bool:
        self.registered.append((hwnd, hotkey_id, modifiers, vk))
        return True

    def unregister_hotkey(self, hwnd: int, hotkey_id: int) -> bool:
        self.unregistered.append((hwnd, hotkey_id))
        return True


class _KeyState:
    def __init__(self, values: list[bool], *, fallback: bool = False) -> None:
        self.values = values
        self.fallback = fallback
        self.calls = 0

    def is_pressed(self, _virtual_key: int) -> bool:
        self.calls += 1
        if self.values:
            return self.values.pop(0)
        return self.fallback

class ThreadHotkeyListenerTests(unittest.TestCase):
    def test_registers_dispatches_off_pump_thread_and_unregisters(self) -> None:
        pump = _Pump()
        facade = _Facade()
        called = threading.Event()
        callback_threads: list[int] = []

        def callback() -> None:
            callback_threads.append(threading.get_ident())
            called.set()

        listener = ThreadHotkeyListener(
            callback,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
        )
        listener.start()
        pump.messages.put((WM_HOTKEY, 1))

        self.assertTrue(called.wait(1))
        self.assertTrue(listener.stop())
        self.assertEqual(pump.quit_thread, 77)
        self.assertEqual(facade.unregistered, [(0, 1)])
        self.assertTrue(facade.registered[0][2] & MOD_NOREPEAT)
        self.assertNotEqual(callback_threads[0], listener._owner.ident)

    def test_start_and_stop_are_idempotent(self) -> None:
        pump = _Pump()
        facade = _Facade()
        listener = ThreadHotkeyListener(
            lambda: None,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
        )
        listener.start()
        listener.start()
        self.assertTrue(listener.stop())
        self.assertTrue(listener.stop())
        self.assertEqual(len(facade.registered), 1)

    def test_registration_failure_is_reported(self) -> None:
        pump = _Pump()

        class Failing(_Facade):
            def register_hotkey(
                self, hwnd: int, hotkey_id: int, modifiers: int, vk: int
            ) -> bool:
                return False

        listener = ThreadHotkeyListener(
            lambda: None,
            pump_factory=lambda: pump,
            facade_factory=Failing,
        )
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            listener.start()

    def test_hold_activation_dispatches_release_after_key_up(self) -> None:
        pump = _Pump()
        facade = _Facade()
        key_state = _KeyState([True, True, False])
        pressed = threading.Event()
        released = threading.Event()

        def activate() -> bool:
            pressed.set()
            return True

        listener = ThreadHotkeyListener(
            activate,
            release_callback=released.set,
            release_poll_seconds=0.001,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
            key_state_factory=lambda: key_state,
        )
        listener.start()
        pump.messages.put((WM_HOTKEY, 1))

        self.assertTrue(pressed.wait(1))
        self.assertTrue(released.wait(1))
        self.assertGreaterEqual(key_state.calls, 3)
        self.assertTrue(listener.stop())

    def test_toggle_activation_never_polls_or_dispatches_release(self) -> None:
        pump = _Pump()
        facade = _Facade()
        released = threading.Event()
        key_state_created = False

        def key_state_factory() -> _KeyState:
            nonlocal key_state_created
            key_state_created = True
            return _KeyState([])

        listener = ThreadHotkeyListener(
            lambda: False,
            release_callback=released.set,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
            key_state_factory=key_state_factory,
        )
        listener.start()
        pump.messages.put((WM_HOTKEY, 1))
        self.assertTrue(listener.stop())

        self.assertFalse(released.is_set())
        self.assertFalse(key_state_created)

    def test_stop_while_hold_key_is_down_suppresses_late_release(self) -> None:
        pump = _Pump()
        facade = _Facade()
        pressed = threading.Event()
        released = threading.Event()
        listener = ThreadHotkeyListener(
            lambda: pressed.set() or True,
            release_callback=released.set,
            release_poll_seconds=0.001,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
            key_state_factory=lambda: _KeyState([], fallback=True),
        )
        listener.start()
        pump.messages.put((WM_HOTKEY, 1))

        self.assertTrue(pressed.wait(1))
        self.assertTrue(listener.stop())
        self.assertFalse(released.is_set())

    def test_saturated_hold_queue_still_stops_both_owned_threads(self) -> None:
        pump = _Pump()
        facade = _Facade()
        pressed = threading.Event()
        listener = ThreadHotkeyListener(
            lambda: pressed.set() or True,
            release_callback=lambda: None,
            release_poll_seconds=0.001,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
            key_state_factory=lambda: _KeyState([], fallback=True),
        )
        listener.start()
        for _ in range(32):
            pump.messages.put((WM_HOTKEY, 1))
        self.assertTrue(pressed.wait(1))

        self.assertTrue(listener.stop())
        self.assertFalse(listener._owner.is_alive())
        self.assertFalse(listener._dispatcher.is_alive())

    def test_blocked_user_callback_makes_shutdown_failure_explicit(self) -> None:
        pump = _Pump()
        facade = _Facade()
        entered = threading.Event()
        release = threading.Event()

        def callback() -> None:
            entered.set()
            release.wait(1)

        listener = ThreadHotkeyListener(
            callback,
            shutdown_deadline_seconds=0.02,
            pump_factory=lambda: pump,
            facade_factory=lambda: facade,
        )
        listener.start()
        pump.messages.put((WM_HOTKEY, 1))
        self.assertTrue(entered.wait(1))

        self.assertFalse(listener.stop())
        release.set()
        listener._dispatcher.join(1)
        self.assertFalse(listener._dispatcher.is_alive())


if __name__ == "__main__":
    unittest.main()
