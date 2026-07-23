"""Tests for the barge-in gate and interruptible delivery (LAW: bargein-gate)."""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from talktomeclaude import listen
from talktomeclaude.listen import barge_in_active


class BargeInActiveTests(unittest.TestCase):
    def test_active_only_when_on_and_headphones_present(self) -> None:
        self.assertTrue(barge_in_active(True, True))

    def test_inactive_without_headphones_even_if_on(self) -> None:
        self.assertFalse(barge_in_active(True, False))

    def test_inactive_when_operator_has_not_opted_in(self) -> None:
        self.assertFalse(barge_in_active(False, True))

    def test_inactive_when_neither_condition_holds(self) -> None:
        self.assertFalse(barge_in_active(False, False))


class SpeakInterruptibleTests(unittest.TestCase):
    """The barge-in must return what the operator said, not just a flag."""

    def _fake_hardware(self):
        block = mock.Mock()
        block.copy.return_value = block
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.return_value = (block, False)
        sounddevice = mock.Mock()
        sounddevice.InputStream.return_value = stream
        return sounddevice

    def test_interruption_returns_the_captured_utterance(self) -> None:
        sounddevice = self._fake_hardware()
        release = threading.Event()
        sounddevice.stop = mock.Mock(side_effect=release.set)
        calls = {"n": 0}

        def rms(_block) -> float:
            calls["n"] += 1
            if calls["n"] <= 6:  # calibration floor
                return 0.0
            if calls["n"] <= 10:  # the operator talks over the playback
                return 0.5
            return 0.0  # trailing silence ends the capture

        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), \
                mock.patch.object(listen, "_rms", side_effect=rms), \
                mock.patch.object(listen, "_finish", side_effect=lambda c: c or None):
            captured = listen._speak_interruptible(
                lambda _text: release.wait(5), "a long reply"
            )

        sounddevice.stop.assert_called()  # playback halted the moment speech hit
        self.assertIsNotNone(captured)  # ...and the utterance came back with it
        self.assertGreater(len(captured), 0)

    def test_undisturbed_playback_returns_none(self) -> None:
        sounddevice = self._fake_hardware()

        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), \
                mock.patch.object(listen, "_rms", return_value=0.0), \
                mock.patch.object(listen, "_finish", side_effect=lambda c: c or None):
            captured = listen._speak_interruptible(lambda _text: None, "a short reply")

        self.assertIsNone(captured)
        sounddevice.stop.assert_not_called()

    def test_slow_synthesis_cannot_leak_delayed_playback(self) -> None:
        sounddevice = self._fake_hardware()
        synthesis_started = threading.Event()
        release_synthesis = threading.Event()
        barge_detected = threading.Event()
        playback_started = threading.Event()
        playback_stopped = threading.Event()
        calls = {"n": 0}

        def rms(_block) -> float:
            calls["n"] += 1
            if calls["n"] <= 6:
                return 0.0
            if calls["n"] <= 10:
                return 0.5
            return 0.0

        def stop() -> None:
            barge_detected.set()
            if playback_started.is_set():
                playback_stopped.set()

        def slow_speak(_text: str) -> None:
            synthesis_started.set()
            release_synthesis.wait(2)
            playback_started.set()
            playback_stopped.wait(2)

        sounddevice.stop.side_effect = stop

        def release_after_barge() -> None:
            synthesis_started.wait(2)
            barge_detected.wait(2)
            release_synthesis.set()

        releaser = threading.Thread(target=release_after_barge)
        releaser.start()
        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), \
                mock.patch.object(listen, "_rms", side_effect=rms), \
                mock.patch.object(listen, "_finish", side_effect=lambda c: c or None):
            captured = listen._speak_interruptible(slow_speak, "a cloned reply")
        releaser.join(2)

        self.assertIsNotNone(captured)
        self.assertTrue(playback_started.is_set())
        self.assertTrue(playback_stopped.is_set())


if __name__ == "__main__":
    unittest.main()
