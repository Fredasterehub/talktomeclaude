"""run_listen records the live session id for a voice-fired command through
an on_session callback."""

from __future__ import annotations

import inspect
import os
import unittest
from unittest import mock

from talktomeclaude import listen


class OnSessionCallbackTests(unittest.TestCase):
    def test_signature_accepts_on_session(self) -> None:
        self.assertIn("on_session", inspect.signature(listen.run_listen).parameters)

    def test_on_session_is_invoked_with_the_established_session_id(self) -> None:
        sessions: list[str] = []

        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "_record_always_on", return_value=object()
        ), mock.patch.object(
            listen, "UtteranceTranscriber"
        ) as transcriber_cls, mock.patch.object(
            listen, "_prompt_claude", return_value=("reply", "session-99")
        ):
            transcriber_cls.return_value.transcribe.return_value = "hello"

            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=True,
                echo=lambda _message: None,
                speak=lambda _message: None,
                status=lambda _message: None,
                on_session=sessions.append,
            )

        self.assertEqual(sessions, ["session-99"])


if __name__ == "__main__":
    unittest.main()
