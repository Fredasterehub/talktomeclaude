from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from talktomeclaude.speech.subprocess_worker import serve


class _Connection:
    def __init__(self, messages: list[object]) -> None:
        self.messages = messages
        self.sent: list[object] = []

    def recv(self) -> object:
        if not self.messages:
            raise EOFError
        return self.messages.pop(0)

    def send(self, value: object) -> None:
        self.sent.append(value)

    def close(self) -> None:
        pass


class SynthesisSubprocessServerTests(unittest.TestCase):
    def test_success_returns_only_identity_and_artifact_path(self) -> None:
        connection = _Connection(
            [
                {
                    "kind": "synthesize",
                    "job_id": "job-1",
                    "text": "Unicode café — 漢字 🚀",
                },
                {"kind": "stop"},
            ]
        )

        def synthesize(text: str, path: Path, voice: str) -> None:
            path.write_bytes(f"{voice}|{text}".encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            serve(connection, "rick", directory, synthesize_fn=synthesize)
            artifact = Path(directory) / "job-1.wav"
            self.assertEqual(
                artifact.read_bytes(), "rick|Unicode café — 漢字 🚀".encode()
            )

        self.assertEqual(connection.sent[0], {"kind": "ready"})
        reply = connection.sent[1]
        self.assertIsInstance(reply, dict)
        assert isinstance(reply, dict)
        self.assertTrue(reply["succeeded"])
        self.assertNotIn("text", reply)

    def test_failure_removes_partial_artifacts_and_returns_no_content(self) -> None:
        connection = _Connection(
            [
                {
                    "kind": "synthesize",
                    "job_id": "job-failed",
                    "text": "SECRET answer",
                },
                {"kind": "stop"},
            ]
        )

        def fail(_text: str, path: Path, _voice: str) -> None:
            path.write_bytes(b"partial")
            raise RuntimeError("SECRET answer")

        with tempfile.TemporaryDirectory() as directory:
            serve(connection, "rick", directory, synthesize_fn=fail)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

        reply = connection.sent[1]
        self.assertIsInstance(reply, dict)
        assert isinstance(reply, dict)
        self.assertFalse(reply["succeeded"])
        self.assertNotIn("SECRET", repr(reply))


if __name__ == "__main__":
    unittest.main()
