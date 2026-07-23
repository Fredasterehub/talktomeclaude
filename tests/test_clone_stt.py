"""Timestamp-preserving STT coverage for clone segment selection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from talktomeclaude import stt


class TimestampedTranscriptionTests(unittest.TestCase):
    def test_preserves_segment_timestamps_and_text(self) -> None:
        class WhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **_kwargs):
                return (
                    [
                        SimpleNamespace(start=1.25, end=3.5, text=" hello "),
                        SimpleNamespace(start=4.0, end=7.75, text="Claude"),
                    ],
                    None,
                )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"RIFFaudio")
            module = SimpleNamespace(WhisperModel=WhisperModel)
            with mock.patch.dict(sys.modules, {"faster_whisper": module}), mock.patch.object(
                stt, "detect_tier", return_value=stt.CPU_TIER
            ):
                segments, tier = stt.transcribe_file_with_timestamps(audio)

        self.assertEqual(tier, stt.CPU_TIER)
        self.assertEqual(
            segments,
            [
                stt.TranscriptSegment(1.25, 3.5, "hello"),
                stt.TranscriptSegment(4.0, 7.75, "Claude"),
            ],
        )

    def test_timestamped_api_uses_gpu_to_cpu_fallback(self) -> None:
        statuses = []
        recovered = [stt.TranscriptSegment(0.0, 2.0, "recovered")]
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"RIFFaudio")
            with mock.patch.object(
                stt, "detect_tier", return_value=stt.GPU_TIER
            ), mock.patch.object(
                stt,
                "_run_tier_timestamped",
                side_effect=[RuntimeError("cuda failed"), recovered],
            ):
                segments, tier = stt.transcribe_file_with_timestamps(
                    audio, on_status=statuses.append
                )

        self.assertEqual(segments, recovered)
        self.assertEqual(tier, stt.CPU_TIER)
        self.assertTrue(any("falling back" in status for status in statuses))


if __name__ == "__main__":
    unittest.main()
