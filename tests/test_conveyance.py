"""Tests for voice conveyance: chunking a long answer into speakable pieces,
the deterministic feedback-verb detector, and the per-session scratchpad."""

from __future__ import annotations

import os
import tempfile
import unittest

from talktomeclaude import conveyance
from talktomeclaude.listen import CONVEYANCE_VERBS, detect_verb


class ChunkTests(unittest.TestCase):
    def test_splits_long_text_into_bounded_chunks(self) -> None:
        text = " ".join(f"This is sentence number {i} here." for i in range(60))
        chunks = conveyance.chunk(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= conveyance.MAX_CHUNK_WORDS for c in chunks))

    def test_short_text_is_a_single_chunk(self) -> None:
        chunks = conveyance.chunk("One short sentence.")
        self.assertEqual(chunks, ["One short sentence."])

    def test_never_splits_mid_sentence(self) -> None:
        text = "First sentence here now. Second sentence follows along."
        chunks = conveyance.chunk(text)
        rejoined = " ".join(chunks)
        self.assertEqual(rejoined, text)

    def test_a_single_oversized_sentence_still_bounds(self) -> None:
        text = " ".join("word" for _ in range(200)) + "."
        chunks = conveyance.chunk(text)
        self.assertTrue(all(len(c.split()) <= conveyance.MAX_CHUNK_WORDS for c in chunks))


class DetectVerbTests(unittest.TestCase):
    def test_detects_every_closed_verb(self) -> None:
        for verb in CONVEYANCE_VERBS:
            self.assertEqual(detect_verb(verb), verb)

    def test_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(detect_verb("  Stop  "), "stop")
        self.assertEqual(detect_verb("REPEAT"), "repeat")

    def test_returns_none_for_content_utterance(self) -> None:
        self.assertIsNone(detect_verb("what is the capital of france"))


class ScratchpadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self._cwd)

    def test_path_is_homed_under_omc_state_sessions(self) -> None:
        path = str(conveyance.scratchpad_path("SID1")).replace("\\", "/")
        self.assertTrue(path.endswith(".omc/state/sessions/SID1/voice-conveyance.json"))

    def test_round_trips_exactly_the_locked_keys(self) -> None:
        conveyance.write_scratchpad("SID1", cursor=2, heading="Intro", status="delivering")
        data = conveyance.read_scratchpad("SID1")
        self.assertEqual(set(data.keys()), set(conveyance.SCRATCHPAD_KEYS))
        self.assertEqual(data["cursor"], 2)
        self.assertEqual(data["heading"], "Intro")
        self.assertEqual(data["status"], "delivering")

    def test_missing_scratchpad_reads_back_empty(self) -> None:
        self.assertEqual(conveyance.read_scratchpad("NEVER-WRITTEN"), {})


if __name__ == "__main__":
    unittest.main()
