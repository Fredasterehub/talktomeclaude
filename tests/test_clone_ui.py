"""Textual tests for mandatory clone reference review."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Input, Static

from talktomeclaude import clone_ui


class SegmentSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            SimpleNamespace(start=float(start), end=float(start + 5), text=f"part {start}")
            for start in range(0, 30, 5)
        ]

    def test_candidates_are_local_segment_aligned_and_bounded(self) -> None:
        segments = [
            SimpleNamespace(start=float(start), end=float(start + 5), text=f"part {start}")
            for start in range(0, 500, 5)
        ]
        candidates = clone_ui.generate_segment_candidates(segments, 500.0)
        segment_bounds = {
            value
            for segment in segments
            for value in (segment.start, segment.end)
        }
        self.assertEqual(len(candidates), clone_ui._MAX_CANDIDATES)
        for candidate in candidates:
            start, end = candidate.bounds
            self.assertIn(start, segment_bounds)
            self.assertIn(end, segment_bounds)
            self.assertGreaterEqual(end - start, 10.0)
            self.assertLessEqual(end - start, 20.0)

    def test_model_cannot_supply_timestamps(self) -> None:
        candidates = clone_ui.generate_segment_candidates(self.segments, 30.0)
        payload = json.dumps(
            {
                "candidate_id": candidates[0].candidate_id,
                "reason": "clean speech",
                "start": 999,
                "end": 1000,
            }
        )
        with self.assertRaises(ValueError):
            clone_ui._parse_selection(payload, candidates)

    def test_fallback_transcript_uses_only_fully_contained_segments(self) -> None:
        transcript = clone_ui._transcript_for_bounds(self.segments, (3.0, 18.0))
        self.assertEqual(transcript, "part 5 part 10")

    def test_download_rejects_file_over_size_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def download(_command, **_kwargs):
                (root / "source.m4a").write_bytes(b"oversized")
                return subprocess.CompletedProcess([], 0, "", "")

            with mock.patch.object(
                clone_ui, "_MAX_DOWNLOAD_BYTES", 4
            ), mock.patch.object(
                clone_ui.subprocess, "run", side_effect=download
            ):
                with self.assertRaisesRegex(RuntimeError, "size cap"):
                    clone_ui.download_youtube_audio("https://youtu.be/example", root)

    def test_scoped_selection_call_is_isolated_and_bounded(self) -> None:
        candidates = clone_ui.generate_segment_candidates(self.segments, 30.0)
        completed = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {"candidate_id": candidates[0].candidate_id, "reason": "clean speech"}
            ),
            "",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"TTMC_UNSAFE_ENV": "operator-project"}, clear=False
        ), mock.patch.object(
            clone_ui.subprocess, "run", return_value=completed
        ) as run:
            selected, reason = clone_ui.select_segment_candidate(
                candidates, Path(directory)
            )
            model_cwd = Path(run.call_args.kwargs["cwd"])
            self.assertEqual(list(model_cwd.iterdir()), [])

        command = run.call_args.args[0]
        self.assertEqual(selected, candidates[0])
        self.assertEqual(reason, "clean speech")
        self.assertIn("-p", command)
        self.assertIn("--output-format", command)
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(command[command.index("--disallowedTools") + 1], "*")
        self.assertEqual(command[command.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(command[command.index("--setting-sources") + 1], "")
        self.assertLessEqual(len(command[2]), clone_ui._MAX_PROMPT_CHARS)
        self.assertEqual(run.call_args.kwargs["timeout"], clone_ui._SELECTION_TIMEOUT)
        self.assertNotIn("TTMC_UNSAFE_ENV", run.call_args.kwargs["env"])

    def test_selection_timeout_uses_fixed_cut_with_transcript(self) -> None:
        self._assert_fallback(
            subprocess.TimeoutExpired("claude", clone_ui._SELECTION_TIMEOUT),
            "selection timed out",
        )

    def test_successful_selection_returns_structured_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            source.write_bytes(b"audio")
            response = json.dumps(
                {
                    "type": "result",
                    "result": json.dumps(
                        {"candidate_id": "c1", "reason": "steady clean speech"}
                    ),
                }
            )
            completed = subprocess.CompletedProcess([], 0, response, "")
            with mock.patch.object(
                clone_ui, "probe_duration", return_value=30.0
            ), mock.patch.object(
                clone_ui,
                "bound_source_for_stt",
                return_value=root / "stt-source.wav",
            ), mock.patch(
                "talktomeclaude.stt.transcribe_file_with_timestamps",
                return_value=(self.segments, mock.sentinel.tier),
            ) as transcribe, mock.patch(
                "talktomeclaude.config.stt_device", return_value="cpu"
            ), mock.patch.object(
                clone_ui, "cut_segment", return_value=root / "segment.wav"
            ) as cut, mock.patch.object(
                clone_ui.subprocess, "run", return_value=completed
            ):
                selection = clone_ui.auto_select_segment(source, root)

        self.assertFalse(selection.fallback)
        self.assertEqual(selection.bounds, (0.0, 15.0))
        self.assertEqual(selection.reason, "steady clean speech")
        self.assertEqual(selection.transcript, "part 0 part 5 part 10")
        transcribe.assert_called_once_with(root / "stt-source.wav", device="cpu")
        cut.assert_called_once_with(
            source,
            root / "segment.wav",
            start=0.0,
            seconds=15.0,
        )

    def test_malformed_selection_json_uses_fixed_cut_with_transcript(self) -> None:
        self._assert_fallback(None, "Expecting value")

    def test_source_duration_cap_prevents_unbounded_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            source.write_bytes(b"audio")
            with mock.patch.object(
                clone_ui,
                "probe_duration",
                return_value=clone_ui._MAX_SOURCE_SECONDS + 1.0,
            ), mock.patch(
                "talktomeclaude.stt.transcribe_file_with_timestamps"
            ) as transcribe, mock.patch.object(
                clone_ui, "cut_segment", return_value=root / "segment.wav"
            ):
                selection = clone_ui.auto_select_segment(source, root)

        transcribe.assert_not_called()
        self.assertTrue(selection.fallback)
        self.assertIn("selection cap", selection.reason)

    def test_underreported_source_is_physically_bounded_before_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            bounded = root / "stt-source.wav"
            source.write_bytes(b"audio")
            with mock.patch.object(
                clone_ui, "probe_duration", return_value=30.0
            ), mock.patch.object(
                clone_ui, "bound_source_for_stt", return_value=bounded
            ) as bound, mock.patch(
                "talktomeclaude.stt.transcribe_file_with_timestamps",
                return_value=(self.segments, mock.sentinel.tier),
            ) as transcribe, mock.patch.object(
                clone_ui,
                "select_segment_candidate",
                return_value=(
                    clone_ui.SegmentCandidate("c1", (0.0, 15.0), "bounded"),
                    "clean",
                ),
            ), mock.patch.object(
                clone_ui, "cut_segment", return_value=root / "segment.wav"
            ):
                clone_ui.auto_select_segment(source, root)

        bound.assert_called_once_with(source, root)
        self.assertEqual(transcribe.call_args.args[0], bounded)

    def test_persisted_stt_device_is_used_for_agent_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            bounded = root / "stt-source.wav"
            source.write_bytes(b"audio")
            with mock.patch.object(
                clone_ui, "probe_duration", return_value=30.0
            ), mock.patch.object(
                clone_ui, "bound_source_for_stt", return_value=bounded
            ), mock.patch(
                "talktomeclaude.config.stt_device", return_value="cuda"
            ), mock.patch(
                "talktomeclaude.stt.transcribe_file_with_timestamps",
                return_value=(self.segments, mock.sentinel.tier),
            ) as transcribe, mock.patch.object(
                clone_ui,
                "select_segment_candidate",
                return_value=(
                    clone_ui.SegmentCandidate("c1", (0.0, 15.0), "bounded"),
                    "clean",
                ),
            ), mock.patch.object(
                clone_ui, "cut_segment", return_value=root / "segment.wav"
            ):
                clone_ui.auto_select_segment(source, root)

        transcribe.assert_called_once_with(bounded, device="cuda")

    def test_stt_source_bound_uses_ffmpeg_duration_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            source.write_bytes(b"audio")

            def complete(_command, **_kwargs):
                (root / "stt-source.wav").write_bytes(b"RIFF" + b"0" * 64)
                return subprocess.CompletedProcess([], 0, "", "")

            with mock.patch.object(
                clone_ui.subprocess, "run", side_effect=complete
            ) as run:
                result = clone_ui.bound_source_for_stt(source, root)

        command = run.call_args.args[0]
        self.assertEqual(result.name, "stt-source.wav")
        self.assertEqual(command[command.index("-t") + 1], str(clone_ui._MAX_SOURCE_SECONDS))
        self.assertEqual(run.call_args.kwargs["timeout"], clone_ui._BOUND_TIMEOUT)

    def _assert_fallback(self, failure, reason: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            source.write_bytes(b"audio")
            completed = subprocess.CompletedProcess([], 0, "not-json", "")
            run_kwargs = {"side_effect": failure} if failure else {"return_value": completed}
            with mock.patch.object(
                clone_ui, "probe_duration", return_value=30.0
            ), mock.patch.object(
                clone_ui,
                "bound_source_for_stt",
                return_value=root / "stt-source.wav",
            ), mock.patch(
                "talktomeclaude.stt.transcribe_file_with_timestamps",
                return_value=(self.segments, mock.sentinel.tier),
            ), mock.patch.object(
                clone_ui, "cut_segment", return_value=root / "segment.wav"
            ) as cut, mock.patch.object(
                clone_ui.subprocess, "run", **run_kwargs
            ):
                selection = clone_ui.auto_select_segment(source, root)

        self.assertTrue(selection.fallback)
        self.assertEqual(selection.bounds, (3.0, 18.0))
        self.assertIn(reason, selection.reason)
        self.assertEqual(selection.transcript, "part 5 part 10")
        cut.assert_called_once_with(
            source,
            root / "segment.wav",
            start=3.0,
            seconds=15.0,
        )


class CloneScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_screen_contract_requires_review(self) -> None:
        self.assertTrue(issubclass(clone_ui.CloneScreen, Screen))
        self.assertIs(clone_ui.review_required, True)

    async def test_clone_registration_requires_audition_then_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "agent-cut.wav"
            reference.write_bytes(b"RIFFreference")
            audition = mock.Mock()
            registered = mock.Mock()
            completed: dict[str, bool] = {}
            screen = clone_ui.CloneScreen(
                "rick",
                reference,
                audition=audition,
            )

            class _Host(App[None]):
                def compose(self) -> ComposeResult:
                    return
                    yield

                def on_mount(self) -> None:
                    self.push_screen(
                        screen,
                        lambda value: completed.setdefault("value", value),
                    )

            with mock.patch.object(
                clone_ui.registry,
                "add_clone",
                return_value=registered,
            ) as add_clone:
                async with _Host().run_test() as pilot:
                    await pilot.pause()
                    await pilot.press("c")
                    await pilot.pause()
                    add_clone.assert_not_called()

                    await pilot.press("a")
                    await pilot.pause()
                    audition.assert_called_once_with(reference)
                    add_clone.assert_not_called()

                    await pilot.press("c")
                    await pilot.pause()

            add_clone.assert_called_once_with("rick", reference)
            self.assertIs(screen.created_voice, registered)
            self.assertIs(completed["value"], True)

    async def test_source_flow_reaches_the_mandatory_review(self) -> None:
        from textual.widgets import Input, OptionList

        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "ref.wav"
            reference.write_bytes(b"RIFFreference")
            audition = mock.Mock()
            screen = clone_ui.CloneScreen(audition=audition)

            class _Host(App[None]):
                def compose(self) -> ComposeResult:
                    return
                    yield

                def on_mount(self) -> None:
                    self.push_screen(screen)

            with mock.patch.object(
                clone_ui.registry, "add_clone", return_value=mock.Mock()
            ) as add_clone:
                async with _Host().run_test() as pilot:
                    await pilot.pause()
                    screen.query_one("#clone-name", Input).value = "rick"
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertIsNotNone(screen.query_one("#clone-source", OptionList))
                    await pilot.press("enter")  # "Audio file on disk"
                    await pilot.pause()
                    screen.query_one("#clone-file", Input).value = str(reference)
                    await pilot.press("enter")
                    await pilot.pause()
                    # The review gate: confirm is refused before audition.
                    await pilot.press("c")
                    await pilot.pause()
                    add_clone.assert_not_called()
                    await pilot.press("a")
                    await pilot.press("c")
                    await pilot.pause()

            audition.assert_called_once_with(reference)
            add_clone.assert_called_once_with("rick", reference)

    async def test_f5_confirmation_passes_reference_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "agent-cut.wav"
            reference.write_bytes(b"RIFFreference")
            screen = clone_ui.CloneScreen(
                "rick_f5",
                reference,
                engine="f5",
                ref_text="Reference transcript.",
                audition=lambda _path: None,
            )

            class _Host(App[None]):
                def compose(self) -> ComposeResult:
                    return
                    yield

                def on_mount(self) -> None:
                    self.push_screen(screen)

            with mock.patch.object(
                clone_ui.registry,
                "add_f5",
                return_value=mock.Mock(),
            ) as add_f5:
                async with _Host().run_test() as pilot:
                    await pilot.pause()
                    await pilot.press("c")
                    await pilot.pause()
                    add_f5.assert_not_called()
                    await pilot.press("a")
                    await pilot.press("c")
                    await pilot.pause()

            add_f5.assert_called_once_with(
                "rick_f5",
                reference,
                "Reference transcript.",
            )

    async def test_acquired_transcript_reaches_f5_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "selected.wav"
            reference.write_bytes(b"RIFFreference")
            selection = clone_ui.SegmentSelection(
                reference,
                (10.0, 25.0),
                "clear, uninterrupted speech",
                "Selected segment transcript.",
                False,
            )
            screen = clone_ui.CloneScreen(
                "selected_f5",
                engine="f5",
                audition=lambda _path: None,
            )

            class _Host(App[None]):
                def compose(self) -> ComposeResult:
                    return
                    yield

                def on_mount(self) -> None:
                    self.push_screen(screen)

            with mock.patch.object(
                clone_ui.registry,
                "add_f5",
                return_value=mock.Mock(),
            ) as add_f5:
                async with _Host().run_test() as pilot:
                    await pilot.pause()
                    screen._acquired(selection, None)
                    await pilot.pause()
                    await pilot.press("a")
                    await pilot.press("c")
                    await pilot.pause()

            add_f5.assert_called_once_with(
                "selected_f5",
                reference,
                "Selected segment transcript.",
            )

    async def test_review_pane_labels_fallback_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "fallback.wav"
            reference.write_bytes(b"RIFFreference")
            screen = clone_ui.CloneScreen("fallback", reference)
            screen.selection = clone_ui.SegmentSelection(
                reference,
                (4.0, 19.0),
                "Fixed-cut fallback: malformed response",
                "Fallback transcript.",
                True,
            )

            class _Host(App[None]):
                def compose(self) -> ComposeResult:
                    return
                    yield

                def on_mount(self) -> None:
                    self.push_screen(screen)

            async with _Host().run_test() as pilot:
                await pilot.pause()
                provenance = screen.query_one(
                    "#clone-selection-provenance",
                    Static,
                )

            self.assertIn("Fallback selection", str(provenance.render()))

    async def test_review_pane_renders_bracketed_model_text_without_markup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "selected.wav"
            reference.write_bytes(b"RIFFreference")
            screen = clone_ui.CloneScreen("safe", reference)
            screen.selection = clone_ui.SegmentSelection(
                reference,
                (0.0, 15.0),
                "clean take [/bold]",
                "[Music] exact transcript",
                False,
            )

            class _Host(App[None]):
                def compose(self) -> ComposeResult:
                    return
                    yield

                def on_mount(self) -> None:
                    self.push_screen(screen)

            async with _Host().run_test() as pilot:
                await pilot.pause()
                provenance = screen.query_one("#clone-selection-provenance", Static)
                transcript = screen.query_one("#clone-selection-transcript", Static)

            self.assertIn("[/bold]", str(provenance.render()))
            self.assertIn("[Music]", str(transcript.render()))

    async def test_non_http_youtube_url_is_rejected_before_download(self) -> None:
        screen = clone_ui.CloneScreen("safe")

        class _Host(App[None]):
            def compose(self) -> ComposeResult:
                return
                yield

            def on_mount(self) -> None:
                self.push_screen(screen)

        async with _Host().run_test() as pilot:
            await pilot.pause()
            screen._show_step("youtube")
            await pilot.pause()
            screen.query_one("#clone-url", Input).value = "--config-location=/tmp/evil"
            with mock.patch.object(screen, "run_worker") as worker:
                await pilot.press("enter")
                await pilot.pause()
            rendered = " ".join(str(widget.render()) for widget in screen.query(Static))

        worker.assert_not_called()
        self.assertEqual(screen._step, "youtube")
        self.assertIn("must use http:// or https://", rendered)


if __name__ == "__main__":
    unittest.main()
