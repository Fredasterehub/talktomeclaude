"""Textual tests for mandatory clone reference review."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from textual.app import App, ComposeResult
from textual.screen import Screen

from talktomeclaude import clone_ui


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
                    await pilot.press("a")
                    await pilot.press("c")
                    await pilot.pause()

            add_f5.assert_called_once_with(
                "rick_f5",
                reference,
                "Reference transcript.",
            )


if __name__ == "__main__":
    unittest.main()
