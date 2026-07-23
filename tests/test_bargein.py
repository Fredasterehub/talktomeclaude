"""Tests for the barge-in activation gate (LAW: bargein-gate)."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
