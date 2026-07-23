"""Tests for the wake-word module: lazy optional-dependency import and the
DEFAULT_WAKE_PHRASE it exposes."""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock


class WakeWordModuleTests(unittest.TestCase):
    def test_default_wake_phrase(self) -> None:
        from talktomeclaude.wakeword import DEFAULT_WAKE_PHRASE

        self.assertEqual(DEFAULT_WAKE_PHRASE, "yo claude")

    def test_module_imports_without_the_optional_engine_installed(self) -> None:
        # Simulate a machine that never installed the optional wake-word
        # engine: sys.modules[name] = None makes any `import <name>` raise
        # ImportError. Reloading the module under that condition must still
        # succeed, proving the optional import is deferred, never eager.
        sys.modules.pop("talktomeclaude.wakeword", None)
        with mock.patch.dict(sys.modules, {"openwakeword": None}):
            module = importlib.import_module("talktomeclaude.wakeword")
            importlib.reload(module)
            self.assertEqual(module.DEFAULT_WAKE_PHRASE, "yo claude")


if __name__ == "__main__":
    unittest.main()
