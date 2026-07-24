from __future__ import annotations

import unittest

from talktomeclaude.speech.canonical import (
    BlockKind,
    CanonicalError,
    ProtectedKind,
    canonicalize,
)


class CanonicalAnswerTests(unittest.TestCase):
    def test_markdown_corpus_has_stable_ids_and_exact_full_coverage(self) -> None:
        source = (
            "# Deployment\r\n"
            "Version 2.4 may fail at 17% (see [RFC 9110]).\r\n"
            "Use `python -m app` from C:\\DEV\\app\\main.py. Warning: never delete /srv/data.\r\n"
            "\r\n"
            "| Item | Value |\r\n"
            "| --- | ---: |\r\n"
            "| retries | 3 |\r\n"
            "- keep the exact port 443\r\n"
            "- risk: timeout could occur\r\n"
            "> Citation [12] remains attached.\r\n"
            "```powershell\r\n"
            "Get-Item C:\\DEV\\app\r\n"
            "```\r\n"
            "Trailing prose remains covered."
        )

        first = canonicalize("answer-1", source)
        second = canonicalize("answer-1", source)

        self.assertEqual(first, second)
        self.assertEqual(source, "".join(block.text for block in first.blocks))
        self.assertEqual(
            [
                BlockKind.HEADING,
                BlockKind.PROSE,
                BlockKind.TABLE,
                BlockKind.LIST,
                BlockKind.QUOTE,
                BlockKind.FENCE,
                BlockKind.PROSE,
            ],
            [block.kind for block in first.blocks],
        )
        self.assertEqual(len(first.blocks), len({block.block_id for block in first.blocks}))
        values = [
            value
            for block in first.blocks
            for value in block.protected_values
        ]
        kinds = {value.kind for value in values}
        self.assertTrue(
            {
                ProtectedKind.CITATION,
                ProtectedKind.PATH,
                ProtectedKind.COMMAND,
                ProtectedKind.NUMBER,
                ProtectedKind.UNCERTAINTY,
                ProtectedKind.RISK,
            }
            <= kinds
        )
        exact = {value.value for value in values}
        self.assertIn("2.4", exact)
        self.assertIn("17%", exact)
        self.assertIn("`python -m app`", exact)
        self.assertNotIn("Trailing prose remains covered", repr(first))

    def test_balanced_and_unbalanced_fences_are_lossless(self) -> None:
        balanced = "Before\n```py\nprint(1)\n```\nAfter"
        unbalanced = "Before\n~~~sh\necho 42\ntrailing inside fence"

        balanced_answer = canonicalize("balanced", balanced)
        unbalanced_answer = canonicalize("unbalanced", unbalanced)

        self.assertEqual(balanced, "".join(item.text for item in balanced_answer.blocks))
        self.assertEqual(unbalanced, "".join(item.text for item in unbalanced_answer.blocks))
        self.assertEqual(BlockKind.FENCE, balanced_answer.blocks[1].kind)
        self.assertEqual(BlockKind.FENCE, unbalanced_answer.blocks[1].kind)
        self.assertTrue(unbalanced_answer.blocks[1].text.endswith("trailing inside fence"))

    def test_identity_and_text_validation_fail_closed(self) -> None:
        for answer_id, text in (("bad/id", "answer"), ("ok", "")):
            with self.subTest(answer_id=answer_id), self.assertRaises(CanonicalError):
                canonicalize(answer_id, text)

    def test_content_bearing_values_are_hidden_from_repr(self) -> None:
        answer = canonicalize(
            "private-answer",
            "SECRET path C:\\private\\answer.txt must remain.",
        )

        rendered = repr(answer)
        self.assertNotIn("SECRET", rendered)
        self.assertNotIn("private\\answer", rendered)


if __name__ == "__main__":
    unittest.main()
