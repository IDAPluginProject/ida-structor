#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "integration_tests"))

from check_fixture_contracts import (  # noqa: E402
    extract_pseudocode_blocks,
    verify_case,
)


class FixtureContractParserTest(unittest.TestCase):
    def test_guessed_type_diagnostic_terminates_pseudocode_block(self) -> None:
        output = """\
Function: update_tag (0x100001000)
-- Pseudocode ---------------------------------------------------------------
auto_result *update_tag(auto_result *result) {
    result->tag = 1;
    return result;
}

100008000: using guessed type __int64 sink;
Function: next_function (0x100001100)
-- Pseudocode ---------------------------------------------------------------
void next_function(void) {
}
Summary
"""

        blocks = extract_pseudocode_blocks(output)

        self.assertEqual(
            blocks["update_tag"],
            "auto_result *update_tag(auto_result *result) {\n"
            "    result->tag = 1;\n"
            "    return result;\n"
            "}",
        )
        self.assertNotIn("using guessed type", blocks["update_tag"])
        self.assertEqual(blocks["next_function"], "void next_function(void) {\n}")

    def test_ordinary_pseudocode_is_preserved(self) -> None:
        output = """\
Function: read_value (0x100001000)
-- Pseudocode ---------------------------------------------------------------
int read_value(auto_result *result) {
    return result->value;
}
Summary
"""

        blocks = extract_pseudocode_blocks(output)

        self.assertEqual(
            blocks["read_value"],
            "int read_value(auto_result *result) {\n"
            "    return result->value;\n"
            "}",
        )

    def test_exact_contract_requires_both_goldens(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing golden_result"):
            verify_case(
                {"fixture": "fixture"},
                {"name": "case"},
                {},
                "",
                {},
                "",
            )

        with self.assertRaisesRegex(
            AssertionError, "missing golden_pseudocode"
        ):
            verify_case(
                {"fixture": "fixture"},
                {"name": "case", "golden_result": {}},
                {},
                "",
                {},
                "",
            )


if __name__ == "__main__":
    unittest.main()
