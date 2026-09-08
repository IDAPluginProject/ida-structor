#!/usr/bin/env python3
"""Verify production collector ranges using fresh IDBs and the C++ API hooks."""

import argparse
import sys
from pathlib import Path

from check_cpp_api_surface import run_api_command
from check_fixture_contracts import build_fixtures


# Expected observed byte offsets, independently specified from the C fixture.
# Header reads are always at 0 and 8; a proven index in [0, 3] adds 32..44.
BASE = {0, 8}
BOUNDED = BASE | {32, 36, 40, 44}
CASES = {
    "read_unsigned_guard": BOUNDED,
    "read_signed_range": BOUNDED,
    "read_signed_upper_only": BASE,
    "read_prior_comparison": BASE,
    "read_later_comparison": BASE,
    "read_unbounded_else": BASE,
    "read_bounded_else": BOUNDED,
    "read_narrowed_guard": BASE,
    "read_large_guard": BASE,
    "read_reassigned_index": BASE,
    "read_alternative_guard": BASE,
    "read_unsigned_loop": BOUNDED,
    "read_do_loop": BASE,
    "read_guard_outside_mutating_loop": BASE,
    "read_nested_guard": BOUNDED,
    "read_unknown_index": BASE,
    "call_unknown_index": BASE,
    "read_truncated_index": BOUNDED,
    "read_wrapping_index": BASE | {32, 36, 40, 1052},
    "read_sparse_index": BASE | {32, 40, 48, 56},
    "read_reverse_index": BOUNDED,
    "read_float_cast_guard": BASE,
    "read_high_finite_range": BASE | {432, 436, 440, 444},
    "read_negative_finite_range": BASE | {24, 28, 32, 36},
    "read_truncated_base": BASE,
    "read_truncated_base_direct": BASE,
    "read_signedness_guard": BASE,
    "read_boolean_guard": BASE,
    "deref_unsigned_guard": {16, 24, 64, 68, 72, 76},
    "deref_prior_comparison": {16, 24},
    "deref_later_comparison": {16, 24},
    "deref_signed_upper_only": {16, 24},
    "deref_narrowed_guard": {16, 24},
    "deref_large_guard": {16, 24},
    "read_preincrement_index": BASE | {36, 40, 44, 48},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    plugin_path = Path(args.plugin).resolve()
    build_fixtures(repo_root, "test_index_guard_scope")

    failures = []
    for function in args.case or CASES:
        data = run_api_command(
            repo_root, plugin_path, args.idump,
            binary="test_index_guard_scope", functions=[function],
            command=f"collect_accesses|{function}|0",
        )
        pattern = data.get("pattern", {})
        accesses = pattern.get("accesses", [])
        actual = {access["offset"] for access in accesses}
        expected = CASES[function]
        expected_sizes = {16: 2, 24: 8} if function.startswith("deref_") else {8: 8}
        sizes_match = all(access.get("size") == expected_sizes.get(access["offset"], 4)
                          for access in accesses)
        if (not data.get("success") or actual != expected or not sizes_match
                or pattern.get("has_vtable")):
            failures.append(
                f"{function}: expected offsets {sorted(expected)}, "
                f"observed {sorted(actual)}; result={data}"
            )
            print(f"[FAIL] {failures[-1]}", flush=True)
        else:
            print(f"[PASS] {function}: {sorted(actual)}", flush=True)
    if failures:
        raise AssertionError("\n".join(failures))
    print("Index guard regressions: PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
