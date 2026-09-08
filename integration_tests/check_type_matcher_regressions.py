#!/usr/bin/env python3
"""Exercise existing-type overlays against IDA's real anonymous type system."""

import argparse
import sys
from pathlib import Path

from check_cpp_api_surface import require_success, run, run_api_command


EXPECTED_CHECKS = {
    "rejected_overlap_preserves_state",
    "padding_split_preserves_extent",
    "array_expansion_updates_metadata",
    "scalar_replacement_clears_array_metadata",
    "type_size_mismatch_is_rejected",
    "imported_names_preserve_analyst_names",
    "observed_bitfield_is_preserved",
    "existing_bitfield_is_not_byte_overlay",
    "distinct_udt_members_are_not_equivalent",
    "distinct_pointees_are_not_equivalent",
    "distinct_callback_signatures_are_not_equivalent",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    plugin_path = Path(args.plugin).resolve()
    if not plugin_path.is_file():
        raise RuntimeError(f"plugin not found: {plugin_path}")

    # A minimal binary supplies an IDA session; the check uses only anonymous
    # tinfo_t values and does not persist or apply a synthesized type.
    build = run(
        ["sh", str(repo_root / "integration_tests/build_fixtures.sh"),
         "test_simple_struct"],
        cwd=repo_root,
    )
    require_success(build, "building matcher host fixture")
    result = run_api_command(
        repo_root, plugin_path, args.idump,
        binary="test_simple_struct", functions=["process_simple"],
        command="inspect_existing_type_matcher",
    )
    checks = result.get("checks", {})
    if set(checks) != EXPECTED_CHECKS:
        raise RuntimeError(f"missing or unexpected matcher checks: {result}")
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if result.get("success") is not True or failed:
        raise RuntimeError(f"live matcher failures {failed}: {result}")
    print(f"[PASS] {len(checks)} existing-type matcher regressions with real IDA types",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
