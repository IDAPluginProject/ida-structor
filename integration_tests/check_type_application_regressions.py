#!/usr/bin/env python3
"""Verify function identity before applying inferred local or signature types."""

import argparse
import sys
from pathlib import Path

from check_cpp_api_surface import require_success, run, run_api_command


CASES = (
    "foreign_apply", "unknown_apply", "high_address_apply", "foreign_propagate",
    "foreign_signature", "unknown_signature", "failed_signature",
    "same_apply", "same_signature",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    plugin = Path(args.plugin).resolve()
    if not plugin.is_file():
        raise RuntimeError(f"plugin not found: {plugin}")
    build = run(["sh", str(root / "integration_tests/build_fixtures.sh"),
                 "test_simple_struct"], cwd=root)
    require_success(build, "building type application fixture")

    failures = []
    for case in CASES:
        data = run_api_command(
            root, plugin, args.idump, binary="test_simple_struct",
            functions=["process_simple"],
            command=f"inspect_type_application_identity|process_simple|main|{case}",
        )
        expected = {"acceptance_matches_identity"}
        if case.startswith("same_"):
            expected.add("requested_signature_applied" if case.endswith("signature")
                         else "requested_local_applied")
        else:
            expected.update(("rejected_without_application", "saved_type_unchanged",
                             "local_types_unchanged"))
        checks = data.get("checks", {})
        if (set(checks) != expected or data.get("success") is not True
                or any(passed is not True for passed in checks.values())):
            failures.append(f"{case}: {data}")
            print(f"[FAIL] {failures[-1]}", flush=True)
        else:
            print(f"[PASS] type application: {case}", flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
