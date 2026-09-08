#!/usr/bin/env python3
"""Verify experimental semantics extraction and identity against live IDA ctree."""

import argparse
import sys
from pathlib import Path

from check_cpp_api_surface import require_success, run, run_api_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    plugin = Path(args.plugin).resolve()
    build = run(["sh", str(root / "integration_tests/build_fixtures.sh"),
                 "test_simple_struct"], cwd=root)
    require_success(build, "building semantics identity host fixture")
    data = run_api_command(
        root, plugin, args.idump,
        binary="test_simple_struct", functions=["process_simple"],
        command="inspect_instruction_semantics_identity|process_simple",
    )
    expected = ("success", "full_extraction", "shared_sorts",
                "independent_variables", "engine_extraction")
    if any(data.get(name) is not True for name in expected):
        raise RuntimeError(f"production semantics identity checks failed: {data}")
    if data.get("constraints", 0) <= 0 or data.get("expressions", 0) <= 0:
        raise RuntimeError(f"full ctree extraction returned no evidence: {data}")
    print(f"[PASS] production semantics identity: {data['constraints']} constraints "
          f"from {data['expressions']} expressions", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
