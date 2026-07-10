#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_fixture_contracts import (  # noqa: E402
    build_fixtures,
    normalize_result,
    render_pseudocode_snapshot,
    run_case,
)


CASES = [
    (
        "test_flags_union",
        {
            "name": "inspect_header",
            "synth": {"kind": "function", "target": "inspect_header", "var_idx": 0},
            "dump_functions": [
                "inspect_header",
                "inspect_float_view",
                "inspect_bits",
                "inspect_bytes",
            ],
        },
    ),
    (
        "test_global_union_overlay",
        {
            "name": "g_overlay_storage",
            "synth": {"kind": "global", "target": "g_overlay_storage"},
            "dump_functions": [
                "initialize_global_overlay",
                "seed_global_overlay",
                "inspect_global_overlay",
                "consume_global_overlay_u32",
                "consume_global_overlay_float",
                "consume_global_overlay_edges",
            ],
        },
    ),
]


def canonical_run(
    repo_root: Path,
    plugin_path: Path,
    idump_path: str,
    fixture: str,
    case: dict,
) -> str:
    raw_result, raw_output = run_case(
        repo_root,
        plugin_path,
        idump_path,
        fixture,
        case,
        debug_mode=False,
    )
    normalized = normalize_result(raw_result)
    pseudocode = render_pseudocode_snapshot(raw_output, case["dump_functions"])
    if raw_result.get("success") is not True:
        raise AssertionError(
            f"{fixture}/{case['name']}: deterministic probe did not synthesize "
            f"successfully: {raw_result.get('error_message')!r}"
        )
    structure = normalized.get("structure")
    if not isinstance(structure, dict) or not structure.get("fields"):
        raise AssertionError(
            f"{fixture}/{case['name']}: successful probe has no recovered fields"
        )
    if not pseudocode.strip():
        raise AssertionError(
            f"{fixture}/{case['name']}: pseudocode snapshot is empty"
        )
    fields = structure["fields"]
    if fixture == "test_flags_union":
        if sum(bool(field.get("is_bitfield")) for field in fields) < 3:
            raise AssertionError(
                f"{fixture}/{case['name']}: bitfield recovery regressed"
            )
        if not any(field.get("is_array") for field in fields):
            raise AssertionError(
                f"{fixture}/{case['name']}: array recovery regressed"
            )
    if fixture == "test_global_union_overlay" and not any(
        field.get("is_union_candidate") and len(field.get("union_members", [])) >= 2
        for field in fields
    ):
        raise AssertionError(
            f"{fixture}/{case['name']}: overlapping global union recovery regressed"
        )
    return json.dumps(
        {"result": normalized, "pseudocode": pseudocode},
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove deterministic live synthesis across fresh IDA databases"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if args.rounds < 2:
        raise RuntimeError("--rounds must be at least 2")

    repo_root = Path(args.repo_root).resolve()
    plugin_path = Path(args.plugin).resolve()
    if not plugin_path.exists():
        raise RuntimeError(f"plugin not found: {plugin_path}")

    build_fixtures(repo_root, *(fixture for fixture, _ in CASES))

    for fixture, case in CASES:
        baseline = None
        for run_index in range(args.rounds):
            actual = canonical_run(
                repo_root, plugin_path, args.idump, fixture, case
            )
            if baseline is None:
                baseline = actual
            elif actual != baseline:
                raise AssertionError(
                    f"{fixture}/{case['name']} changed on deterministic run "
                    f"{run_index + 1} of {args.rounds}"
                )
        print(
            f"[PASS] {fixture}/{case['name']}: "
            f"{args.rounds} byte-identical normalized runs",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
