#!/usr/bin/env python3
"""Verify typed array discovery through candidate generation and production Z3 layout."""

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
    if not plugin.is_file():
        raise RuntimeError(f"plugin not found: {plugin}")
    build = run(["sh", str(root / "integration_tests/build_fixtures.sh"),
                 "test_simple_struct"], cwd=root)
    require_success(build, "building array solver host fixture")

    layouts = {}
    for case in ("independent", "overlay", "overlay_reverse", "sparse", "interior_conflict",
                 "unsat", "unsat_relaxation"):
        data = run_api_command(
            root, plugin, args.idump,
            binary="test_simple_struct", functions=["process_simple"],
            command=f"inspect_array_layout|{case}",
        )
        if data.get("success") is not True or data.get("diagnostics_detached") is not True:
            raise RuntimeError(f"production array layout failed for {case}: {data}")
        if case.startswith("unsat"):
            if data.get("unsat_core_count", 0) <= 0 or data.get("result", {}).get("success") is not False:
                raise RuntimeError(f"missing actual UNSAT failure diagnostics for {case}: {data}")
            print(f"[PASS] production UNSAT diagnostics lifetime: {case}", flush=True)
            continue
        if data.get("layout_shape_valid") is not True:
            raise RuntimeError(f"production array shape failed for {case}: {data}")
        structure = data.get("result", {}).get("structure")
        if not structure:
            raise RuntimeError(f"missing array layout for {case}: {data}")
        if case == "interior_conflict":
            # The newly discovered integer run remains an optional hypothesis.
            # The interior float observation must survive even though this
            # prevents selecting that array in the current layout model.
            expected_preferences = [
                f"Prefer richer aggregate at 0x0 over overlap at 0x{offset:X}"
                for offset in (0, 4, 8, 8)
            ]
            z3 = data["result"].get("z3", {})
            if (z3.get("status") != "success_relaxed" or
                    z3.get("arrays_detected") != 1 or
                    sorted(data.get("dropped_constraints", [])) != expected_preferences):
                raise RuntimeError(f"interior conflict lost optional array provenance: {data}")
        layouts[case] = structure
        print(f"[PASS] production array layout: {case}", flush=True)

    # Compare actual materialized field descriptions, not only solver status.
    if layouts["overlay"]["fields"] != layouts["overlay_reverse"]["fields"]:
        raise RuntimeError(f"array union layout depends on observation order: {layouts}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
