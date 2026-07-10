#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_fixture_contracts import (  # noqa: E402
    build_fixtures,
    expand_function_filters,
    normalize_result,
    prepare_plugin_home,
    require_success,
    run,
    strip_ansi,
    write_structor_config,
)


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, message: str, output: str | None = None) -> None:
    if condition:
        return
    if output:
        raise RuntimeError(f"{message}\n{output}")
    raise RuntimeError(message)


def run_repeated_synthesis(
    repo_root: Path,
    plugin_path: Path,
    idump_path: str,
    fixture: str,
    targets: str | tuple[str, ...],
    functions: list[str],
    fault_stage: str | None = None,
    fault_iteration: int = 0,
    iterations: int = 2,
) -> list[tuple[dict, dict, str]]:
    sandbox_home = prepare_plugin_home(plugin_path, Path.home())
    write_structor_config(sandbox_home, debug_mode=False)
    run_dir = Path(tempfile.mkdtemp(prefix="structor-persistence-binary."))
    source_binary = repo_root / "integration_tests" / fixture
    sandbox_binary = run_dir / source_binary.name
    shutil.copy2(source_binary, sandbox_binary)
    saved_database = sandbox_binary.with_suffix(".i64")
    result_path = sandbox_home / "structor_last_result.json"

    try:
        env = os.environ.copy()
        env["HOME"] = str(sandbox_home)
        env["STRUCTOR_EXPORT_LAST_RESULT"] = str(result_path)
        iteration_targets = (
            tuple(targets for _ in range(iterations))
            if isinstance(targets, str)
            else targets
        )
        require(
            len(iteration_targets) == iterations,
            "persistence regression target count does not match iteration count",
        )

        results: list[tuple[dict, dict, str]] = []
        for iteration in range(iterations):
            input_path = sandbox_binary if iteration == 0 else saved_database
            if iteration != 0:
                require(
                    saved_database.exists(),
                    f"persistence regression: idump did not save {saved_database}",
                )
            result_path.unlink(missing_ok=True)
            env["STRUCTOR_AUTO_SYNTH"] = f"{iteration_targets[iteration]}:0"
            if iteration == fault_iteration and fault_stage is not None:
                env["STRUCTOR_INTEGRATION_TESTING"] = "1"
                env["STRUCTOR_TEST_PERSISTENCE_FAULT"] = fault_stage
            else:
                env.pop("STRUCTOR_INTEGRATION_TESTING", None)
                env.pop("STRUCTOR_TEST_PERSISTENCE_FAULT", None)
            proc = run(
                [
                    idump_path,
                    "--plugin",
                    "structor",
                    "--keep-i64",
                    "--pseudo-only",
                    "-F",
                    ",".join(expand_function_filters(functions)),
                    str(input_path),
                ],
                cwd=repo_root,
                env=env,
            )
            require_success(
                proc,
                f"running persistence regression {fixture} iteration {iteration}",
            )
            output = strip_ansi((proc.stdout or "") + (proc.stderr or ""))
            require(
                result_path.exists(),
                f"persistence regression: missing result for {fixture} iteration {iteration}",
                output,
            )
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            results.append((raw, normalize_result(raw), output))
        return results
    finally:
        shutil.rmtree(sandbox_home, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


def check_equal_layout_update(
    repo_root: Path, plugin_path: Path, idump_path: str
) -> None:
    first, second = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_packed_union_overlap",
        "read_whole",
        ["read_whole", "read_parts"],
    )
    first_raw, first_normalized, first_output = first
    second_raw, second_normalized, second_output = second
    combined_output = first_output + second_output

    require(
        first_raw.get("success") is True,
        "initial packed synthesis failed",
        combined_output,
    )
    require(
        second_raw.get("success") is True,
        "equal-layout update was refused",
        combined_output,
    )
    require(
        first_raw.get("struct_tid") == second_raw.get("struct_tid"),
        "equal-layout update changed the named type TID",
        combined_output,
    )
    require(
        first_normalized.get("structure") == second_normalized.get("structure"),
        "equal-layout update changed the recovered structure",
        combined_output,
    )
    structure = second_normalized.get("structure") or {}
    require(
        (structure.get("size"), structure.get("alignment"), structure.get("packing"))
        == (7, 1, 1),
        "equal-layout update did not preserve size/alignment/pack(1)",
        combined_output,
    )
    require(
        "Refusing generated-type update" not in second_output,
        "equal-layout update triggered the destructive-shrink guard",
        combined_output,
    )


def check_noninteractive_reuse_isolation(
    repo_root: Path, plugin_path: Path, idump_path: str
) -> None:
    first, second = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_simple_struct",
        ("process_simple", "init_simple"),
        ["process_simple", "init_simple"],
    )
    first_raw, first_normalized, first_output = first
    second_raw, second_normalized, second_output = second
    combined_output = first_output + second_output

    require(
        first_raw.get("success") is True,
        "initial reuse-isolation synthesis failed",
        combined_output,
    )
    require(
        second_raw.get("success") is True,
        "second reuse-isolation synthesis failed",
        combined_output,
    )
    require(
        first_raw.get("struct_tid") != second_raw.get("struct_tid"),
        "noninteractive synthesis silently reused the prior named type",
        combined_output,
    )
    first_structure = first_normalized.get("structure") or {}
    second_structure = second_normalized.get("structure") or {}
    require(
        first_structure.get("name") != second_structure.get("name"),
        "noninteractive synthesis replaced its generated name with a reuse candidate",
        combined_output,
    )
    require(
        (first_structure.get("size"), first_structure.get("alignment"))
        == (second_structure.get("size"), second_structure.get("alignment"))
        == (24, 8),
        "reuse-isolation fixtures did not produce comparable 24-byte ABI layouts",
        combined_output,
    )
    require(
        "Reuse existing struct" not in second_output,
        "noninteractive synthesis entered the interactive reuse path",
        combined_output,
    )


def check_destructive_shrink_guard(
    repo_root: Path, plugin_path: Path, idump_path: str
) -> None:
    first, second = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_array_of_structs",
        "read_table",
        ["read_table", "update_tag", "read_checksum"],
    )
    first_raw, first_normalized, first_output = first
    second_raw, _, second_output = second
    combined_output = first_output + second_output

    require(
        first_raw.get("success") is True,
        "initial array synthesis failed",
        combined_output,
    )
    first_structure = first_normalized.get("structure") or {}
    require(
        first_structure.get("size") == 80,
        f"initial array synthesis produced {first_structure.get('size')} bytes instead of 80",
        combined_output,
    )
    require(
        second_raw.get("success") is False,
        "destructive repeated synthesis unexpectedly replaced the generated type",
        combined_output,
    )
    require(
        "size would shrink from 80 to 8 bytes" in second_output,
        "destructive-shrink refusal diagnostic was absent",
        combined_output,
    )
    require(
        "not creating a duplicate replacement" in second_output,
        "destructive-shrink path did not suppress duplicate type creation",
        combined_output,
    )


def check_union_bitfield_roundtrip(
    repo_root: Path, plugin_path: Path, idump_path: str
) -> None:
    first, second = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_flags_union",
        "inspect_header",
        ["inspect_header", "inspect_float_view", "inspect_bits", "inspect_bytes"],
    )
    first_raw, first_normalized, first_output = first
    second_raw, _, second_output = second
    combined_output = first_output + second_output

    require(
        first_raw.get("success") is True,
        "union/bitfield persistence failed its verified IDB round trip",
        combined_output,
    )
    # Applying the first recovered type intentionally changes the second-pass
    # decompilation: only the source function's 16-byte view remains as fresh
    # evidence.  The monotonic update guard must reject that lossy replacement.
    require(
        second_raw.get("success") is False and
        "size would shrink from 24 to 16 bytes" in second_output,
        "union/bitfield second pass bypassed the destructive-update guard",
        combined_output,
    )

    structure = first_normalized.get("structure") or {}
    require(
        (structure.get("size"), structure.get("alignment"), structure.get("packing"))
        == (24, 4, None),
        "union/bitfield round trip lost the 24-byte default-packed ABI layout",
        combined_output,
    )
    fields = structure.get("fields") or []
    unions = [field for field in fields if field.get("is_union_candidate")]
    require(
        len(unions) == 1 and len(unions[0].get("union_members") or []) == 2,
        "union/bitfield round trip did not retain both exact union alternatives",
        combined_output,
    )
    bitfields = {
        (field.get("offset"), field.get("bit_offset"), field.get("bit_size"))
        for field in fields
        if field.get("is_bitfield")
    }
    require(
        {(14, 0, 2), (14, 2, 3), (14, 5, 1)} <= bitfields,
        "union/bitfield round trip did not retain exact bit ranges",
        combined_output,
    )
    require(
        "Round-trip verification failed" not in first_output,
        "union/bitfield persistence emitted a round-trip mismatch",
        first_output,
    )


def check_transaction_fault_rollback(
    repo_root: Path, plugin_path: Path, idump_path: str
) -> None:
    functions = ["inspect_header", "inspect_float_view", "inspect_bits", "inspect_bytes"]

    before_root, recovered = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_flags_union",
        "inspect_header",
        functions,
        fault_stage="before_root_write",
    )
    failed_raw, _, failed_output = before_root
    recovered_raw, _, recovered_output = recovered
    require(
        failed_raw.get("success") is False and recovered_raw.get("success") is True,
        "pre-root fault did not fail closed and recover on the clean second pass",
        failed_output + recovered_output,
    )
    require(
        "Transaction rollback removed new type 'union_8'" in failed_output and
        "Transaction rollback removed new type 'auto_result'" not in failed_output,
        "pre-root rollback did not remove exactly the materialized union auxiliary",
        failed_output,
    )
    require(
        "Creating struct 'auto_result'" in recovered_output,
        "pre-root rollback left a root type that the second pass reused",
        recovered_output,
    )

    apply_failure, recovered = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_flags_union",
        "inspect_header",
        functions,
        fault_stage="required_source_apply",
    )
    failed_raw, _, failed_output = apply_failure
    recovered_raw, _, recovered_output = recovered
    require(
        failed_raw.get("success") is False and
        failed_raw.get("struct_tid") in (None, -1, 18446744073709551615) and
        recovered_raw.get("success") is True,
        "required-apply fault did not return a truthful failure and recover",
        failed_output + recovered_output,
    )
    require(
        "Transaction rollback removed new type 'auto_result'" in failed_output and
        "Transaction rollback removed new type 'union_8'" in failed_output,
        "required-apply rollback did not remove both root and union auxiliary",
        failed_output,
    )
    require(
        "Creating struct 'auto_result'" in recovered_output,
        "required-apply rollback left a root type that the second pass reused",
        recovered_output,
    )

    created, replacement_failed, restored = run_repeated_synthesis(
        repo_root,
        plugin_path,
        idump_path,
        "test_packed_union_overlap",
        "read_whole",
        ["read_whole", "read_parts"],
        fault_stage="required_source_apply",
        fault_iteration=1,
        iterations=3,
    )
    created_raw, created_normalized, created_output = created
    failed_raw, _, failed_output = replacement_failed
    restored_raw, restored_normalized, restored_output = restored
    combined_output = created_output + failed_output + restored_output
    require(
        created_raw.get("success") is True and
        failed_raw.get("success") is False and
        restored_raw.get("success") is True,
        "existing-type replacement fault did not fail closed and recover",
        combined_output,
    )
    require(
        "Transaction rollback restored type 'auto_result'" in failed_output and
        "Transaction rollback removed new type 'auto_result'" not in failed_output,
        "replacement rollback did not restore the pre-transaction owned root type",
        failed_output,
    )
    require(
        created_raw.get("struct_tid") == restored_raw.get("struct_tid") and
        created_normalized.get("structure") == restored_normalized.get("structure"),
        "post-rollback synthesis did not observe the original root identity and layout",
        combined_output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run focused live structure-persistence regressions"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    plugin_path = Path(args.plugin).resolve()
    require(plugin_path.exists(), f"plugin not found: {plugin_path}")

    build_fixtures(
        repo_root,
        "test_packed_union_overlap",
        "test_array_of_structs",
        "test_simple_struct",
        "test_flags_union",
    )
    log("Persistence regression: checking noninteractive reuse isolation")
    check_noninteractive_reuse_isolation(repo_root, plugin_path, args.idump)
    log("Persistence regression: checking equal-layout replacement")
    check_equal_layout_update(repo_root, plugin_path, args.idump)
    log("Persistence regression: checking destructive-shrink refusal")
    check_destructive_shrink_guard(repo_root, plugin_path, args.idump)
    log("Persistence regression: checking union/bitfield round trips")
    check_union_bitfield_roundtrip(repo_root, plugin_path, args.idump)
    log("Persistence regression: checking transactional fault rollback")
    check_transaction_fault_rollback(repo_root, plugin_path, args.idump)
    log("Persistence regression: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
