#!/usr/bin/env python3

import argparse
import contextlib
import hashlib
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
    normalize_result,
    render_pseudocode_snapshot,
    run_case,
    load_contracts,
    verify_case,
)


def function_case(
    name: str,
    target: str,
    dump_functions: list[str],
    *,
    var_idx: int = 0,
    config: dict | None = None,
    expect: dict | None = None,
):
    case = {
        "name": name,
        "synth": {
            "kind": "function",
            "target": target,
            "var_idx": var_idx,
        },
        "dump_functions": dump_functions,
    }
    if config:
        case["config"] = config
    if expect:
        case["expect"] = expect
    return case


def global_case(
    name: str,
    target: str,
    dump_functions: list[str],
    *,
    expect: dict | None = None,
):
    case = {
        "name": name,
        "synth": {
            "kind": "global",
            "target": target,
        },
        "dump_functions": dump_functions,
    }
    if expect:
        case["expect"] = expect
    return case


CONTRACT_MANIFEST = [
    {
        "fixture": "test_simple_struct",
        "cases": [
            function_case(
                "process_simple", "process_simple", ["process_simple", "init_simple"]
            ),
            function_case(
                "candidate_limit_failure",
                "process_simple",
                ["process_simple"],
                config={"z3_max_candidates": 1},
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "candidates",
                },
            ),
            function_case(
                "field_limit_failure",
                "process_simple",
                ["process_simple"],
                config={"z3_max_fields": 1},
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "fields",
                },
            ),
            function_case(
                "structure_span_limit_failure",
                "process_simple",
                ["process_simple"],
                config={"z3_max_structure_size": 8},
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "structure_size",
                },
            ),
            function_case(
                "access_limit_failure",
                "process_simple",
                ["process_simple"],
                config={"z3_max_accesses": 1},
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "accesses",
                },
            ),
            function_case(
                "constraint_pair_limit_failure",
                "process_simple",
                ["process_simple"],
                config={"z3_max_constraint_pairs": 1},
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "constraint_pairs",
                },
            ),
            function_case(
                "maxsmt_memory_limit_failure",
                "process_simple",
                ["process_simple"],
                config={
                    "z3_enable_maxsmt": True,
                    "z3_memory_limit_mb": 1,
                },
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "solver_memory",
                },
            ),
            function_case(
                "confidence_threshold_preserves_direct_evidence",
                "process_simple",
                ["process_simple"],
                config={"z3_min_confidence": 100},
                expect={
                    "success": True,
                    "z3_status": "success",
                    "used_fallback": False,
                    "required_field_offsets": [0, 8, 16],
                },
            ),
        ],
    },
    {
        "fixture": "test_function_ptr",
        "cases": [
            function_case(
                "invoke_handler",
                "invoke_handler",
                ["invoke_handler", "setup_handler", "update_and_invoke"],
            )
        ],
    },
    {
        "fixture": "test_linked_list",
        "cases": [
            function_case(
                "traverse_list",
                "traverse_list",
                ["traverse_list", "sum_list", "insert_after"],
            )
        ],
    },
    {
        "fixture": "test_mixed_access",
        "cases": [
            function_case(
                "read_mixed",
                "read_mixed",
                ["read_mixed", "write_mixed", "modify_mixed"],
            )
        ],
    },
    {
        "fixture": "test_nested",
        "cases": [
            function_case(
                "access_nested", "access_nested", ["access_nested", "modify_array"]
            )
        ],
    },
    {
        "fixture": "test_nested_2d",
        "cases": [
            function_case("read_matrix", "read_matrix", ["read_matrix", "read_marks"])
        ],
    },
    {
        "fixture": "test_substructure",
        "cases": [
            function_case(
                "process_data", "process_data", ["process_data", "process_node_d"]
            )
        ],
    },
    {
        "fixture": "test_callgraph_return",
        "cases": [
            function_case(
                "process_root",
                "process_root",
                ["process_root", "sibling_reader", "make_sub"],
            ),
            function_case("process_sub", "process_sub", ["process_sub"]),
        ],
    },
    {
        "fixture": "test_cross_conflict_union",
        "cases": [
            function_case(
                "process_conflict",
                "process_conflict",
                ["process_conflict", "read_payload_whole", "read_payload_split"],
            )
        ],
    },
    {
        "fixture": "test_packed_struct",
        "cases": [
            function_case(
                "read_packed",
                "read_packed",
                ["read_packed", "read_small_array", "inspect_flag_slices"],
            )
        ],
    },
    {
        "fixture": "test_packing_matrix",
        "cases": [
            function_case(
                "read_pack2", "read_pack2", ["read_pack2", "seed_pack2"]
            ),
            function_case(
                "read_pack4", "read_pack4", ["read_pack4", "seed_pack4"]
            ),
            function_case(
                "read_default_chars",
                "read_default_chars",
                ["read_default_chars", "seed_default_chars"],
            ),
        ],
    },
    {
        "fixture": "test_packed_nested_array",
        "cases": [
            function_case("read_bundle", "read_bundle", ["read_bundle", "read_tail"])
        ],
    },
    {
        "fixture": "test_packed_union_overlap",
        "cases": [
            function_case("read_whole", "read_whole", ["read_whole", "read_parts"])
        ],
    },
    {
        "fixture": "test_negative_offsets",
        "cases": [
            function_case("consume_window", "consume_window", ["consume_window"])
        ],
    },
    {
        "fixture": "test_array_of_structs",
        "cases": [
            function_case(
                "read_table",
                "read_table",
                ["read_table", "update_tag", "read_checksum"],
            )
        ],
    },
    {
        "fixture": "test_array_of_structs_nested",
        "cases": [
            function_case(
                "read_packets",
                "read_packets",
                ["read_packets", "write_packet_byte", "read_footer"],
            )
        ],
    },
    {
        "fixture": "test_bounded_index",
        "cases": [
            function_case(
                "read_indexed",
                "read_indexed",
                ["read_indexed", "read_inclusive", "read_marks"],
            ),
            function_case(
                "read_inclusive",
                "read_inclusive",
                ["read_inclusive", "read_indexed", "read_marks"],
            ),
            function_case(
                "read_reversed_bound",
                "read_reversed_bound",
                ["read_reversed_bound"],
            ),
        ],
    },
    {
        "fixture": "test_enum_constants",
        "cases": [
            function_case(
                "inspect_mode", "inspect_mode", ["inspect_mode", "inspect_state"]
            )
        ],
    },
    {
        "fixture": "test_flags_union",
        "cases": [
            function_case(
                "inspect_header",
                "inspect_header",
                [
                    "inspect_header",
                    "inspect_float_view",
                    "inspect_bits",
                    "inspect_bytes",
                ],
            ),
            function_case(
                "union_alternative_limit_failure",
                "inspect_header",
                ["inspect_header", "inspect_float_view"],
                config={"z3_max_union_alternatives": 1},
                expect={
                    "success": False,
                    "require_structure": False,
                    "z3_status": "resource_limit",
                    "used_fallback": False,
                    "resource_limit_kind": "union_alternatives",
                },
            ),
            function_case(
                "array_candidate_cap_preserves_scalar_evidence",
                "inspect_header",
                ["inspect_header", "inspect_bytes"],
                config={
                    "z3_min_array_elements": 2,
                    "z3_max_array_elements": 2,
                },
                expect={
                    "success": True,
                    "z3_status": "success",
                    "used_fallback": False,
                    "max_array_count": 2,
                    "required_field_offsets": [16, 17, 18, 19],
                },
            ),
        ],
    },
    {
        "fixture": "test_callback_table",
        "cases": [
            function_case(
                "invoke_slot0",
                "invoke_slot0",
                ["invoke_slot0", "invoke_slot2", "read_states"],
            )
        ],
    },
    {
        "fixture": "test_indirect_shifted_call",
        "cases": [
            function_case(
                "dispatch_parent",
                "dispatch_parent",
                ["dispatch_parent", "invoke_child"],
            )
        ],
    },
    {
        "fixture": "test_local_alias_positive",
        "cases": [
            function_case(
                "use_alias_read",
                "use_alias_read",
                ["use_alias_read", "use_alias_chain"],
            )
        ],
    },
    {
        "fixture": "test_pointer_field_pointee",
        "cases": [
            function_case(
                "use_pointer_field",
                "use_pointer_field",
                ["use_pointer_field"],
            )
        ],
    },
    {
        "fixture": "test_alias_lifetime",
        "cases": [
            function_case(
                "alias_rebind_read",
                "alias_rebind_read",
                ["alias_rebind_read"],
            ),
            function_case(
                "alias_overwrite_read",
                "alias_overwrite_read",
                ["alias_overwrite_read"],
            ),
        ],
    },
    {
        "fixture": "test_pointer_constants",
        "cases": [
            function_case(
                "configure_and_invoke",
                "configure_and_invoke",
                ["configure_and_invoke"],
            )
        ],
    },
    {
        "fixture": "test_mixed_subobject_deltas",
        "cases": [
            function_case(
                "read_mixed_anchor",
                "read_mixed_anchor",
                ["read_mixed_anchor", "read_child"],
            )
        ],
    },
    {
        "fixture": "test_shifted_siblings",
        "cases": [
            function_case(
                "process_parent",
                "process_parent",
                ["process_parent", "consume_child"],
            )
        ],
    },
    {
        "fixture": "test_recursive_ctor_chain",
        "cases": [
            function_case(
                "root_init",
                "root_init",
                ["root_init", "child_init", "leaf_init", "use_root"],
            )
        ],
    },
    {
        "fixture": "test_tree_struct",
        "cases": [
            function_case(
                "sum_children",
                "sum_children",
                ["sum_children", "walk_two_levels"],
            )
        ],
    },
    {
        "fixture": "test_partial_overlap",
        "cases": [
            function_case(
                "read_overlap",
                "read_overlap",
                ["read_overlap", "read_shifted_overlap"],
            )
        ],
    },
    {
        "fixture": "test_vtable_direct",
        "cases": [
            function_case(
                "access_object_fields",
                "access_object_fields",
                ["access_object_fields", "modify_object_fields", "increment_fields"],
            )
        ],
    },
    {
        "fixture": "test_vtable_positive",
        "cases": [
            function_case(
                "call_vtable_direct",
                "__Z18call_vtable_directPv",
                ["__Z18call_vtable_directPv", "__Z19call_multiple_slotsPvi"],
            )
        ],
    },
    {
        "fixture": "test_vtable",
        "cases": [
            function_case(
                "call_through_vtable",
                "__Z19call_through_vtablePv",
                ["__Z19call_through_vtablePv", "__Z12access_valuePv"],
            ),
            function_case(
                "main_dispatch_object",
                "main",
                ["main", "__Z19call_through_vtablePv", "__Z12access_valuePv"],
                var_idx=4,
            ),
        ],
    },
    {
        "fixture": "test_global_ctor_chain",
        "cases": [
            global_case(
                "g_widget",
                "g_widget",
                ["widget_ctor", "widget_use_global", "widget_use_leaf"],
            )
        ],
    },
    {
        "fixture": "test_global_ctor_return",
        "cases": [
            global_case("g_session", "g_session", ["session_ctor", "consume_session"])
        ],
    },
    {
        "fixture": "test_global_split_init",
        "cases": [
            global_case(
                "g_device",
                "g_device",
                ["device_header_ctor", "device_attach_cookie", "device_publish_slots"],
            )
        ],
    },
    {
        "fixture": "test_global_subobject_chain",
        "cases": [global_case("g_manager", "g_manager", ["manager_ctor"])],
    },
    {
        "fixture": "test_global_recursive_ctor_chain",
        "cases": [
            global_case(
                "g_root",
                "g_root",
                ["install_root", "root_ctor", "child_ctor", "leaf_ctor", "use_root"],
            )
        ],
    },
    {
        "fixture": "test_global_pointer_singleton",
        "cases": [
            global_case(
                "g_state_storage", "g_state_storage", ["state_ctor", "use_state"]
            )
        ],
    },
    {
        "fixture": "test_global_placement_new",
        "cases": [
            global_case(
                "g_gadget_storage",
                "g_gadget_storage",
                ["__Z23construct_gadget_stage1v", "__Z10use_gadgetv"],
            )
        ],
    },
    {
        "fixture": "test_global_adjacent_objects",
        "cases": [
            global_case(
                "g_dual_arena",
                "g_dual_arena",
                ["build_left", "build_right", "use_left", "use_right"],
            )
        ],
    },
    {
        "fixture": "test_global_cpp_static_ctor",
        "cases": [
            global_case(
                "g_engine",
                "g_engine",
                ["__Z12drive_enginev", "__Z14inspect_enginev"],
            )
        ],
    },
    {
        "fixture": "test_global_ambiguous_scratch",
        "cases": [
            global_case(
                "g_scratch",
                "g_scratch",
                ["fill_scratch", "scramble_scratch", "checksum_scratch"],
                expect={
                    "success": False,
                    "require_structure": False,
                    "error_contains": "No global/static structure accesses found",
                },
            )
        ],
    },
    {
        "fixture": "test_global_union_overlay",
        "cases": [
            global_case(
                "g_overlay_storage",
                "g_overlay_storage",
                [
                    "initialize_global_overlay",
                    "seed_global_overlay",
                    "inspect_global_overlay",
                    "consume_global_overlay_u32",
                    "consume_global_overlay_float",
                    "consume_global_overlay_edges",
                ],
            )
        ],
    },
    {
        "fixture": "test_static_local_singleton",
        "cases": [
            global_case(
                "local_cache",
                "__ZZL15get_local_cachevE5cache",
                ["__Z16warm_local_cachev", "__Z16read_local_cachev"],
            )
        ],
    },
]


def _generate_contracts_locked(
    repo_root: Path,
    plugin_path: Path,
    idump_path: str,
    contracts_dir: Path,
    selected_fixtures: set[str],
) -> None:
    manifest = [
        entry
        for entry in CONTRACT_MANIFEST
        if not selected_fixtures or entry["fixture"] in selected_fixtures
    ]
    if not manifest:
        raise RuntimeError("no fixture entries selected for contract generation")

    build_fixtures(repo_root, *(entry["fixture"] for entry in manifest))
    generated_contracts: dict[str, dict] = {}

    for entry in manifest:
        fixture_name = entry["fixture"]
        contract = {"fixture": fixture_name, "cases": []}

        for case in entry["cases"]:
            raw_result, raw_output = run_case(
                repo_root,
                plugin_path,
                idump_path,
                fixture_name,
                case,
                debug_mode=False,
            )
            normalized_result = normalize_result(raw_result)
            pseudocode_snapshot = render_pseudocode_snapshot(
                raw_output,
                case.get("snapshot_functions")
                or case.get("dump_functions")
                or [],
            )
            validation_case = dict(case)
            validation_expect = dict(case.get("expect", {}))
            validation_expect.setdefault("success", True)
            validation_expect.setdefault(
                "require_structure", validation_expect["success"]
            )
            validation_case["expect"] = validation_expect
            verify_case(
                {"fixture": fixture_name},
                validation_case,
                raw_result,
                raw_output,
                normalized_result,
                pseudocode_snapshot,
                require_goldens=False,
            )
            contract["cases"].append(
                {
                    "name": case["name"],
                    "synth": case["synth"],
                    "dump_functions": case["dump_functions"],
                    **({"config": case["config"]} if case.get("config") else {}),
                    **({"expect": case["expect"]} if case.get("expect") else {}),
                    "golden_result": normalized_result,
                    "golden_pseudocode": pseudocode_snapshot,
                }
            )
            print(f"[RECORDED] {fixture_name}/{case['name']}")

        generated_contracts[fixture_name] = contract

    # Do not mutate the blessed corpus until every requested live IDA run has
    # completed and the complete staged corpus passes the authoritative loader.
    contracts_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{contracts_dir.name}.stage-",
            dir=contracts_dir.parent,
        )
    )
    staged_contracts_dir = stage_root / contracts_dir.name
    backup_root: Path | None = None
    moved_original = False
    installed_stage = False
    try:
        if selected_fixtures and contracts_dir.exists():
            shutil.copytree(contracts_dir, staged_contracts_dir)
        else:
            staged_contracts_dir.mkdir()

        for fixture_name, contract in generated_contracts.items():
            output_path = staged_contracts_dir / f"{fixture_name}.json"
            output_path.write_text(
                json.dumps(contract, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )

        # Always validate the entire corpus. A selected regeneration therefore
        # cannot bless files against a missing or stale peer contract.
        load_contracts(staged_contracts_dir, [])

        if contracts_dir.exists():
            backup_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{contracts_dir.name}.backup-",
                    dir=contracts_dir.parent,
                )
            )
            os.replace(contracts_dir, backup_root / contracts_dir.name)
            moved_original = True

        try:
            os.replace(staged_contracts_dir, contracts_dir)
            installed_stage = True
        except BaseException as install_error:
            if moved_original and backup_root is not None:
                try:
                    os.replace(
                        backup_root / contracts_dir.name, contracts_dir
                    )
                    moved_original = False
                except BaseException as restore_error:
                    raise RuntimeError(
                        "contract corpus installation and restoration failed; "
                        f"original corpus retained at "
                        f"{backup_root / contracts_dir.name}"
                    ) from restore_error
            raise install_error

        for fixture_name in generated_contracts:
            print(f"[WROTE] {contracts_dir / f'{fixture_name}.json'}")
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
        if backup_root is not None and (installed_stage or not moved_original):
            shutil.rmtree(backup_root, ignore_errors=True)


@contextlib.contextmanager
def contract_recorder_lock(contracts_dir: Path):
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError(
            "contract generation requires POSIX advisory file locking"
        ) from exc

    lock_id = hashlib.sha256(str(contracts_dir).encode("utf-8")).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"structor-contracts-{lock_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another contract recorder holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def recover_interrupted_swap(contracts_dir: Path) -> None:
    backup_roots = sorted(
        path
        for path in contracts_dir.parent.glob(
            f".{contracts_dir.name}.backup-*"
        )
        if path.is_dir()
    )
    stage_roots = sorted(
        path
        for path in contracts_dir.parent.glob(f".{contracts_dir.name}.stage-*")
        if path.is_dir()
    )
    if len(backup_roots) > 1:
        raise RuntimeError(
            "multiple interrupted corpus backups require inspection: "
            + ", ".join(str(path) for path in backup_roots)
        )
    if backup_roots:
        backup_root = backup_roots[0]
        backup_corpus = backup_root / contracts_dir.name
        if contracts_dir.exists():
            raise RuntimeError(
                "an interrupted corpus swap retained both destination and "
                f"backup; no data was removed: {backup_corpus}"
            )
        if not backup_corpus.is_dir():
            raise RuntimeError(
                f"interrupted backup is missing its corpus: {backup_root}"
            )
        os.replace(backup_corpus, contracts_dir)
        shutil.rmtree(backup_root)

    if stage_roots:
        if not contracts_dir.exists():
            raise RuntimeError(
                "staged corpus exists without a recoverable destination: "
                + ", ".join(str(path) for path in stage_roots)
            )
        for stage_root in stage_roots:
            shutil.rmtree(stage_root)


def generate_contracts(
    repo_root: Path,
    plugin_path: Path,
    idump_path: str,
    contracts_dir: Path,
    selected_fixtures: set[str],
) -> None:
    expected_contracts_dir = repo_root / "integration_tests" / "contracts"
    if expected_contracts_dir.is_symlink() or (
        contracts_dir != expected_contracts_dir.resolve()
    ):
        raise RuntimeError(
            "--contracts-dir must resolve to the repository contract corpus: "
            f"{expected_contracts_dir}"
        )

    known_fixtures = {entry["fixture"] for entry in CONTRACT_MANIFEST}
    unknown_fixtures = selected_fixtures - known_fixtures
    if unknown_fixtures:
        raise RuntimeError(
            "unknown fixture selection(s): "
            + ", ".join(sorted(unknown_fixtures))
        )

    contracts_dir.parent.mkdir(parents=True, exist_ok=True)
    with contract_recorder_lock(contracts_dir):
        recover_interrupted_swap(contracts_dir)
        _generate_contracts_locked(
            repo_root,
            plugin_path,
            idump_path,
            contracts_dir,
            selected_fixtures,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate exact live fixture contracts from the current blessed baseline"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    parser.add_argument(
        "--contracts-dir",
        default="integration_tests/contracts",
        help="Destination directory for generated contract JSON files",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Only regenerate the named fixture contract (can be repeated)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    plugin_path = Path(args.plugin).resolve()
    contracts_dir = (repo_root / args.contracts_dir).resolve()
    if not plugin_path.exists():
        raise RuntimeError(f"plugin not found: {plugin_path}")

    generate_contracts(
        repo_root,
        plugin_path,
        args.idump,
        contracts_dir,
        set(args.fixture),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
