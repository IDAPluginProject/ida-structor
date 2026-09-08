#!/usr/bin/env python3
"""Verify collector sequencing against actual Hex-Rays ctree and byte offsets."""
import argparse
import json
import sys
from pathlib import Path
from check_cpp_api_surface import run_api_command
from check_fixture_contracts import build_fixtures

HEADER = {16: 2, 24: 8}
BOUNDED = HEADER | {64: 4, 68: 4, 72: 4, 76: 4}
CASES = {
    "traverse_list": ({0: 8, 16: 4}, "self_alias"),
    "next_pointer_self_assignment": ({0: 8, 24: 2}, "self_assignment"),
    "offset_self_assignment": ({8: 8, 24: 2}, "self_assignment"),
    "assignment_index_loop": (BOUNDED, "assigned_index"),
    "compound_assignment_index": (BOUNDED, "assigned_index"),
    "assignment_postincrement_index": (BOUNDED, None),
    "assignment_preincrement_index": (HEADER | {68: 4, 72: 4, 76: 4, 80: 4}, None),
    "call_previously_escaped_index": (BOUNDED, "argument_load"),
    "call_address_before_value": (BOUNDED, "argument_load"),
    "call_nested_mutator_before_value": (HEADER, "sequential_mutation"),
    "call_postincrement_index": (HEADER, "hoisted_increment"),
}
MEMORY = {"cot_ptr", "cot_idx", "cot_memptr", "cot_memref"}
ASSIGNMENTS = {"cot_asg", "cot_asgadd", "cot_asgsub"}

def witness_present(data, kind):
    if kind is None:
        return True
    nodes = {node["id"]: node for node in data.get("ctree", [])}
    children = {node_id: [] for node_id in nodes}
    for node in nodes.values():
        if node.get("parent") in children:
            children[node["parent"]].append(node["id"])
    def descendants(node_id):
        result = []
        for child in children[node_id]:
            result.append(nodes[child])
            result.extend(descendants(child))
        return result
    if kind in {"argument_load", "sequential_mutation", "hoisted_increment"}:
        for call in nodes.values():
            if call.get("operation") != "cot_call" or "consume_" not in call.get("text", ""):
                continue
            nested = descendants(call["id"])
            index_vars = set()
            for multiply in nested:
                if multiply.get("operation") == "cot_mul":
                    index_vars.update(n["var_idx"] for n in descendants(multiply["id"])
                                      if n.get("operation") == "cot_var")
            ancestor = nodes.get(call.get("parent"))
            while ancestor and ancestor.get("opcode") != 73:  # cit_if
                ancestor = nodes.get(ancestor.get("parent"))
            if not ancestor:
                continue
            guards = [n for n in descendants(ancestor["id"])
                      if n.get("operation") == "cot_ult" and n.get("parent") == ancestor["id"]]
            guard_vars = {n["var_idx"] for g in guards for n in descendants(g["id"])
                          if n.get("operation") == "cot_var"}
            references = {n["var_idx"] for ref in nested if ref.get("operation") == "cot_ref"
                          for n in descendants(ref["id"]) if n.get("operation") == "cot_var"}
            if kind == "argument_load" and index_vars & guard_vars & references:
                return True
            if kind == "sequential_mutation" and index_vars & guard_vars:
                for effect in descendants(ancestor["id"]):
                    if (effect.get("operation") == "cot_call" and
                            "_mutate_index(" in effect.get("text", "") and effect["id"] < call["id"]):
                        effect_vars = {n["var_idx"] for ref in descendants(effect["id"])
                                       if ref.get("operation") == "cot_ref"
                                       for n in descendants(ref["id"]) if n.get("operation") == "cot_var"}
                        if effect_vars & index_vars & guard_vars:
                            return True
            if kind == "hoisted_increment":
                for assignment in descendants(ancestor["id"]):
                    if assignment.get("operation") != "cot_asg" or assignment["id"] >= call["id"]:
                        continue
                    direct = children[assignment["id"]]
                    if len(direct) != 2:
                        continue
                    lhs, rhs = (nodes[n] for n in direct)
                    increment_vars = {n["var_idx"] for n in descendants(rhs["id"])
                                      if n.get("operation") == "cot_var"}
                    if (lhs.get("var_idx") in index_vars and rhs.get("operation") == "cot_postinc" and
                            increment_vars & guard_vars & references):
                        return True
        return False
    target = data.get("pattern", {}).get("var_idx")
    for node in nodes.values():
        direct = children[node["id"]]
        if node.get("operation") not in ASSIGNMENTS or len(direct) < 2:
            continue
        lhs = nodes[direct[0]]
        if lhs.get("operation") != "cot_var":
            continue
        if kind == "self_alias" and lhs.get("var_idx") == target:
            continue
        if kind == "assigned_index" and "*" in lhs.get("type", ""):
            continue
        rhs_nodes = [nodes[direct[1]]] + descendants(direct[1])
        for memory in rhs_nodes:
            if memory.get("operation") in MEMORY and any(
                    n.get("operation") == "cot_var" and n.get("var_idx") == lhs["var_idx"]
                    for n in descendants(memory["id"])):
                return True
    return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--idump", default="idump")
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--record-dir")
    parser.add_argument("--constructed-only", action="store_true")
    parser.add_argument("--skip-constructed", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    plugin = Path(args.plugin).resolve()
    record = Path(args.record_dir).resolve() if args.record_dir else None
    if record:
        record.mkdir(parents=True, exist_ok=True)
    if not args.constructed_only:
        build_fixtures(root, "test_assignment_order", "test_linked_list")
    failures = []
    for name in ([] if args.constructed_only else args.case or CASES):
        data = run_api_command(root, plugin, args.idump,
            binary="test_linked_list" if name == "traverse_list" else "test_assignment_order", functions=[name],
            command=f"inspect_base_inference|{name}|0|ctree")
        if name.startswith("call_"):
            nodes = {node["id"]: node for node in data.get("ctree", [])}
            base_vars = []
            for node in nodes.values():
                if node.get("operation") == "cot_ptr" and "unsigned __int16" in node.get("type", ""):
                    stack = [node["id"]]
                    while stack:
                        parent = stack.pop()
                        children = [n for n in nodes.values() if n.get("parent") == parent]
                        base_vars.extend(n["var_idx"] for n in children if n.get("operation") == "cot_var")
                        stack.extend(n["id"] for n in children)
            if base_vars and base_vars[-1] != data.get("pattern", {}).get("var_idx"):
                selected = run_api_command(root, plugin, args.idump,
                    binary="test_assignment_order", functions=[name],
                    command=f"collect_accesses|{name}|{base_vars[-1]}")
                data["pattern"] = selected.get("pattern", {})
        if record:
            (record / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
        pattern = data.get("pattern", {})
        accesses = pattern.get("accesses", [])
        expected, kind = CASES[name]
        observed = {a["offset"]: a["size"] for a in accesses}
        correct_sizes = all(expected.get(a["offset"]) == a["size"] for a in accesses)
        witness = not data.get("ctree_truncated") and witness_present(data, kind)
        if observed != expected or not correct_sizes or pattern.get("has_vtable") or not witness:
            failures.append(f"{name}: expected {expected}, observed {observed}, actual ctree witness={witness}")
            print("[FAIL] " + failures[-1], flush=True)
        else:
            print(f"[PASS] {name}: {observed}; actual ctree witness={witness}", flush=True)
    if not args.skip_constructed and not args.case:
        build_fixtures(root, "test_assignment_order_ctree")
        data = run_api_command(root, plugin, args.idump,
            binary="test_assignment_order_ctree", functions=["assignment_order_ctree_carrier"],
            command="check_assignment_order_ctree|assignment_order_ctree_carrier")
        if record:
            (record / "constructed_ctree.json").write_text(json.dumps(data, indent=2) + "\n")
        cases = data.get("cases", [])
        if data.get("evidence") != "constructed SDK ctree" or len(cases) != 5:
            failures.append("constructed ctree evidence missing or wrong case count")
        for case in cases:
            passed = case.get("passed") and case.get("original_body_restored") and not case.get("error")
            print(f"[{'PASS' if passed else 'FAIL'}] constructed ctree {case.get('name')}", flush=True)
            if not passed:
                failures.append(f"constructed ctree {case.get('name')}: {case}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print("Assignment order regressions: PASS", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
