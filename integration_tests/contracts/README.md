# Exact fixture snapshots

These JSON files assert both normalized synthesis results and rendered
pseudocode. The verifier does not discard field metadata, propagation sites, or
pseudocode differences. A snapshot change requires inspection of the affected
storage and control flow; recording a failing run is not sufficient evidence.

## Runtime baseline: 2026-09-08

Six cases were refreshed using macOS arm64, IDA/Hex-Rays 9.4, an `idump` built
against that SDK, and Apple Clang 21.0.0. The prior snapshot-generating runtime is
unknown. A complete build of commit `a2252de` and the pending accuracy changes
produced identical normalized results and pseudocode for these cases:

| Fixture / case | Difference from the prior snapshot |
| --- | --- |
| `test_function_ptr / invoke_handler` | Three observed fields occupy the same 24-byte structure; four bytes of terminal alignment padding are implicit. |
| `test_global_ctor_chain / g_widget` | Propagation additionally records `widget_ctor_stage2`, which forwards the same object. |
| `test_global_placement_new / g_gadget_storage` | Propagation additionally records the two constructor forwarding wrappers; pseudocode retains the wrapper call. |
| `test_pointer_constants / configure_and_invoke` | The same pointer constants render as named functions and global objects. |
| `test_vtable / call_through_vtable` | The propagation set reflects the current constructor representations; `access_value` receives the synthesized object type. |
| `test_vtable / main_dispatch_object` | The propagation set includes both derived constructors; pseudocode retains constructor and `access_value` calls. |

Assumption: these exact snapshots describe this runtime and fixture toolchain.
Falsification probe: rebuild the unchanged baseline and changed plugin against
the same SDK, run each in a fresh database, and compare both outputs. Cross-SDK
equivalence is not established by these snapshots.

## Accuracy-change comparisons

- The recursive-constructor snapshots retain their unsigned 64-bit tail-field
  requirement. Their pseudocode and recorded root provenance were refreshed
  only where the unchanged baseline already produced the same difference.
- `test_linked_list / traverse_list` retains its existing field layout. Both
  the unchanged and changed plugins preserve the count limit present in the
  machine code but missing from the old pseudocode snapshot. A collector-only
  comparison attributes the removed pointer-union alternative to an address
  alias compared with zero, rather than a pointer-typed memory access.
- `test_flags_union / inspect_header` preserves every field and changes only
  the solver status to `success_relaxed`. Typed-run recovery retains an optional
  array interpretation whose aggregate preferences lose to the observed
  scalar/union layout. A separate live interior-conflict check verifies the
  exact retained storage and dropped-preference diagnostics.
