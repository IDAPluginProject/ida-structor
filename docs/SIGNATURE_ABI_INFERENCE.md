# Signature constraints and target ABI evidence

`TypeInferenceEngine::add_calling_convention_constraints` uses `cfunc_t::argidx` to map each recovered prototype parameter to its actual local-variable index. It validates the complete mapping before querying the derived function type or emitting constraints. Negative, out-of-range, duplicate, and cardinality-mismatched mappings produce no signature constraints. Existing constraint weights and unknown-type filtering remain in effect. The SDK defines `argidx` as indexes into the local-variable collection. [Hex-Rays `cfunc_t` reference](https://cpp.docs.hex-rays.com/structcfunc__t.html)

`CallingConventionDetector::detect_with_evidence` returns a convention and its provenance. `RecoveredPrototype` denotes a recognized legacy x86 convention in IDA's recovered function type. `TargetDefault` denotes a supported default family selected from the analyzed processor, address bitness, pointer size, object format, and database ABI name. `Unavailable` accompanies `Unknown`. None of these states establishes that IDA recovered an arbitrary function's source ABI correctly. `detect` remains a compatibility wrapper returning the convention alone.

## Represented targets

| Analyzed target | Additional conditions | Result and evidence |
| --- | --- | --- |
| x86, 32 bits | 4-byte pointers; recognized cdecl/stdcall/fastcall/thiscall; empty database ABI | Corresponding legacy name; `RecoveredPrototype` |
| x86, 64 bits, PE | 8-byte pointers; empty database ABI | Microsoft x64; `TargetDefault` |
| x86, 64 bits, ELF | 8-byte pointers; empty database ABI | System V x64; `TargetDefault` |
| x86, 64 bits, Mach-O | 8-byte pointers; empty or exactly `osx` database ABI | System V x64; `TargetDefault` |
| Arm, 32 bits, ELF | 4-byte pointers; `eabi` base ABI | AAPCS family; `TargetDefault` |
| Arm, 64 bits, ELF | 8-byte pointers; empty, `eabi`, or `aapcs64` base ABI | AAPCS64 family; `TargetDefault` |
| Other targets, custom/language function conventions, unrepresented ABI names | Includes Apple/Microsoft Arm variants, x32, AArch64 ILP32, and legacy Arm OABI | `Unknown`; `Unavailable` |

The Arm family labels do not synthesize argument locations or identify a floating-point variant. Extended SDK calling-convention values retain their full `callcnv_t` width. Host operating-system macros do not participate in target selection. SDK ABI names use a base name with optional suffixes. [Hex-Rays compiler/ABI functions](https://cpp.docs.hex-rays.com/group___c_c__funcs.html), [processor identifiers](https://cpp.docs.hex-rays.com/group___p_l_f_m__.html)

## Location model and units

Location synthesis accepts an explicit `ParameterPassingMode::FixedPrototype` contract. This contract asserts that the selected standard convention applies and that the input contains the complete lowered ABI argument list, including hidden arguments. A source parameter list alone does not meet that contract when an aggregate return requires a hidden pointer. A `TargetDefault` result alone also does not meet it. The default `Unspecified` mode, variadic/unprototyped modes, unsupported conventions, aggregates, vectors, and unsupported scalar widths produce an empty result.

The supported model covers x64 integer/pointer arguments of 1, 2, 4, or 8 bytes and floating-point arguments of 4 or 8 bytes. Microsoft x64 uses the first four argument positions to select RCX/RDX/R8/R9 or XMM0–XMM3; a floating-point argument at position 1 therefore uses XMM1. System V x64 allocates its integer and SSE register banks independently. Explicitly supplied hidden return pointers consume the corresponding integer slot and, for Microsoft x64, an argument position. [Microsoft x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170), [System V AMD64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI/-/blob/master/x86-64-ABI/low-level-sys-info.tex)

Stack offsets are measured in bytes from the callee-entry stack pointer before the prologue. The first Microsoft stack argument is at `8 bytes` of return address plus `32 bytes` of register home space, giving `SP + 40 bytes`; the next occupies `SP + 48 bytes`. The first System V stack argument is at `SP + 8 bytes`. Supported stack arguments consume 8-byte slots. Bounds checks reject an offset addition that would overflow the signed 64-bit representation. The implementation does not infer a prologue-relative displacement.

Apple's ordinary x64 scalar calls follow the System V register rules; Apple documents additional differences outside this model. Apple Arm and Microsoft Arm have platform-specific rules and are not assigned generic argument locations here. [Apple x64 ABI](https://developer.apple.com/documentation/xcode/writing-64-bit-intel-code-for-apple-platforms), [Apple Arm64 ABI](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms), [Microsoft Arm64 ABI](https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170), [AAPCS32](https://github.com/ARM-software/abi-aa/blob/main/aapcs32/aapcs32.rst), [AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)

## Assumption register and falsification probes

| ID | Assumption and dependent result | Stress test / falsification probe |
| --- | --- | --- |
| S1 | The supplied `argidx` and recovered prototype describe the same parameter sequence. Signature constraints depend on this mapping. | Live nonidentity permutation with distinguishable types; negative, out-of-range, and duplicate indexes; portable cardinality mismatch. Invalid mappings emit no facts. |
| S2 | Recovered parameter types are usable evidence, rather than source-type proof. Signature facts remain soft at the configured weight. | Live checker compares exact parameter/local/type/weight/source facts and verifies that lvar types, mapping, function prototype, and saved database type remain unchanged. |
| A1 | The analyzed processor, object format, pointer data model, and supported ABI name describe the target default. `TargetDefault` depends on these facts. | Cross-decompile ELF x64, ELF Arm32/AArch64, PE x64, and Mach-O x64/Arm64 on the same host. Unknown machine, ABI option, object format, and ILP32 controls remain unknown. |
| A2 | A particular function obeys that default unless independent function-level evidence establishes otherwise. Applying location results to a function depends on this assumption. | **Falsified for the two foreign-ABI fixtures.** `ms_abi` in ELF and `sysv_abi` in PE can be absent from stripped recovered prototypes. The checker retains the known source override, the default selection, and the contradictory recovered parameter locations. |
| L1 | The supplied convention is applicable, and the explicit fixed-prototype list includes hidden arguments. Every modeled location depends on L1. | A supplied hidden return pointer shifts visible GP arguments and Microsoft positional SSE arguments. An omitted hidden pointer cannot be detected by this API; its location result is invalid for that function. |
| L2 | Every argument belongs to the implemented x64 scalar class and data model. Every modeled location depends on L2. | Integer widths 3/16 bytes, floating widths 1/16 bytes, unsupported type classes, variadic/unprototyped/default modes, and unsupported conventions reject the entire plan. Mixed classes and register exhaustion check register banks and stack origins. |

GCC explicitly permits per-function `ms_abi` and `sysv_abi` overrides independently of the default object-file ABI. On the tested IDA 9.4 runtime, the stripped ELF override is recovered as generic FASTCALL with five SysV argument locations, and the PE override as generic FASTCALL with two Microsoft argument locations, despite each source function taking three arguments. Comparing those recovered locations with the selected default would falsely appear consistent. These fixtures establish a limitation; they do not justify adding source-ABI facts. [GCC x86 function attributes](https://gcc.gnu.org/onlinedocs/gcc/x86-Attributes.html)

## Validation and reproduction

Validation on 2026-09-08 passed 15 portable tests, five production signature-mapping scenarios, and six supported/unknown target-family checks. Two additional foreign-ABI fixtures produced the recorded falsification witnesses. Portable tests exercise the production mapping and ABI helpers. Live test hooks invoke the production signature-constraint method and detector in isolated `idump` databases. The two foreign override records are reported as falsification witnesses, separately from supported-target passes. No live test applies inferred types. Test-only command names are excluded from production builds and checked by the release artifact scan.

```sh
python3 integration_tests/check_signature_abi_regressions.py \
  --repo-root "$PWD" --plugin build/structor.dylib \
  --idump /path/to/idump --record-dir /tmp/structor-signature-abi-evidence
```

Cross-target fixture generation requires macOS Xcode command-line tools (`xcrun`/Clang), `ld.lld`, and `lld-link`; licensed decompilers for x86 and Arm are required for the live cases. The checker uses the existing sandbox runner and ad-hoc signs its installed plugin. A missing tool or decompiler fails the run explicitly.

For `p` parameters, mapping validation takes expected `O(p)` time and `O(p)` space with an unordered set (worst-case collision behavior is `O(p²)`). ABI-name selection takes `O(a)` time for `a` name bytes and `O(1)` space. Scalar location modeling takes `O(p)` time and `O(p)` output space. SDK prototype acquisition and decompilation costs are separate and are not bounded by these helper analyses.

## Bounded findings

- **High impact:** Source ABI overrides can be erased before this layer receives a recovered prototype. Per-function ABI reconstruction from machine-code uses remains outside this change; target defaults retain explicit provenance.
- **High impact:** Hidden return arguments and variadic register duplication require richer inputs. This location model requires a complete fixed ABI argument list and rejects variadic or unspecified mode.
- **Medium impact:** Recognizing an Arm family does not establish its floating-point or platform-specific argument locations. Those location results remain unavailable.
- **Low impact:** Empty location vectors represent either no arguments or unsupported inputs. Callers that need distinct diagnostics can use the input contract and detection evidence; no fabricated stack-only fallback is emitted.
