#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace structor::z3 {

enum class CallingConvention {
    Unknown,
    CDecl,
    Stdcall,
    Fastcall,
    Thiscall,
    SystemV_x64,
    Microsoft_x64,
    ARM_AAPCS,
    ARM64_AAPCS64,
};

enum class CallingConventionEvidence {
    Unavailable,
    RecoveredPrototype,
    TargetDefault,
};

struct CallingConventionDetection {
    CallingConvention convention = CallingConvention::Unknown;
    CallingConventionEvidence evidence = CallingConventionEvidence::Unavailable;
};

/// Location synthesis requires an explicit fixed-prototype contract. A
/// single-location result cannot represent duplicated vararg registers.
enum class ParameterPassingMode {
    Unspecified,
    FixedPrototype,
    Variadic,
    Unprototyped,
};

namespace detail {

enum class TargetMachine { Unknown, X86, Arm };
enum class TargetObjectFormat { Unknown, PE, ELF, MachO };
enum class SignatureConvention {
    Default,
    CDecl,
    Stdcall,
    Fastcall,
    Thiscall,
    Unsupported,
};

struct CallingConventionTarget {
    TargetMachine machine = TargetMachine::Unknown;
    unsigned bitness = 0;
    unsigned pointer_size = 0; // bytes in the analyzed data model
    TargetObjectFormat format = TargetObjectFormat::Unknown;
    std::string_view database_abi;
};

/// Select the represented default ABI from analyzed target facts. Host
/// compilation macros are deliberately absent. Unrepresented explicit ABI
/// names and language/user conventions remain unknown. O(a) time for a ABI
/// name bytes and O(1) space.
[[nodiscard]] inline CallingConvention select_calling_convention(
    const CallingConventionTarget& target, SignatureConvention signature)
{
    if (signature == SignatureConvention::Unsupported) {
        return CallingConvention::Unknown;
    }
    // SDK ABI names have the form base-option1-option2. Arm family detection
    // accepts its named EABI variants without promising their locations.
    const auto base = target.database_abi.substr(
        0, target.database_abi.find('-'));
    if (target.machine == TargetMachine::X86) {
        const bool apple_x64 = target.bitness == 64 &&
            target.format == TargetObjectFormat::MachO && target.database_abi == "osx";
        if (!target.database_abi.empty() && !apple_x64) return CallingConvention::Unknown;
        if (target.bitness == 32 && target.pointer_size == 4) {
            switch (signature) {
                case SignatureConvention::CDecl: return CallingConvention::CDecl;
                case SignatureConvention::Stdcall: return CallingConvention::Stdcall;
                case SignatureConvention::Fastcall: return CallingConvention::Fastcall;
                case SignatureConvention::Thiscall: return CallingConvention::Thiscall;
                default: return CallingConvention::Unknown;
            }
        }
        if (target.bitness != 64 || target.pointer_size != 8) {
            return CallingConvention::Unknown;
        }
        switch (target.format) {
            case TargetObjectFormat::PE: return CallingConvention::Microsoft_x64;
            case TargetObjectFormat::ELF:
            case TargetObjectFormat::MachO: return CallingConvention::SystemV_x64;
            default: return CallingConvention::Unknown;
        }
    }
    if (target.machine == TargetMachine::Arm &&
        target.format == TargetObjectFormat::ELF) {
        if (signature == SignatureConvention::Stdcall ||
            signature == SignatureConvention::Thiscall) {
            return CallingConvention::Unknown;
        }
        if (target.bitness == 32 && target.pointer_size == 4 && base == "eabi") {
            return CallingConvention::ARM_AAPCS;
        }
        if (target.bitness == 64 && target.pointer_size == 8 &&
            (base.empty() || base == "eabi" || base == "aapcs64")) {
            return CallingConvention::ARM64_AAPCS64;
        }
    }
    // Apple and Microsoft Arm variants, OABI, ILP32-on-AArch64, x32,
    // unrecognized machines, and unidentified file ABIs are not represented.
    return CallingConvention::Unknown;
}

enum class ABIArgumentClass { Unsupported, IntegerOrPointer, Floating };

struct ABIArgument {
    ABIArgumentClass kind = ABIArgumentClass::Unsupported;
    unsigned size = 0; // bytes
};

struct ABIParameterLocation {
    bool is_register = false;
    std::string_view register_name;
    // Bytes from the callee-entry stack pointer, before the prologue.
    std::int64_t stack_offset = 0;
};

/// Model only fixed-prototype x64 scalar/pointer arguments of at most eight
/// bytes. The supplied convention and complete ABI argument list (including
/// hidden arguments) are preconditions, not inferred facts. Unsupported inputs
/// return no guessed locations. O(p) time/space.
[[nodiscard]] inline std::vector<ABIParameterLocation>
fixed_x64_parameter_locations(CallingConvention convention,
                              std::span<const ABIArgument> arguments,
                              ParameterPassingMode mode)
{
    if (mode != ParameterPassingMode::FixedPrototype ||
        (convention != CallingConvention::SystemV_x64 &&
         convention != CallingConvention::Microsoft_x64)) {
        return {};
    }
    for (const auto& argument : arguments) {
        const bool integer = argument.kind == ABIArgumentClass::IntegerOrPointer &&
            (argument.size == 1 || argument.size == 2 ||
             argument.size == 4 || argument.size == 8);
        const bool floating = argument.kind == ABIArgumentClass::Floating &&
            (argument.size == 4 || argument.size == 8);
        if (!integer && !floating) return {};
    }

    constexpr std::string_view sysv_integer[] = {
        "rdi", "rsi", "rdx", "rcx", "r8", "r9"};
    constexpr std::string_view microsoft_integer[] = {"rcx", "rdx", "r8", "r9"};
    constexpr std::string_view floating[] = {
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"};
    std::size_t integer_index = 0;
    std::size_t floating_index = 0;
    std::int64_t stack_offset =
        convention == CallingConvention::Microsoft_x64 ? 40 : 8;
    std::vector<ABIParameterLocation> result;
    result.reserve(arguments.size());
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        ABIParameterLocation location;
        if (convention == CallingConvention::Microsoft_x64 && index < 4) {
            location.is_register = true;
            location.register_name = arguments[index].kind == ABIArgumentClass::Floating
                ? floating[index] : microsoft_integer[index];
        } else if (convention == CallingConvention::SystemV_x64) {
            if (arguments[index].kind == ABIArgumentClass::Floating) {
                if (floating_index < 8) {
                    location.is_register = true;
                    location.register_name = floating[floating_index++];
                }
            } else if (integer_index < 6) {
                location.is_register = true;
                location.register_name = sysv_integer[integer_index++];
            }
        }
        if (!location.is_register) {
            location.stack_offset = stack_offset;
            if (stack_offset > std::numeric_limits<std::int64_t>::max() - 8) {
                return {};
            }
            stack_offset += 8;
        }
        result.push_back(location);
    }
    return result;
}

} // namespace detail
} // namespace structor::z3
