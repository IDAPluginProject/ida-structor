#include <gtest/gtest.h>

#include <structor/z3/calling_convention_model.hpp>
#include <structor/z3/signature_argument_map.hpp>

#include <array>
#include <map>
#include <string_view>

namespace structor::z3::test {
namespace {
using namespace detail;
using Convention = CallingConvention;

CallingConventionTarget target(TargetMachine machine, unsigned bits,
                               TargetObjectFormat format,
                               std::string_view abi = {}) {
    return {machine, bits, bits / 8, format, abi};
}

constexpr ABIArgument integer(unsigned size = 8) {
    return {ABIArgumentClass::IntegerOrPointer, size};
}
constexpr ABIArgument floating(unsigned size = 8) {
    return {ABIArgumentClass::Floating, size};
}
} // namespace

TEST(SignatureArgumentMapTest, NonidentityMappingConstrainsOnlyMappedLocals) {
    const std::array argument_indexes{3, 1, 4};
    const std::array signature_types{"pointer", "double", "uint32"};
    const auto bindings = map_signature_arguments(argument_indexes, 3, 6);
    ASSERT_TRUE(bindings.has_value());
    std::map<int, std::string_view> local_constraints;
    for (const auto& binding : *bindings) {
        local_constraints[binding.local_index] = signature_types[binding.parameter_index];
    }
    EXPECT_EQ(local_constraints.at(3), "pointer");
    EXPECT_EQ(local_constraints.at(1), "double");
    EXPECT_EQ(local_constraints.at(4), "uint32");
    EXPECT_FALSE(local_constraints.contains(0));
    EXPECT_FALSE(local_constraints.contains(2));
    EXPECT_FALSE(local_constraints.contains(5));
}

TEST(SignatureArgumentMapTest, IdentityAndZeroArgumentMappingsRemainValid) {
    const std::array argument_indexes{0, 1};
    const auto bindings = map_signature_arguments(argument_indexes, 2, 5);
    ASSERT_TRUE(bindings.has_value());
    ASSERT_EQ(bindings->size(), 2u);
    EXPECT_EQ((*bindings)[0].local_index, 0);
    EXPECT_EQ((*bindings)[1].local_index, 1);
    const auto empty = map_signature_arguments({}, 0, 0);
    ASSERT_TRUE(empty.has_value());
    EXPECT_TRUE(empty->empty());
}

TEST(SignatureArgumentMapTest, InvalidOrDuplicateMappingsRejectAllFacts) {
    for (const auto& argument_indexes : {
            std::vector<int>{-1, 1}, {0, 6}, {2, 2}, {1, 0, 1}}) {
        EXPECT_FALSE(map_signature_arguments(argument_indexes,
            argument_indexes.size(), 6).has_value());
    }
    const std::array argument_indexes{3, 1};
    EXPECT_FALSE(map_signature_arguments(argument_indexes, 1, 6).has_value());
    EXPECT_FALSE(map_signature_arguments(argument_indexes, 3, 6).has_value());
    EXPECT_FALSE(map_signature_arguments(argument_indexes, 2, 0).has_value());
}

TEST(CallingConventionTargetTest, X64FollowsAnalyzedFileFormat) {
    for (const auto signature : {
            SignatureConvention::Default, SignatureConvention::CDecl,
            SignatureConvention::Stdcall, SignatureConvention::Fastcall,
            SignatureConvention::Thiscall}) {
        EXPECT_EQ(select_calling_convention(target(TargetMachine::X86, 64,
            TargetObjectFormat::PE), signature), Convention::Microsoft_x64);
        EXPECT_EQ(select_calling_convention(target(TargetMachine::X86, 64,
            TargetObjectFormat::ELF), signature), Convention::SystemV_x64);
        EXPECT_EQ(select_calling_convention(target(TargetMachine::X86, 64,
            TargetObjectFormat::MachO), signature), Convention::SystemV_x64);
        EXPECT_EQ(select_calling_convention(target(TargetMachine::X86, 64,
            TargetObjectFormat::MachO, "osx"), signature), Convention::SystemV_x64);
    }
}

TEST(CallingConventionTargetTest, ArmIsNeverClassifiedAsX64) {
    EXPECT_EQ(select_calling_convention(target(TargetMachine::Arm, 64,
        TargetObjectFormat::ELF), SignatureConvention::Fastcall),
        Convention::ARM64_AAPCS64);
    EXPECT_EQ(select_calling_convention(target(TargetMachine::Arm, 32,
        TargetObjectFormat::ELF, "eabi"), SignatureConvention::CDecl),
        Convention::ARM_AAPCS);
    EXPECT_EQ(select_calling_convention(target(TargetMachine::Arm, 32,
        TargetObjectFormat::ELF, "eabi-hard_float"), SignatureConvention::Fastcall),
        Convention::ARM_AAPCS);
    for (const auto format : {TargetObjectFormat::PE, TargetObjectFormat::MachO}) {
        EXPECT_EQ(select_calling_convention(target(TargetMachine::Arm, 64, format),
            SignatureConvention::Fastcall), Convention::Unknown);
    }
}

TEST(CallingConventionTargetTest, X86LegacyNamesRequire32BitTarget) {
    const auto x86 = target(TargetMachine::X86, 32, TargetObjectFormat::PE);
    EXPECT_EQ(select_calling_convention(x86, SignatureConvention::CDecl), Convention::CDecl);
    EXPECT_EQ(select_calling_convention(x86, SignatureConvention::Stdcall), Convention::Stdcall);
    EXPECT_EQ(select_calling_convention(x86, SignatureConvention::Fastcall), Convention::Fastcall);
    EXPECT_EQ(select_calling_convention(x86, SignatureConvention::Thiscall), Convention::Thiscall);
    EXPECT_EQ(select_calling_convention(x86, SignatureConvention::Default), Convention::Unknown);
    EXPECT_EQ(select_calling_convention(target(TargetMachine::Arm, 32,
        TargetObjectFormat::ELF, "eabi"), SignatureConvention::Thiscall), Convention::Unknown);
}

TEST(CallingConventionTargetTest, UnknownTargetsAndDataModelsRemainUnknown) {
    for (const auto& unknown : {
            target(TargetMachine::Unknown, 64, TargetObjectFormat::ELF),
            target(TargetMachine::X86, 64, TargetObjectFormat::Unknown),
            target(TargetMachine::X86, 16, TargetObjectFormat::PE),
            target(TargetMachine::Arm, 32, TargetObjectFormat::ELF),
            target(TargetMachine::X86, 64, TargetObjectFormat::ELF, "custom"),
            target(TargetMachine::X86, 64, TargetObjectFormat::PE, "-pack_stkargs"),
            target(TargetMachine::X86, 64, TargetObjectFormat::MachO, "osx-custom"),
            target(TargetMachine::X86, 64, TargetObjectFormat::ELF, "osx"),
            target(TargetMachine::Arm, 64, TargetObjectFormat::ELF, "custom")}) {
        EXPECT_EQ(select_calling_convention(unknown, SignatureConvention::Default),
            Convention::Unknown);
    }
    for (const auto machine : {TargetMachine::X86, TargetMachine::Arm}) {
        auto ilp32 = target(machine, 64, TargetObjectFormat::ELF);
        ilp32.pointer_size = 4;
        EXPECT_EQ(select_calling_convention(ilp32, SignatureConvention::Default),
            Convention::Unknown);
    }
}

TEST(CallingConventionTargetTest, UnsupportedFunctionABICannotFallBackToFileABI) {
    for (const auto machine : {TargetMachine::X86, TargetMachine::Arm}) {
        for (const auto format : {
                TargetObjectFormat::PE, TargetObjectFormat::ELF, TargetObjectFormat::MachO}) {
            EXPECT_EQ(select_calling_convention(target(machine, 64, format),
                SignatureConvention::Unsupported), Convention::Unknown);
        }
    }
}

TEST(CallingConventionLocationTest, MicrosoftMixedParametersUsePositionalRegisters) {
    const std::array arguments{
        integer(4), floating(), integer(), floating(4), integer(4), floating()};
    const auto locations = fixed_x64_parameter_locations(Convention::Microsoft_x64,
        arguments, ParameterPassingMode::FixedPrototype);
    ASSERT_EQ(locations.size(), arguments.size());
    const std::array registers{"rcx", "xmm1", "r8", "xmm3"};
    for (std::size_t index = 0; index < registers.size(); ++index) {
        EXPECT_TRUE(locations[index].is_register);
        EXPECT_EQ(locations[index].register_name, registers[index]);
    }
    EXPECT_FALSE(locations[4].is_register);
    EXPECT_EQ(locations[4].stack_offset, 40);
    EXPECT_FALSE(locations[5].is_register);
    EXPECT_EQ(locations[5].stack_offset, 48);
}

TEST(CallingConventionLocationTest, SystemVMixedParametersUseIndependentRegisterBanks) {
    const std::array arguments{integer(), floating(), integer(4), floating(4)};
    const auto locations = fixed_x64_parameter_locations(Convention::SystemV_x64,
        arguments, ParameterPassingMode::FixedPrototype);
    ASSERT_EQ(locations.size(), 4u);
    const std::array registers{"rdi", "xmm0", "rsi", "xmm1"};
    for (std::size_t index = 0; index < registers.size(); ++index) {
        EXPECT_TRUE(locations[index].is_register);
        EXPECT_EQ(locations[index].register_name, registers[index]);
    }
}

TEST(CallingConventionLocationTest, ExplicitHiddenReturnPointerConsumesArgumentSlot) {
    // Complete lowered ABI list for a memory-return function with visible
    // parameters (uint64_t, double). The helper does not infer this pointer.
    const std::array arguments{integer(), integer(), floating()};
    const auto microsoft = fixed_x64_parameter_locations(Convention::Microsoft_x64,
        arguments, ParameterPassingMode::FixedPrototype);
    ASSERT_EQ(microsoft.size(), 3u);
    EXPECT_EQ(microsoft[0].register_name, "rcx");
    EXPECT_EQ(microsoft[1].register_name, "rdx");
    EXPECT_EQ(microsoft[2].register_name, "xmm2");
    const auto sysv = fixed_x64_parameter_locations(Convention::SystemV_x64,
        arguments, ParameterPassingMode::FixedPrototype);
    ASSERT_EQ(sysv.size(), 3u);
    EXPECT_EQ(sysv[0].register_name, "rdi");
    EXPECT_EQ(sysv[1].register_name, "rsi");
    EXPECT_EQ(sysv[2].register_name, "xmm0");
}

TEST(CallingConventionLocationTest, SystemVRegisterExhaustionUsesCalleeEntryStackOffsets) {
    std::vector<ABIArgument> arguments(7, integer());
    arguments.insert(arguments.end(), 9, floating());
    const auto locations = fixed_x64_parameter_locations(Convention::SystemV_x64,
        arguments, ParameterPassingMode::FixedPrototype);
    ASSERT_EQ(locations.size(), 16u);
    EXPECT_EQ(locations[5].register_name, "r9");
    EXPECT_FALSE(locations[6].is_register);
    EXPECT_EQ(locations[6].stack_offset, 8);
    EXPECT_EQ(locations[7].register_name, "xmm0");
    EXPECT_EQ(locations[14].register_name, "xmm7");
    EXPECT_FALSE(locations[15].is_register);
    EXPECT_EQ(locations[15].stack_offset, 16);
}

TEST(CallingConventionLocationTest, UnsupportedConventionsHaveNoInventedStackLayout) {
    const std::array arguments{integer(), integer()};
    for (const auto convention : {
            Convention::Unknown, Convention::CDecl, Convention::Stdcall,
            Convention::Fastcall, Convention::Thiscall,
            Convention::ARM_AAPCS, Convention::ARM64_AAPCS64}) {
        EXPECT_TRUE(fixed_x64_parameter_locations(convention, arguments,
            ParameterPassingMode::FixedPrototype).empty());
    }
}

TEST(CallingConventionLocationTest, VariadicUnprototypedAndUnspecifiedModesRemainUnknown) {
    const std::array arguments{floating()};
    for (const auto convention : {Convention::Microsoft_x64, Convention::SystemV_x64}) {
        for (const auto mode : {ParameterPassingMode::Unspecified,
                ParameterPassingMode::Variadic, ParameterPassingMode::Unprototyped}) {
            EXPECT_TRUE(fixed_x64_parameter_locations(convention, arguments, mode).empty());
        }
    }
}

TEST(CallingConventionLocationTest, UnsupportedTypesRejectTheEntireLocationPlan) {
    for (const auto unsupported : {
            ABIArgument{}, ABIArgument{ABIArgumentClass::Unsupported, 8},
            integer(16), integer(3), floating(16), floating(1)}) {
        const std::array arguments{integer(), unsupported, floating()};
        for (const auto convention : {Convention::Microsoft_x64, Convention::SystemV_x64}) {
            EXPECT_TRUE(fixed_x64_parameter_locations(convention, arguments,
                ParameterPassingMode::FixedPrototype).empty());
        }
    }
}

} // namespace structor::z3::test
