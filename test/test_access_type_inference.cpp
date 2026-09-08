#include <gtest/gtest.h>

#include "mock_ida.hpp"
#include <structor/access_type_inference.hpp>
#include <structor/function_variable_key.hpp>

namespace structor::test {
namespace {

tinfo_t scalar(std::uint32_t flags) {
    tinfo_t result;
    result.create_simple_type(flags);
    return result;
}

tinfo_t pointer_to(const tinfo_t& pointee) {
    tinfo_t result;
    result.create_ptr(pointee);
    return result;
}

FieldAccess load(const tinfo_t& type, sval_t offset = 0, ea_t site = 0x1000) {
    FieldAccess result;
    result.offset = offset;
    result.size = static_cast<std::uint32_t>(type.get_size());
    result.inferred_type = type;
    result.access_type = AccessType::Read;
    result.insn_ea = site;
    result.source_func_ea = 0x1000;
    result.base_indirection = 1;
    return result;
}

void expect_void_pointer(const tinfo_t& type) {
    ASSERT_TRUE(type.is_ptr());
    EXPECT_TRUE(type.get_pointed_object().is_void());
}

} // namespace

TEST(AccessBaseTypeInferenceTest, ScalarLoadInfersPointerToLoadedType) {
    AccessPattern pattern;
    const auto int32 = scalar(BTF_INT32);
    pattern.accesses.push_back(load(int32));
    TypeConfidence confidence;

    const auto inferred = detail::infer_base_type_from_accesses(pattern, confidence);
    ASSERT_TRUE(inferred.is_ptr());
    EXPECT_TRUE(inferred.get_pointed_object().equals_to(int32));
    EXPECT_EQ(confidence, TypeConfidence::Low);
}

TEST(AccessBaseTypeInferenceTest, PointerLoadRetainsAdditionalIndirection) {
    AccessPattern pattern;
    const auto int32_pointer = pointer_to(scalar(BTF_INT32));
    pattern.accesses.push_back(load(int32_pointer));
    TypeConfidence confidence;

    const auto inferred = detail::infer_base_type_from_accesses(pattern, confidence);
    ASSERT_TRUE(inferred.is_ptr());
    EXPECT_TRUE(inferred.get_pointed_object().equals_to(int32_pointer));
}

TEST(AccessBaseTypeInferenceTest, DirectMemberLoadCanReportZeroDerefCount) {
    AccessPattern pattern;
    const auto int32 = scalar(BTF_INT32);
    auto access = load(int32);
    access.base_indirection.reset();
    pattern.accesses.push_back(access);
    TypeConfidence confidence;

    const auto inferred = detail::infer_base_type_from_accesses(pattern, confidence);
    EXPECT_TRUE(inferred.equals_to(pointer_to(int32)));
}

TEST(AccessBaseTypeInferenceTest, CallbackSlotRetainsPointerToFunctionPointer) {
    func_type_data_t signature;
    signature.rettype = scalar(BTF_INT32);
    tinfo_t function;
    ASSERT_TRUE(function.create_func(signature));
    const auto callback = pointer_to(function);
    AccessPattern pattern;
    pattern.accesses.push_back(load(callback));
    TypeConfidence confidence;

    const auto inferred = detail::infer_base_type_from_accesses(pattern, confidence);
    ASSERT_TRUE(inferred.is_ptr());
    EXPECT_TRUE(inferred.get_pointed_object().equals_to(callback));
}

TEST(AccessBaseTypeInferenceTest, RepeatedSameFieldDoesNotBecomeAggregate) {
    AccessPattern pattern;
    const auto int32 = scalar(BTF_INT32);
    pattern.accesses.push_back(load(int32));
    pattern.accesses.push_back(load(int32, 0, 0x1004));
    TypeConfidence confidence;

    const auto inferred = detail::infer_base_type_from_accesses(pattern, confidence);
    EXPECT_TRUE(inferred.equals_to(pointer_to(int32)));
    EXPECT_EQ(confidence, TypeConfidence::Medium);
}

TEST(AccessBaseTypeInferenceTest, DuplicatingAnObservationDoesNotRaiseConfidence) {
    AccessPattern pattern;
    for (int i = 0; i < 8; ++i) {
        pattern.accesses.push_back(load(scalar(BTF_INT32)));
    }
    TypeConfidence confidence;

    (void)detail::infer_base_type_from_accesses(pattern, confidence);
    EXPECT_EQ(confidence, TypeConfidence::Low);
}

TEST(AccessBaseTypeInferenceTest, DisplacedAccessDoesNotIdentifyBasePointee) {
    for (sval_t displacement : {sval_t(-4), sval_t(4), sval_t(16)}) {
        AccessPattern pattern;
        pattern.accesses.push_back(load(scalar(BTF_INT32), displacement));
        TypeConfidence confidence;
        expect_void_pointer(detail::infer_base_type_from_accesses(pattern, confidence));
    }
}

TEST(AccessBaseTypeInferenceTest, MultipleFieldsRequireLayoutSynthesis) {
    AccessPattern pattern;
    pattern.accesses.push_back(load(scalar(BTF_INT32)));
    pattern.accesses.push_back(load(pointer_to(scalar(BTF_VOID)), 8, 0x1004));
    TypeConfidence confidence;
    expect_void_pointer(detail::infer_base_type_from_accesses(pattern, confidence));
}

TEST(AccessBaseTypeInferenceTest, IncompatibleViewsDoNotPickAPriorityWinner) {
    for (const auto& alternative : {scalar(BTF_INT64), scalar(BTF_FLOAT)}) {
        AccessPattern pattern;
        pattern.accesses.push_back(load(scalar(BTF_INT32)));
        pattern.accesses.push_back(load(alternative, 0, 0x1004));
        TypeConfidence confidence;
        expect_void_pointer(detail::infer_base_type_from_accesses(pattern, confidence));
    }
}

TEST(AccessBaseTypeInferenceTest, NarrowLoadDoesNotShrinkExistingPointee) {
    AccessPattern pattern;
    pattern.original_type = pointer_to(scalar(BTF_INT64));
    pattern.accesses.push_back(load(scalar(BTF_INT32)));
    TypeConfidence confidence;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pattern.original_type));
}

TEST(AccessBaseTypeInferenceTest, PartialEvidenceDoesNotErasePointerDepth) {
    AccessPattern pattern;
    pattern.original_type = pointer_to(pointer_to(scalar(BTF_INT32)));
    pattern.accesses.push_back(load(scalar(BTF_INT32), 8));
    TypeConfidence confidence;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pattern.original_type));
}

TEST(AccessBaseTypeInferenceTest, SameWidthScalarViewDoesNotErasePointerDepth) {
    AccessPattern pattern;
    pattern.original_type = pointer_to(pointer_to(scalar(BTF_INT32)));
    pattern.accesses.push_back(load(scalar(BTF_UINT64)));
    TypeConfidence confidence;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pattern.original_type));
}

TEST(AccessBaseTypeInferenceTest, ShallowerPointerViewDoesNotEraseOriginalDepth) {
    AccessPattern pattern;
    pattern.original_type = pointer_to(pointer_to(pointer_to(scalar(BTF_INT32))));
    pattern.accesses.push_back(load(pointer_to(scalar(BTF_VOID))));
    TypeConfidence confidence;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pattern.original_type));
}

TEST(AccessBaseTypeInferenceTest, GenericBasePointerCanStillBeSpecialized) {
    AccessPattern pattern;
    pattern.original_type = pointer_to(scalar(BTF_VOID));
    pattern.accesses.push_back(load(scalar(BTF_INT32)));
    TypeConfidence confidence;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pointer_to(scalar(BTF_INT32))));

    pattern.original_type = pointer_to(pointer_to(scalar(BTF_VOID)));
    pattern.accesses.front() = load(pointer_to(scalar(BTF_INT32)));
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pointer_to(pointer_to(scalar(BTF_INT32)))));
}

TEST(AccessBaseTypeInferenceTest, GenericViewsDoNotEraseFunctionPrototype) {
    func_type_data_t signature;
    signature.rettype = scalar(BTF_INT32);
    signature.set_cc(CM_CC_FASTCALL);
    funcarg_t argument;
    argument.type = pointer_to(scalar(BTF_INT32));
    signature.push_back(argument);
    tinfo_t function;
    ASSERT_TRUE(function.create_func(signature));

    func_type_data_t erased_signature;
    erased_signature.rettype = scalar(BTF_VOID);
    tinfo_t erased_function;
    ASSERT_TRUE(erased_function.create_func(erased_signature));
    ASSERT_FALSE(function.equals_to(erased_function));

    for (const auto& observed : {
            pointer_to(scalar(BTF_VOID)), pointer_to(erased_function)}) {
        AccessPattern pattern;
        pattern.original_type = pointer_to(pointer_to(function));
        pattern.accesses.push_back(load(observed));
        TypeConfidence confidence;
        const auto inferred = detail::infer_base_type_from_accesses(pattern, confidence);
        EXPECT_TRUE(inferred.equals_to(pattern.original_type));
        func_type_data_t preserved;
        ASSERT_TRUE(inferred.get_pointed_object().get_pointed_object()
            .get_func_details(&preserved));
        EXPECT_EQ(preserved.get_cc(), signature.get_cc());
        EXPECT_TRUE(preserved.rettype.equals_to(signature.rettype));
        ASSERT_EQ(preserved.size(), 1U);
        EXPECT_TRUE(preserved[0].type.equals_to(signature[0].type));
    }
}

TEST(AccessBaseTypeInferenceTest, SameWidthScalarViewDoesNotEraseAggregate) {
    for (bool is_union : {false, true}) {
        udt_type_data_t fields;
        fields.is_union = is_union;
        fields.total_size = 4;
        udm_t field;
        field.name = "value";
        field.size = 32;
        field.type = scalar(BTF_INT32);
        fields.push_back(field);
        tinfo_t aggregate;
        ASSERT_TRUE(aggregate.create_udt(fields, is_union ? BTF_UNION : BTF_STRUCT));
        AccessPattern pattern;
        pattern.original_type = pointer_to(aggregate);
        pattern.accesses.push_back(load(scalar(BTF_INT32)));
        TypeConfidence confidence;
        EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
            .equals_to(pattern.original_type));
    }
}

TEST(AccessBaseTypeInferenceTest, SameWidthScalarViewDoesNotEraseArray) {
    tinfo_t array;
    array.create_array(scalar(BTF_INT32), 1);
    AccessPattern pattern;
    pattern.original_type = pointer_to(array);
    pattern.accesses.push_back(load(scalar(BTF_INT32)));
    TypeConfidence confidence;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence)
        .equals_to(pattern.original_type));
}

TEST(AccessBaseTypeInferenceTest, NestedAndVtableAccessesDoNotIdentifyDirectPointee) {
    for (bool vtable : {false, true}) {
        AccessPattern pattern;
        auto access = load(pointer_to(scalar(BTF_VOID)));
        access.is_vtable_access = vtable;
        access.base_indirection = vtable ? 0 : 2;
        pattern.accesses.push_back(access);
        TypeConfidence confidence;
        expect_void_pointer(detail::infer_base_type_from_accesses(pattern, confidence));
    }
}

TEST(AccessBaseTypeInferenceTest, StridedFieldsDoNotBecomeScalarPointerArithmetic) {
    AccessPattern pattern;
    auto access = load(scalar(BTF_INT32));
    access.array_stride_hint = 16;
    pattern.accesses.push_back(access);
    TypeConfidence confidence;
    expect_void_pointer(detail::infer_base_type_from_accesses(pattern, confidence));
}

TEST(AccessBaseTypeInferenceTest, InvalidOrMismatchedWidthsDoNotInventPointee) {
    AccessPattern pattern;
    auto access = load(scalar(BTF_INT32));
    access.size = 2;
    pattern.accesses.push_back(access);
    TypeConfidence confidence;
    expect_void_pointer(detail::infer_base_type_from_accesses(pattern, confidence));

    pattern.accesses.front().size = 0;
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence).empty());
    pattern.accesses.clear();
    EXPECT_TRUE(detail::infer_base_type_from_accesses(pattern, confidence).empty());
}

TEST(FunctionVariableKeyTest, UpperAddressBitsArePartOfCallerIdentity) {
    std::unordered_map<detail::FunctionVariableKey, int,
        detail::FunctionVariableKeyHash> cache;
    cache[{0x0000000100001000ULL, 3}] = 11;
    cache[{0x0000000200001000ULL, 3}] = 22;
    cache[{0x0000000100001000ULL, 4}] = 33;

    ASSERT_EQ(cache.size(), 3U);
    EXPECT_EQ(cache.at({0x0000000100001000ULL, 3}), 11);
    EXPECT_EQ(cache.at({0x0000000200001000ULL, 3}), 22);
    EXPECT_EQ(cache.at({0x0000000100001000ULL, 4}), 33);
}

TEST(FunctionVariableKeyTest, HashCollisionsDoNotMergeVariableTypes) {
    struct CollidingHash {
        std::size_t operator()(const detail::FunctionVariableKey&) const noexcept {
            return 0;
        }
    };
    const detail::FunctionVariableKey first{0x1000, 2};
    const detail::FunctionVariableKey second{0x1004, 0};
    std::unordered_map<detail::FunctionVariableKey, int,
        CollidingHash> cache;
    cache[first] = 11;
    cache[second] = 22;

    ASSERT_EQ(cache.size(), 2U);
    EXPECT_EQ(cache.at(first), 11);
    EXPECT_EQ(cache.at(second), 22);
}

} // namespace structor::test
