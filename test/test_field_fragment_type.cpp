#include <gtest/gtest.h>

#include "mock_ida.hpp"
#include <structor/field_fragment_type.hpp>

namespace structor::test {
namespace {

tinfo_t fragment_scalar(std::uint32_t flags) {
    tinfo_t type;
    type.create_simple_type(flags);
    return type;
}

SynthField selected_array(const tinfo_t& element, sval_t offset = 0x18,
                          std::uint32_t count = 4) {
    SynthField source;
    source.offset = offset;
    source.type.create_array(element, count);
    source.size = static_cast<std::uint32_t>(source.type.get_size());
    source.is_array = true;
    source.array_count = count;
    source.semantic = SemanticType::Array;
    return source;
}

SynthField observed_fragment(sval_t offset = 0x30, std::uint32_t size = 8) {
    SynthField fragment;
    fragment.offset = offset;
    fragment.size = size;
    fragment.type = fragment_scalar(BT_UNK_QWORD);
    fragment.semantic = SemanticType::Unknown;
    fragment.name = "tail_evidence";
    fragment.comment = "observed in parent";
    fragment.confidence = TypeConfidence::High;
    FieldAccess evidence;
    evidence.offset = offset;
    evidence.size = size;
    evidence.insn_ea = 0x1010;
    evidence.source_func_ea = 0x1000;
    fragment.source_accesses.push_back(evidence);
    return fragment;
}

} // namespace

TEST(FieldFragmentTypeTest, ConstructorTailRetainsSelectedUnsignedElement) {
    // A selected uint64[4] at [0x18, 0x38) crosses a nested object ending
    // at 0x30. Rebuilding the remaining load must retain its solved type.
    const auto element = fragment_scalar(BTF_UINT64);
    const auto source = selected_array(element);
    auto fragment = observed_fragment();

    ASSERT_TRUE(detail::retain_array_fragment_type(source, fragment));
    EXPECT_TRUE(fragment.type.equals_to(element));
    EXPECT_EQ(fragment.semantic, SemanticType::UnsignedInteger);
    EXPECT_FALSE(fragment.is_array);
    EXPECT_EQ(fragment.array_count, 1u);
    EXPECT_EQ(fragment.offset, 0x30);
    EXPECT_EQ(fragment.size, 8u);
    EXPECT_STREQ(fragment.name.c_str(), "tail_evidence");
    EXPECT_STREQ(fragment.comment.c_str(), "observed in parent");
    EXPECT_EQ(fragment.confidence, TypeConfidence::High);
    ASSERT_EQ(fragment.source_accesses.size(), 1u);
    EXPECT_EQ(fragment.source_accesses[0].insn_ea, 0x1010u);
    EXPECT_EQ(fragment.source_accesses[0].source_func_ea, 0x1000u);
}

TEST(FieldFragmentTypeTest, CompleteMultipleElementsRetainArrayExtent) {
    const auto element = fragment_scalar(BTF_UINT64);
    const auto source = selected_array(element);
    auto fragment = observed_fragment(0x28, 16);

    ASSERT_TRUE(detail::retain_array_fragment_type(source, fragment));
    array_type_data_t array;
    ASSERT_TRUE(fragment.type.get_array_details(&array));
    EXPECT_TRUE(array.elem_type.equals_to(element));
    EXPECT_EQ(array.nelems, 2u);
    EXPECT_EQ(fragment.type.get_size(), 16u);
    EXPECT_EQ(fragment.semantic, SemanticType::Array);
    EXPECT_TRUE(fragment.is_array);
    EXPECT_EQ(fragment.array_count, 2u);
}

TEST(FieldFragmentTypeTest, CallbackElementRetainsSignatureAndClassification) {
    func_type_data_t signature;
    signature.rettype = fragment_scalar(BTF_INT32);
    tinfo_t function;
    ASSERT_TRUE(function.create_func(signature));
    tinfo_t callback;
    callback.create_ptr(function);
    const auto source = selected_array(callback);
    auto fragment = observed_fragment();

    ASSERT_TRUE(detail::retain_array_fragment_type(source, fragment));
    EXPECT_TRUE(fragment.type.equals_to(callback));
    EXPECT_EQ(fragment.semantic, SemanticType::FunctionPointer);
}

TEST(FieldFragmentTypeTest, PartialMisalignedAndOutsideFragmentsRemainObserved) {
    const auto source = selected_array(fragment_scalar(BTF_UINT64));
    const std::pair<sval_t, std::uint32_t> rejected[] = {
        {0x30, 4}, {0x2C, 8}, {0x10, 8}, {0x38, 8}, {0x30, 16}, {0x30, 0},
        {std::numeric_limits<sval_t>::max() - 3, 8},
    };
    for (const auto& [offset, size] : rejected) {
        SCOPED_TRACE(offset);
        auto fragment = observed_fragment(offset, size);
        const auto original_type = fragment.type;
        EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));
        EXPECT_TRUE(fragment.type.equals_to(original_type));
        EXPECT_EQ(fragment.semantic, SemanticType::Unknown);
        EXPECT_FALSE(fragment.is_array);
        EXPECT_EQ(fragment.array_count, 1u);
        EXPECT_EQ(fragment.offset, offset);
        EXPECT_EQ(fragment.size, size);
        EXPECT_EQ(fragment.confidence, TypeConfidence::High);
    }
}

TEST(FieldFragmentTypeTest, InvalidSourceExtentCannotSupplyElementType) {
    auto source = selected_array(fragment_scalar(BTF_UINT64));
    source.size = 24;
    auto fragment = observed_fragment(0x20, 8);
    EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));

    source = selected_array(fragment_scalar(BTF_UINT64),
                            std::numeric_limits<sval_t>::max() - 15);
    fragment = observed_fragment(source.offset, 8);
    EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));

    source.type = fragment_scalar(BTF_UINT64);
    source.offset = fragment.offset = 0;
    source.size = 8;
    EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));
}

TEST(FieldFragmentTypeTest, ElementArrayRetainsInnerDimension) {
    tinfo_t inner_array;
    inner_array.create_array(fragment_scalar(BTF_INT32), 2);
    const auto source = selected_array(inner_array);
    auto fragment = observed_fragment();

    ASSERT_TRUE(detail::retain_array_fragment_type(source, fragment));
    EXPECT_TRUE(fragment.type.equals_to(inner_array));
    EXPECT_EQ(fragment.semantic, SemanticType::Array);
    EXPECT_TRUE(fragment.is_array);
    EXPECT_EQ(fragment.array_count, 2u);
}

TEST(FieldFragmentTypeTest, GenericByteArrayCannotEraseKnownResidualTypes) {
    const auto source = selected_array(fragment_scalar(BTF_UINT8), 0, 32);
    tinfo_t pointer;
    pointer.create_ptr(fragment_scalar(BTF_INT32));
    func_type_data_t signature;
    signature.rettype = fragment_scalar(BTF_INT32);
    tinfo_t function;
    ASSERT_TRUE(function.create_func(signature));
    tinfo_t callback;
    callback.create_ptr(function);
    udt_type_data_t members;
    udm_t member;
    member.name = "value";
    member.offset = 0;
    member.size = 64;
    member.type = fragment_scalar(BTF_UINT64);
    members.push_back(member);
    tinfo_t structure;
    ASSERT_TRUE(structure.create_udt(members, BTF_STRUCT));
    tinfo_t array;
    array.create_array(fragment_scalar(BTF_INT32), 2);
    tinfo_t callback_array;
    callback_array.create_array(callback, 2);
    const tinfo_t known_types[] = {
        pointer, callback, structure, array, callback_array,
        fragment_scalar(BTF_UINT64), fragment_scalar(BTF_DOUBLE),
    };

    for (const auto& known_type : known_types) {
        auto fragment = observed_fragment(0,
            static_cast<std::uint32_t>(known_type.get_size()));
        fragment.type = known_type;
        const auto original_semantic = fragment.semantic;
        EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));
        EXPECT_TRUE(fragment.type.equals_to(known_type));
        EXPECT_EQ(fragment.semantic, original_semantic);
        EXPECT_EQ(fragment.confidence, TypeConfidence::High);
        ASSERT_EQ(fragment.source_accesses.size(), 1u);
        EXPECT_EQ(fragment.source_accesses[0].insn_ea, 0x1010u);
    }
}

TEST(FieldFragmentTypeTest, SolvedArrayCannotChangeKnownScalarInterpretation) {
    const auto source = selected_array(fragment_scalar(BTF_UINT64));
    auto fragment = observed_fragment();
    fragment.type = fragment_scalar(BTF_INT64);
    fragment.semantic = SemanticType::Integer;
    EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));
    EXPECT_TRUE(fragment.type.is_signed());
    EXPECT_EQ(fragment.semantic, SemanticType::Integer);
}

TEST(FieldFragmentTypeTest, KnownEqualTypePreservesIndependentMetadata) {
    const auto element = fragment_scalar(BTF_UINT64);
    const auto source = selected_array(element);
    auto fragment = observed_fragment();
    fragment.type = element;
    // Existing usage classification can differ from the retained tinfo.
    // Refining partial storage must not silently normalize known fields.
    fragment.semantic = SemanticType::Integer;
    fragment.confidence = TypeConfidence::Certain;
    ASSERT_TRUE(fragment.type.equals_to(element));

    EXPECT_FALSE(detail::retain_array_fragment_type(source, fragment));
    EXPECT_TRUE(fragment.type.equals_to(element));
    EXPECT_EQ(fragment.semantic, SemanticType::Integer);
    EXPECT_EQ(fragment.confidence, TypeConfidence::Certain);
    EXPECT_STREQ(fragment.name.c_str(), "tail_evidence");
    EXPECT_STREQ(fragment.comment.c_str(), "observed in parent");
    EXPECT_FALSE(fragment.is_array);
    EXPECT_EQ(fragment.array_count, 1u);
    ASSERT_EQ(fragment.source_accesses.size(), 1u);
    EXPECT_EQ(fragment.source_accesses[0].insn_ea, 0x1010u);
}

} // namespace structor::test
