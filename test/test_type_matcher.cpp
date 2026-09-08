#include <gtest/gtest.h>
#include "mock_ida.hpp"
#include <structor/type_matcher.hpp>

namespace structor::test {
namespace {

tinfo_t integer_type(std::uint32_t kind = BTF_INT32) {
    tinfo_t type;
    type.create_simple_type(kind);
    return type;
}

SynthField field_at(sval_t offset, std::uint32_t kind = BTF_INT32) {
    SynthField field;
    field.offset = offset;
    field.type = integer_type(kind);
    field.size = static_cast<std::uint32_t>(field.type.get_size());
    field.name.sprnt("field_%X", static_cast<unsigned>(offset));
    field.semantic = SemanticType::Integer;
    field.naming.origin = NameOrigin::GeneratedFallback;
    return field;
}

ExistingTypeField existing_at(sval_t offset, const char* name,
                            std::uint32_t kind = BTF_INT32) {
    ExistingTypeField field;
    field.offset = offset;
    field.name = name;
    field.type = integer_type(kind);
    field.size = static_cast<std::uint32_t>(field.type.get_size());
    return field;
}

TypeOverlapCandidate candidate_with(const ExistingTypeField& field) {
    TypeOverlapCandidate candidate;
    candidate.tid = 42;
    candidate.name = "Existing";
    candidate.fields.push_back(field);
    return candidate;
}

} // namespace

TEST(TypeMatcherTest, RejectedPartialOverlapPreservesAllFieldState) {
    SynthStruct structure;
    structure.size = 16;
    structure.fields = {field_at(0), field_at(8)};
    structure.fields[1].comment = "observed read";
    FieldAccess access;
    access.offset = 8;
    access.size = 4;
    structure.fields[1].source_accesses.push_back(access);
    const auto original = structure;

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(6, "conflict")));

    EXPECT_EQ(result.fields_skipped, 1U);
    EXPECT_EQ(result.fields_added, 0U);
    ASSERT_EQ(structure.fields.size(), original.fields.size());
    for (std::size_t i = 0; i < structure.fields.size(); ++i) {
        EXPECT_EQ(structure.fields[i].name, original.fields[i].name);
        EXPECT_TRUE(structure.fields[i].type.equals_to(original.fields[i].type));
        EXPECT_EQ(structure.fields[i].comment, original.fields[i].comment);
        EXPECT_EQ(structure.fields[i].source_accesses.size(),
                  original.fields[i].source_accesses.size());
    }
    EXPECT_EQ(structure.size, original.size);
}

TEST(TypeMatcherTest, RejectedExpansionDoesNotRenameOrGrowStructure) {
    SynthStruct structure;
    structure.size = 8;
    structure.fields = {field_at(0), field_at(4)};
    auto wide = existing_at(0, "conflict", BTF_INT64);
    wide.size = 16;

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(wide));

    EXPECT_EQ(result.fields_skipped, 1U);
    EXPECT_EQ(result.fields_renamed, 0U);
    EXPECT_EQ(result.fields_retyped, 0U);
    EXPECT_EQ(structure.size, 8U);
    EXPECT_EQ(structure.fields[0].name, qstring("field_0"));
    EXPECT_EQ(structure.fields[0].size, 4U);
}

TEST(TypeMatcherTest, NarrowerExistingFieldCannotEraseObservedExtent) {
    SynthStruct structure;
    structure.size = 8;
    structure.fields = {field_at(0, BTF_INT64)};

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(0, "narrow")));

    EXPECT_EQ(result.fields_skipped, 1U);
    EXPECT_EQ(result.fields_renamed, 0U);
    EXPECT_EQ(structure.fields[0].size, 8U);
    EXPECT_TRUE(structure.fields[0].type.equals_to(integer_type(BTF_INT64)));
}

TEST(TypeMatcherTest, FieldInsertionSplitsPaddingAndPreservesOtherFields) {
    SynthStruct structure;
    structure.size = 20;
    structure.fields = {SynthField::create_padding(0, 16), field_at(16)};

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(4, "middle")));

    EXPECT_EQ(result.fields_added, 1U);
    ASSERT_EQ(structure.fields.size(), 4U);
    EXPECT_TRUE(structure.fields[0].is_padding);
    EXPECT_EQ(structure.fields[0].offset, 0);
    EXPECT_EQ(structure.fields[0].size, 4U);
    EXPECT_EQ(structure.fields[1].name, qstring("middle"));
    EXPECT_TRUE(structure.fields[2].is_padding);
    EXPECT_EQ(structure.fields[2].offset, 8);
    EXPECT_EQ(structure.fields[2].size, 8U);
    EXPECT_EQ(structure.fields[3].name, qstring("field_10"));
    EXPECT_FALSE(structure.fields[3].type.empty());
}

TEST(TypeMatcherTest, ExactExpansionTrimsPaddingAndMaintainsArrayMetadata) {
    SynthStruct structure;
    structure.size = 16;
    structure.fields = {field_at(0), SynthField::create_padding(4, 12)};
    auto existing = existing_at(0, "values");
    existing.type.create_array(integer_type(), 3);
    existing.size = 12;

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing));

    EXPECT_EQ(result.fields_retyped, 1U);
    EXPECT_EQ(result.fields_renamed, 1U);
    ASSERT_EQ(structure.fields.size(), 2U);
    EXPECT_TRUE(structure.fields[0].is_array);
    EXPECT_EQ(structure.fields[0].array_count, 3U);
    EXPECT_EQ(structure.fields[0].size, 12U);
    EXPECT_EQ(structure.fields[1].offset, 12);
    EXPECT_EQ(structure.fields[1].size, 4U);
}

TEST(TypeMatcherTest, RetypingArrayAsScalarClearsArrayMetadata) {
    SynthStruct structure;
    structure.size = 8;
    structure.fields = {SynthField::create_array(0, integer_type(), 2)};

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(0, "word", BTF_INT64)));

    EXPECT_EQ(result.fields_retyped, 1U);
    EXPECT_FALSE(structure.fields[0].is_array);
    EXPECT_EQ(structure.fields[0].array_count, 1U);
}

TEST(TypeMatcherTest, ScalarDoesNotReplaceUnionAlternativesOrBitfields) {
    for (bool bitfield : {false, true}) {
        SynthStruct structure;
        structure.size = 4;
        structure.fields = {field_at(0)};
        if (bitfield) {
            structure.fields[0].is_bitfield = true;
            structure.fields[0].bit_size = 3;
        } else {
            structure.fields[0].is_union_candidate = true;
            SynthField::UnionMember alternative;
            alternative.name = "original";
            alternative.type = integer_type();
            alternative.size = 4;
            structure.fields[0].union_members.push_back(alternative);
        }

        const auto result = ExistingTypeMatcher{}.merge_existing_type(
            structure, candidate_with(existing_at(0, "replacement")));

        EXPECT_EQ(result.fields_skipped, 1U);
        EXPECT_EQ(result.fields_retyped, 0U);
        EXPECT_EQ(structure.fields[0].name, qstring("field_0"));
        EXPECT_EQ(structure.fields[0].is_bitfield, bitfield);
        EXPECT_EQ(structure.fields[0].union_members.size(), bitfield ? 0U : 1U);
    }
}

TEST(TypeMatcherTest, UserNamesSurviveValidRetyping) {
    SynthStruct structure;
    structure.size = 4;
    structure.fields = {field_at(0)};
    structure.fields[0].name = "analyst_label";
    structure.fields[0].naming.origin = NameOrigin::UserProvided;

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(0, "replacement", BTF_FLOAT)));

    EXPECT_EQ(result.fields_retyped, 1U);
    EXPECT_EQ(result.fields_renamed, 0U);
    EXPECT_EQ(structure.fields[0].name, qstring("analyst_label"));
    EXPECT_TRUE(structure.fields[0].type.is_floating());
}

TEST(TypeMatcherTest, InvalidTypeExtentDoesNotMutateStructure) {
    SynthStruct structure;
    structure.size = 4;
    structure.fields = {field_at(0)};
    auto invalid = existing_at(8, "invalid");
    invalid.size = 8;

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(invalid));

    EXPECT_EQ(result.fields_skipped, 1U);
    EXPECT_EQ(result.fields_added, 0U);
    EXPECT_EQ(structure.size, 4U);
    ASSERT_EQ(structure.fields.size(), 1U);
    EXPECT_FALSE(structure.fields[0].type.empty());
}

TEST(TypeMatcherTest, FieldNamesAloneDoNotEraseTypedData) {
    for (const char* name : {"alignment", "gap_count", "padding_mode"}) {
        auto field = field_at(0);
        field.name = name;
        EXPECT_FALSE(ExistingTypeMatcher::is_effective_padding(field));
    }
}

TEST(TypeMatcherTest, SameWidthCompositeTypesAreNotEquivalent) {
    tinfo_t int_pointer;
    int_pointer.create_ptr(integer_type());
    tinfo_t float_pointer;
    float_pointer.create_ptr(integer_type(BTF_FLOAT));
    EXPECT_FALSE(ExistingTypeMatcher::types_compatible(int_pointer, float_pointer));
    EXPECT_TRUE(ExistingTypeMatcher::types_compatible(int_pointer, int_pointer));

    tinfo_t int_array;
    int_array.create_array(integer_type(), 2);
    tinfo_t byte_array;
    byte_array.create_array(integer_type(BTF_UINT8), 8);
    EXPECT_FALSE(ExistingTypeMatcher::types_compatible(int_array, byte_array));
}

TEST(TypeMatcherTest, ImportedNameCannotRenameLockedFieldAtALaterOffset) {
    SynthStruct structure;
    structure.size = 12;
    structure.fields = {field_at(8)};
    structure.fields[0].name = "analyst_label";
    structure.fields[0].naming.locked = true;

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(0, "analyst_label")));

    EXPECT_EQ(result.fields_added, 1U);
    ASSERT_EQ(structure.fields.size(), 2U);
    EXPECT_EQ(structure.fields[0].name, qstring("analyst_label_1"));
    EXPECT_EQ(structure.fields[1].name, qstring("analyst_label"));
    EXPECT_TRUE(structure.fields[1].naming.locked);
}

TEST(TypeMatcherTest, AllSkippedOverlayIsAnExactNoOp) {
    SynthStruct structure;
    structure.size = 16;
    structure.fields = {field_at(8), field_at(0)};
    structure.fields[0].name = "duplicate";
    structure.fields[1].name = "duplicate";

    const auto result = ExistingTypeMatcher{}.merge_existing_type(
        structure, candidate_with(existing_at(6, "conflict")));

    EXPECT_EQ(result.fields_skipped, 1U);
    ASSERT_EQ(structure.fields.size(), 2U);
    EXPECT_EQ(structure.fields[0].offset, 8);
    EXPECT_EQ(structure.fields[1].offset, 0);
    EXPECT_EQ(structure.fields[0].name, qstring("duplicate"));
    EXPECT_EQ(structure.fields[1].name, qstring("duplicate"));
}

} // namespace structor::test
