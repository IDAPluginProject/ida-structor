#pragma once

#if defined(STRUCTOR_LIVE_TEST_HOOKS)

#include <structor/type_matcher.hpp>

#include <utility>
#include <vector>

namespace structor::detail {

// These checks use IDA's real anonymous tinfo_t implementation. They do not
// create named/numbered types or apply types to database addresses.
inline std::vector<std::pair<const char*, bool>> run_live_existing_type_checks() {
    std::vector<std::pair<const char*, bool>> checks;
    const ExistingTypeMatcher matcher;
    const auto simple_type = [](type_t kind) {
        tinfo_t type;
        type.create_simple_type(kind);
        return type;
    };
    const auto integer = simple_type(BTF_INT32);
    const auto floating = simple_type(BTF_FLOAT);
    const auto field = [&](sval_t offset) {
        SynthField result;
        result.offset = offset;
        result.size = 4;
        result.type = integer;
        result.semantic = SemanticType::Integer;
        result.name.sprnt("field_%X", static_cast<unsigned>(offset));
        result.naming.origin = NameOrigin::GeneratedFallback;
        return result;
    };
    const auto existing = [&](sval_t offset, const char* name) {
        ExistingTypeField result;
        result.offset = offset;
        result.size = 4;
        result.type = integer;
        result.name = name;
        return result;
    };
    const auto candidate = [](const ExistingTypeField& member) {
        TypeOverlapCandidate result;
        // merge_existing_type treats this only as a non-BADADDR identity; no
        // named type lookup or write is performed by the merge API.
        result.tid = 0;
        result.name = "anonymous_matcher_test";
        result.fields.push_back(member);
        return result;
    };

    {
        SynthStruct structure;
        structure.size = 16;
        structure.fields.push_back(field(8));
        structure.fields.push_back(field(0));
        structure.fields[0].comment = "observed read";
        FieldAccess evidence;
        evidence.offset = 8;
        evidence.size = 4;
        evidence.access_type = AccessType::Read;
        evidence.inferred_type = integer;
        structure.fields[0].source_accesses.push_back(evidence);
        const auto original = structure;
        const auto result = matcher.merge_existing_type(
            structure, candidate(existing(6, "overlap")));
        bool preserved = result.success && result.fields_skipped == 1 &&
            result.fields_added == 0 && result.fields_retyped == 0 &&
            result.fields_renamed == 0 && structure.size == original.size &&
            structure.fields.size() == original.fields.size();
        for (size_t i = 0; preserved && i < structure.fields.size(); ++i) {
            const auto& actual = structure.fields[i];
            const auto& before = original.fields[i];
            preserved = actual.offset == before.offset && actual.size == before.size &&
                actual.name == before.name && actual.type.equals_to(before.type) &&
                actual.comment == before.comment &&
                actual.source_accesses.size() == before.source_accesses.size();
            if (preserved && !actual.source_accesses.empty()) {
                preserved = actual.source_accesses[0].offset == 8 &&
                    actual.source_accesses[0].size == 4 &&
                    actual.source_accesses[0].inferred_type.equals_to(integer);
            }
        }
        checks.emplace_back("rejected_overlap_preserves_state", preserved);
    }

    {
        SynthStruct structure;
        structure.size = 20;
        structure.fields.push_back(SynthField::create_padding(0, 16));
        structure.fields.push_back(field(16));
        const auto result = matcher.merge_existing_type(
            structure, candidate(existing(4, "middle")));
        checks.emplace_back("padding_split_preserves_extent",
            result.success && result.fields_added == 1 && structure.size == 20 &&
            structure.fields.size() == 4 &&
            structure.fields[0].is_padding && structure.fields[0].offset == 0 &&
            structure.fields[0].size == 4 && structure.fields[0].type.get_size() == 4 &&
            structure.fields[1].name == "middle" && structure.fields[1].offset == 4 &&
            structure.fields[2].is_padding && structure.fields[2].offset == 8 &&
            structure.fields[2].size == 8 && structure.fields[2].type.get_size() == 8 &&
            structure.fields[3].offset == 16 && structure.fields[3].type.equals_to(integer));
    }

    {
        SynthStruct structure;
        structure.size = 16;
        structure.fields.push_back(field(0));
        structure.fields.push_back(SynthField::create_padding(4, 12));
        auto member = existing(0, "values");
        member.type.create_array(integer, 3);
        member.size = 12;
        const auto result = matcher.merge_existing_type(structure, candidate(member));
        checks.emplace_back("array_expansion_updates_metadata",
            result.success && result.fields_retyped == 1 && structure.fields.size() == 2 &&
            structure.fields[0].size == 12 && structure.fields[0].type.get_size() == 12 &&
            structure.fields[0].is_array && structure.fields[0].array_count == 3 &&
            structure.fields[1].is_padding && structure.fields[1].offset == 12 &&
            structure.fields[1].size == 4);

        // A scalar of the same storage width replaces array metadata completely.
        SynthStruct scalar;
        scalar.size = 8;
        scalar.fields.push_back(SynthField::create_array(0, integer, 2));
        member.type = simple_type(BTF_INT64);
        member.size = 8;
        const auto scalar_result = matcher.merge_existing_type(scalar, candidate(member));
        checks.emplace_back("scalar_replacement_clears_array_metadata",
            scalar_result.success && scalar_result.fields_retyped == 1 &&
            scalar.fields.size() == 1 && !scalar.fields[0].is_array &&
            scalar.fields[0].array_count == 1 && scalar.fields[0].type.get_size() == 8);
    }

    {
        SynthStruct structure;
        structure.size = 4;
        structure.fields.push_back(field(0));
        auto member = existing(8, "wrong_width");
        member.size = 8;
        const auto result = matcher.merge_existing_type(structure, candidate(member));
        checks.emplace_back("type_size_mismatch_is_rejected",
            result.success && result.fields_skipped == 1 && result.fields_added == 0 &&
            structure.size == 4 && structure.fields.size() == 1 &&
            structure.fields[0].type.equals_to(integer));
    }

    {
        SynthStruct structure;
        structure.size = 12;
        structure.fields.push_back(field(8));
        structure.fields[0].name = "analyst_label";
        structure.fields[0].naming.origin = NameOrigin::UserProvided;
        structure.fields[0].naming.locked = true;
        const auto result = matcher.merge_existing_type(
            structure, candidate(existing(0, "analyst_label")));
        checks.emplace_back("imported_names_preserve_analyst_names",
            result.success && result.fields_added == 1 && structure.fields.size() == 2 &&
            structure.fields[0].name != "analyst_label" &&
            structure.fields[1].name == "analyst_label" &&
            structure.fields[1].naming.locked);
    }

    {
        SynthStruct structure;
        structure.size = 4;
        structure.fields.push_back(SynthField::create_bitfield(0, 4, 0, 3));
        const auto result = matcher.merge_existing_type(
            structure, candidate(existing(0, "replacement")));
        checks.emplace_back("observed_bitfield_is_preserved",
            result.success && result.fields_skipped == 1 &&
            structure.fields.size() == 1 && structure.fields[0].is_bitfield &&
            structure.fields[0].bit_size == 3);

        tinfo_t bit_type;
        const bool created = bit_type.create_bitfield(4, 3, true);
        auto member = existing(0, "bits");
        member.type = bit_type;
        member.is_bitfield = bit_type.is_decl_bitfield();
        structure.fields[0] = field(0);
        const auto imported = matcher.merge_existing_type(structure, candidate(member));
        checks.emplace_back("existing_bitfield_is_not_byte_overlay",
            created && member.is_bitfield && imported.success &&
            imported.fields_skipped == 1 && !structure.fields[0].is_bitfield &&
            structure.fields[0].type.equals_to(integer));
    }

    {
        const auto aggregate = [](const tinfo_t& member_type) {
            udt_type_data_t udt;
            udt.total_size = 4;
            udm_t member;
            member.name = "value";
            member.offset = 0;
            member.size = 32;
            member.type = member_type;
            udt.push_back(member);
            tinfo_t type;
            type.create_udt(udt, BTF_STRUCT);
            return type;
        };
        const auto int_struct = aggregate(integer);
        const auto float_struct = aggregate(floating);
        checks.emplace_back("distinct_udt_members_are_not_equivalent",
            int_struct.is_struct() && float_struct.is_struct() &&
            int_struct.get_size() == 4 && float_struct.get_size() == 4 &&
            !ExistingTypeMatcher::types_compatible(int_struct, float_struct) &&
            ExistingTypeMatcher::types_compatible(int_struct, int_struct));

        tinfo_t int_pointer;
        tinfo_t float_pointer;
        int_pointer.create_ptr(integer);
        float_pointer.create_ptr(floating);
        checks.emplace_back("distinct_pointees_are_not_equivalent",
            !ExistingTypeMatcher::types_compatible(int_pointer, float_pointer) &&
            ExistingTypeMatcher::types_compatible(int_pointer, int_pointer));

        const auto callback = [](const tinfo_t& return_type) {
            func_type_data_t details;
            details.rettype = return_type;
            details.set_cc(CM_CC_CDECL);
            tinfo_t function;
            function.create_func(details);
            tinfo_t pointer;
            pointer.create_ptr(function);
            return pointer;
        };
        const auto int_callback = callback(integer);
        const auto float_callback = callback(floating);
        checks.emplace_back("distinct_callback_signatures_are_not_equivalent",
            int_callback.is_funcptr() && float_callback.is_funcptr() &&
            !ExistingTypeMatcher::types_compatible(int_callback, float_callback) &&
            ExistingTypeMatcher::types_compatible(int_callback, int_callback));
    }
    return checks;
}

} // namespace structor::detail

#endif
