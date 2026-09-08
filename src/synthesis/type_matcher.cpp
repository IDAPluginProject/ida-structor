/// @file type_matcher.cpp
/// @brief Sparse matching of synthesized structures against existing IDA types

#ifdef STRUCTOR_TESTING
#include "mock_ida.hpp"
#endif
#include <structor/type_matcher.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>

namespace structor {

namespace {

qstring render_type_decl(const tinfo_t& type) {
    qstring decl;
    if (!type.empty()) {
        type.print(&decl);
    }
    return decl;
}

std::uint32_t member_size_bytes(const udm_t& member) {
    if (member.size != 0) {
        const auto bytes = member.size / 8 + (member.size % 8 != 0);
        return bytes <= MAX_STRUCT_SIZE ? static_cast<std::uint32_t>(bytes) : 0;
    }

    const size_t type_size = member.type.get_size();
    if (type_size != BADSIZE && type_size > 0 && type_size <= MAX_STRUCT_SIZE) {
        return static_cast<std::uint32_t>(type_size);
    }

    return 0;
}

bool extract_existing_fields(const tinfo_t& type, qvector<ExistingTypeField>& out) {
    udt_type_data_t udt;
    if (!type.get_udt_details(&udt)) {
        return false;
    }

    for (const auto& member : udt) {
        ExistingTypeField field;
        field.name = member.name;
        field.offset = static_cast<sval_t>(member.offset / 8);
        field.size = member_size_bytes(member);
        field.type = member.type;
        field.type_decl = render_type_decl(member.type);
        field.is_bitfield = member.is_bitfield() || (member.offset % 8) != 0;
        // A human field called alignment or gap_count is still data. Only
        // conventionally named byte arrays are candidates for padding reuse.
        array_type_data_t array;
        field.is_padding = !field.is_bitfield &&
            ExistingTypeMatcher::is_padding_name(field.name) &&
            field.type.get_array_details(&array) && array.elem_type.get_size() == 1;
        out.push_back(std::move(field));
    }

    return true;
}

std::uint32_t count_synth_fields(const SynthStruct& synth_struct) {
    std::uint32_t count = 0;
    for (const auto& field : synth_struct.fields) {
        if (!ExistingTypeMatcher::is_effective_padding(field) && field.size != 0) {
            ++count;
        }
    }
    return count;
}

std::uint32_t count_existing_fields(const qvector<ExistingTypeField>& fields) {
    std::uint32_t count = 0;
    for (const auto& field : fields) {
        if (!field.is_padding && field.size != 0) {
            ++count;
        }
    }
    return count;
}

bool existing_field_less(const ExistingTypeField& a, const ExistingTypeField& b) {
    if (a.offset != b.offset) {
        return a.offset < b.offset;
    }
    if (a.size != b.size) {
        return a.size < b.size;
    }
    return std::string(a.name.c_str()) < std::string(b.name.c_str());
}

void uniquify_field_names(qvector<SynthField>& fields) {
    std::unordered_set<std::string> seen;
    std::unordered_map<std::string, unsigned> next_suffix;
    for (const auto& field : fields) {
        if (!field.name.empty() &&
            (field.naming.locked || field.naming.origin == NameOrigin::UserProvided)) {
            seen.insert(field.name.c_str());
        }
    }
    for (auto& field : fields) {
        if (!field.name.empty() &&
            (field.naming.locked || field.naming.origin == NameOrigin::UserProvided)) {
            continue;
        }
        if (field.name.empty()) {
            field.name.sprnt("field_%X", static_cast<unsigned>(field.offset));
        }

        std::string base(field.name.c_str());
        std::string candidate = base;
        const auto cursor = next_suffix.try_emplace(base, 1).first;
        while (seen.find(candidate) != seen.end()) {
            candidate = base + "_" + std::to_string(cursor->second++);
        }

        if (candidate != base) {
            field.name = candidate.c_str();
        }
        seen.insert(candidate);
    }
}

} // namespace

bool ExistingTypeMatcher::ranges_overlap(
    sval_t a_offset,
    std::uint32_t a_size,
    sval_t b_offset,
    std::uint32_t b_size) noexcept
{
    if (a_size == 0 || b_size == 0) {
        return false;
    }

    const auto a_end = checked_interval_end(a_offset, a_size);
    const auto b_end = checked_interval_end(b_offset, b_size);
    return a_end.has_value() && b_end.has_value() &&
        a_offset < *b_end && b_offset < *a_end;
}

bool ExistingTypeMatcher::is_padding_name(const qstring& name) {
    if (name.empty()) {
        return false;
    }

    std::string lower(name.c_str());
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    for (const char* prefix : {"__pad_", "_pad_", "pad_", "padding_", "gap_", "align_"}) {
        const std::string start(prefix);
        if (lower.rfind(start, 0) != 0) {
            continue;
        }
        std::string suffix = lower.substr(start.size());
        if (suffix.rfind("0x", 0) == 0) {
            suffix.erase(0, 2);
        }
        return !suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(), [](unsigned char ch) {
                return std::isxdigit(ch) != 0;
            });
    }
    return false;
}

bool ExistingTypeMatcher::is_effective_padding(const SynthField& field) {
    return field.is_padding || field.semantic == SemanticType::Padding;
}

SemanticType ExistingTypeMatcher::semantic_from_type(const tinfo_t& type) {
    if (type.empty()) {
        return SemanticType::Unknown;
    }
    if (type.is_func() || type.is_funcptr()) {
        return SemanticType::FunctionPointer;
    }
    if (type.is_ptr()) {
        tinfo_t pointed = type.get_pointed_object();
        if (!pointed.empty() && pointed.is_func()) {
            return SemanticType::FunctionPointer;
        }
        return SemanticType::Pointer;
    }
    if (type.is_struct() || type.is_union()) {
        return SemanticType::NestedStruct;
    }
    if (type.is_array()) {
        return SemanticType::Array;
    }
    if (type.is_floating()) {
        return type.get_size() == 8 ? SemanticType::Double : SemanticType::Float;
    }
    if (!type.is_signed()) {
        return SemanticType::UnsignedInteger;
    }
    return SemanticType::Integer;
}

bool ExistingTypeMatcher::types_compatible(const tinfo_t& a, const tinfo_t& b) {
    if (a.empty() || b.empty()) {
        return false;
    }

    try {
        if (a.compare_with(b, TCMP_IGNMODS | TCMP_DECL)) {
            return true;
        }
    } catch (...) {
    }

    // Equal pointer widths, aggregate sizes, or array extents do not imply
    // equivalent pointees, members, element types, or function signatures.
    if (a.is_ptr() || b.is_ptr() || a.is_func() || b.is_func() ||
        a.is_array() || b.is_array() || a.is_struct() || b.is_struct() ||
        a.is_union() || b.is_union()) {
        return false;
    }

    const size_t a_size = a.get_size();
    const size_t b_size = b.get_size();
    return a_size != BADSIZE && a_size == b_size && semantic_from_type(a) == semantic_from_type(b);
}

bool ExistingTypeMatcher::field_name_can_be_reused(const SynthField& field) {
    if (field.name.empty()) {
        return true;
    }

    if (field.naming.locked || field.naming.origin == NameOrigin::UserProvided) {
        return false;
    }

    return is_generated_name(field.name, &field.naming) ||
           field.naming.origin == NameOrigin::GeneratedFallback ||
           field.naming.origin == NameOrigin::HeuristicRole;
}

qvector<TypeOverlapCandidate> ExistingTypeMatcher::find_matches(
    const SynthStruct& synth_struct,
    std::size_t max_results,
    double min_score) const
{
    qvector<TypeOverlapCandidate> result;
    const std::uint32_t synth_count = count_synth_fields(synth_struct);
    if (synth_count == 0) {
        return result;
    }

#ifdef STRUCTOR_TESTING
    (void)max_results;
    (void)min_score;
    return result;
#else
    til_t* til = get_idati();
    if (!til) {
        return result;
    }

    const uint32_t limit = get_ordinal_limit(til);
    for (uint32_t ord = 1; ord < limit; ++ord) {
        tinfo_t type;
        if (!type.get_numbered_type(til, ord) || !type.is_struct()) {
            continue;
        }

        qvector<ExistingTypeField> existing_fields;
        if (!extract_existing_fields(type, existing_fields)) {
            continue;
        }

        const std::uint32_t existing_count = count_existing_fields(existing_fields);
        if (existing_count == 0) {
            continue;
        }

        TypeOverlapCandidate candidate;
        candidate.tid = type.get_tid();
        const char* type_name = get_numbered_type_name(til, ord);
        if (!type_name || type_name[0] == '\0' || candidate.tid == BADADDR) {
            continue;
        }
        candidate.name = type_name;

        const size_t type_size = type.get_size();
        candidate.size = type_size != BADSIZE ? static_cast<std::uint32_t>(type_size) : 0;
        candidate.synth_field_count = synth_count;
        candidate.existing_field_count = existing_count;
        candidate.fields = std::move(existing_fields);

        std::unordered_set<std::size_t> matched_synth_indexes;
        for (const auto& existing : candidate.fields) {
            if (existing.is_padding || existing.size == 0) {
                continue;
            }

            bool existing_matched = false;
            bool exact_matched = false;
            bool type_matched = false;
            for (std::size_t i = 0; i < synth_struct.fields.size(); ++i) {
                const SynthField& synth = synth_struct.fields[i];
                if (is_effective_padding(synth) || synth.size == 0) {
                    continue;
                }

                if (!ranges_overlap(synth.offset, synth.size, existing.offset, existing.size)) {
                    continue;
                }

                existing_matched = true;
                matched_synth_indexes.insert(i);

                if (synth.offset == existing.offset) {
                    exact_matched = true;
                    if (synth.size == existing.size &&
                        types_compatible(synth.type, existing.type)) {
                        type_matched = true;
                    }
                }
            }

            if (existing_matched) {
                ++candidate.matched_existing_fields;
            }
            candidate.exact_offset_matches += exact_matched;
            candidate.type_matches += type_matched;
        }

        candidate.matched_synth_fields = static_cast<std::uint32_t>(matched_synth_indexes.size());
        if (candidate.matched_existing_fields == 0) {
            continue;
        }

        const double existing_coverage = static_cast<double>(candidate.matched_existing_fields) /
                                         static_cast<double>(candidate.existing_field_count);
        const double synth_coverage = static_cast<double>(candidate.matched_synth_fields) /
                                      static_cast<double>(candidate.synth_field_count);
        const double exact_ratio = static_cast<double>(candidate.exact_offset_matches) /
                                   static_cast<double>(candidate.matched_existing_fields);
        const double type_ratio = candidate.exact_offset_matches == 0
            ? 0.0
            : static_cast<double>(candidate.type_matches) /
              static_cast<double>(candidate.exact_offset_matches);

        candidate.score = (0.55 * existing_coverage) +
                          (0.25 * synth_coverage) +
                          (0.12 * exact_ratio) +
                          (0.08 * type_ratio);

        if (candidate.score < min_score) {
            continue;
        }

        candidate.summary.sprnt("%u/%u existing, %u/%u synthesized, %u exact, %u type",
                                candidate.matched_existing_fields,
                                candidate.existing_field_count,
                                candidate.matched_synth_fields,
                                candidate.synth_field_count,
                                candidate.exact_offset_matches,
                                candidate.type_matches);
        std::sort(candidate.fields.begin(), candidate.fields.end(), existing_field_less);
        result.push_back(std::move(candidate));
    }

    std::sort(result.begin(), result.end(), [](const TypeOverlapCandidate& a,
                                               const TypeOverlapCandidate& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        if (a.matched_existing_fields != b.matched_existing_fields) {
            return a.matched_existing_fields > b.matched_existing_fields;
        }
        return std::string(a.name.c_str()) < std::string(b.name.c_str());
    });

    if (result.size() > max_results) {
        result.resize(max_results);
    }

    return result;
#endif
}

TypeMergeResult ExistingTypeMatcher::merge_existing_type(
    SynthStruct& synth_struct,
    const TypeOverlapCandidate& candidate) const
{
    TypeMergeResult result;
    if (candidate.tid == BADADDR || candidate.fields.empty()) {
        result.message = "No existing type fields to merge";
        return result;
    }

    // Stage the complete overlay so failed allocations and rejected fields
    // never leave the caller with partially moved names, types, or evidence.
    SynthStruct merged_structure = synth_struct;
    for (const auto& existing : candidate.fields) {
        if (existing.is_padding || existing.size == 0) {
            continue;
        }
        const auto existing_end =
            checked_interval_end(existing.offset, existing.size);
        if (existing.is_bitfield || existing.offset < 0 || !existing_end.has_value() ||
            *existing_end <= 0 ||
            static_cast<std::uint64_t>(*existing_end) > MAX_STRUCT_SIZE) {
            ++result.fields_skipped;
            continue;
        }

        const SynthField* exact = nullptr;
        bool ambiguous_exact = false;
        for (const auto& synth : merged_structure.fields) {
            if (synth.offset == existing.offset && !is_effective_padding(synth)) {
                if (exact != nullptr) {
                    ambiguous_exact = true;
                    break;
                }
                exact = &synth;
            }
        }

        // Existing types may supply names and fill holes, but cannot erase
        // observed storage or collapse union/bitfield evidence to one scalar.
        if (ambiguous_exact || (exact &&
            (existing.size < exact->size || exact->is_bitfield ||
             exact->is_union_candidate || !exact->union_members.empty()))) {
            ++result.fields_skipped;
            continue;
        }

        array_type_data_t array;
        if ((!existing.type.empty() &&
             existing.type.get_size() != existing.size) ||
            (existing.type.empty() && (!exact || existing.size != exact->size)) ||
            (existing.type.is_array() &&
             (!existing.type.get_array_details(&array) || array.nelems == 0 ||
              array.nelems > std::numeric_limits<std::uint32_t>::max()))) {
            ++result.fields_skipped;
            continue;
        }

        bool conflicts_with_real_field = false;
        for (const auto& synth : merged_structure.fields) {
            if (&synth == exact || is_effective_padding(synth)) {
                continue;
            }
            if (ranges_overlap(synth.offset, synth.size, existing.offset, existing.size)) {
                conflicts_with_real_field = true;
                break;
            }
        }

        if (conflicts_with_real_field) {
            ++result.fields_skipped;
            continue;
        }

        SynthField merged = exact ? *exact : SynthField{};
        const bool renamed = exact && !existing.name.empty() &&
            field_name_can_be_reused(*exact) && exact->name != existing.name;
        if (!exact || renamed) {
            merged.name = existing.name;
            merged.naming.kind = GeneratedNameKind::Field;
            merged.naming.origin = NameOrigin::ReusedType;
            merged.naming.confidence = NameConfidence::High;
        }
        if (merged.name.empty()) {
            merged.name.sprnt("field_%X", static_cast<unsigned>(existing.offset));
        }
        merged.offset = existing.offset;
        merged.size = existing.size;
        if (!existing.type.empty()) {
            merged.type = existing.type;
            merged.semantic = semantic_from_type(existing.type);
            merged.confidence = TypeConfidence::High;
            merged.is_array = existing.type.is_array();
            merged.array_count = merged.is_array
                ? static_cast<std::uint32_t>(array.nelems) : 1;
        }
        if (!exact) {
            merged.comment.sprnt("Merged from existing type %s", candidate.name.c_str());
        }

        qvector<SynthField> kept;
        kept.reserve(merged_structure.fields.size() + 2);
        for (const auto& synth : merged_structure.fields) {
            if (&synth == exact) {
                continue;
            }
            if (!ranges_overlap(synth.offset, synth.size, existing.offset, existing.size)) {
                kept.push_back(synth);
                continue;
            }

            // Only padding can overlap here. Retain both uncovered fragments;
            // dropping the entire original gap destroys its declared extent.
            const auto padding_end = checked_interval_end(synth.offset, synth.size);
            if (synth.offset < existing.offset) {
                kept.push_back(SynthField::create_padding(
                    synth.offset, static_cast<std::uint32_t>(existing.offset - synth.offset)));
            }
            if (padding_end && *padding_end > *existing_end) {
                kept.push_back(SynthField::create_padding(
                    *existing_end, static_cast<std::uint32_t>(*padding_end - *existing_end)));
            }
        }
        kept.push_back(std::move(merged));
        if (kept.size() > MAX_FIELDS) {
            ++result.fields_skipped;
            continue;
        }
        const bool retyped = exact && !existing.type.empty();
        const bool added = exact == nullptr;
        merged_structure.fields = std::move(kept);
        merged_structure.size = std::max(
            merged_structure.size, static_cast<std::uint32_t>(*existing_end));
        result.fields_added += added;
        result.fields_renamed += renamed;
        result.fields_retyped += retyped;
    }

    if (result.fields_added || result.fields_renamed || result.fields_retyped) {
        std::sort(merged_structure.fields.begin(), merged_structure.fields.end(), [](const SynthField& a,
                                                                         const SynthField& b) {
        if (a.offset != b.offset) {
            return a.offset < b.offset;
        }
        if (a.is_bitfield != b.is_bitfield) {
            return a.is_bitfield;
        }
        return a.bit_offset < b.bit_offset;
    });
        uniquify_field_names(merged_structure.fields);
        synth_struct = std::move(merged_structure);
    }

    result.success = true;
    result.message.sprnt("Merged %s: +%u fields, renamed %u, retyped %u, skipped %u conflicts",
                         candidate.name.c_str(),
                         result.fields_added,
                         result.fields_renamed,
                         result.fields_retyped,
                         result.fields_skipped);
    return result;
}

} // namespace structor
