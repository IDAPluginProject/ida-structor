#pragma once

#include "synth_types.hpp"

namespace structor::detail {

/// Preserve the selected element type when subobject extraction splits an
/// array field into complete elements. Byte offsets and sizes must describe
/// a contained, aligned slice of the source's actual type extent. Partial
/// elements and independently known types retain their observed type. O(1)
/// type operations; type-system comparison costs are external.
[[nodiscard]] inline bool retain_array_fragment_type(
    const SynthField& source, SynthField& fragment)
{
    array_type_data_t source_array;
    if (source.size == 0 || fragment.size == 0 ||
        !source.type.get_array_details(&source_array) ||
        source.type.get_size() != source.size ||
        !checked_interval_contains(source.offset, source.size,
                                   fragment.offset, fragment.size)) {
        return false;
    }

    const size_t element_size = source_array.elem_type.get_size();
    const auto relative_offset =
        checked_interval_span(source.offset, fragment.offset);
    if (element_size == 0 || element_size == BADSIZE ||
        !relative_offset.has_value() ||
        *relative_offset % element_size != 0 ||
        fragment.size % element_size != 0) {
        return false;
    }

    const auto element_count =
        static_cast<std::uint32_t>(fragment.size / element_size);
    tinfo_t retained_type = source_array.elem_type;
    if (element_count > 1) {
        retained_type.create_array(source_array.elem_type, element_count);
    }
    if (retained_type.empty() || retained_type.get_size() != fragment.size) {
        return false;
    }
    if (!fragment.type.empty() && !fragment.type.is_partial()) {
        // The solved array can refine unknown storage such as _QWORD, but
        // cannot erase a residual's independently known pointer, signature,
        // aggregate, scalar interpretation, or array dimensions. Even equal
        // types can carry a distinct spelling or typedef identity.
        return false;
    }

    SemanticType semantic = SemanticType::Unknown;
    std::uint32_t array_count = 1;
    array_type_data_t retained_array;
    if (retained_type.get_array_details(&retained_array)) {
        semantic = SemanticType::Array;
        array_count = retained_array.nelems;
    } else if (retained_type.is_funcptr()) {
        semantic = SemanticType::FunctionPointer;
    } else if (retained_type.is_ptr()) {
        semantic = SemanticType::Pointer;
    } else if (retained_type.is_struct() || retained_type.is_union()) {
        semantic = SemanticType::NestedStruct;
    } else if (retained_type.is_floating()) {
        if (fragment.size == 4) {
            semantic = SemanticType::Float;
        } else if (fragment.size == 8) {
            semantic = SemanticType::Double;
        }
    } else if (retained_type.is_unsigned()) {
        semantic = SemanticType::UnsignedInteger;
    } else if (retained_type.is_signed()) {
        semantic = SemanticType::Integer;
    }

    fragment.type = retained_type;
    fragment.semantic = semantic;
    fragment.is_array = retained_type.is_array();
    fragment.array_count = array_count;
    return true;
}

} // namespace structor::detail
