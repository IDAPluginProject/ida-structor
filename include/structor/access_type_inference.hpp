#pragma once

#include "synth_types.hpp"

namespace structor::detail {

/// Infer the type of an access pattern's base, not the type of a loaded field.
/// A homogeneous, direct access at byte offset zero supports T*. Displaced,
/// nested, overlapping, or incomplete observations establish pointer use but
/// require layout synthesis to identify the pointee. O(n + d) type operations
/// for n accesses and d levels in the compared pointer types, with
/// O(1) additional type handles; type-system comparison costs are external.
[[nodiscard]] inline tinfo_t infer_base_type_from_accesses(
    const AccessPattern& pattern,
    TypeConfidence& out_confidence)
{
    out_confidence = TypeConfidence::Low;
    tinfo_t pointee;
    bool saw_access = false;
    bool can_identify_pointee = true;
    bool has_distinct_site = false;
    ea_t first_site = BADADDR;
    ea_t first_function = BADADDR;

    for (const auto& access : pattern.accesses) {
        if (access.size == 0 ||
            !checked_interval_end(access.offset, access.size).has_value()) {
            can_identify_pointee = false;
            continue;
        }
        if (!saw_access) {
            first_site = access.insn_ea;
            first_function = access.source_func_ea;
        } else if (access.insn_ea != BADADDR && first_site != BADADDR &&
                   (access.insn_ea != first_site ||
                    access.source_func_ea != first_function)) {
            has_distinct_site = true;
        }
        saw_access = true;

        const size_t type_size = access.inferred_type.get_size();
        if (access.offset != 0 || access.is_vtable_access ||
            access.is_nested_access() ||
            // cot_ptr observations include their own load in this count;
            // member accesses can report zero for the equivalent direct load.
            access.base_indirection.value_or(0) > 1 ||
            (access.array_stride_hint.has_value() &&
             *access.array_stride_hint != access.size) ||
            access.inferred_type.empty() || type_size == BADSIZE ||
            type_size != access.size) {
            can_identify_pointee = false;
            continue;
        }
        if (pointee.empty()) {
            pointee = access.inferred_type;
        } else if (!pointee.equals_to(access.inferred_type)) {
            // Do not resolve incompatible views by semantic priority. A union
            // or a narrower cast cannot identify a unique scalar pointee.
            can_identify_pointee = false;
        }
    }

    if (!saw_access) {
        return {};
    }
    if (has_distinct_site) {
        out_confidence = TypeConfidence::Medium;
    }

    if (can_identify_pointee && !pointee.empty()) {
        if (pattern.original_type.is_ptr()) {
            const tinfo_t original_pointee =
                pattern.original_type.get_pointed_object();
            const size_t original_size = original_pointee.get_size();
            tinfo_t original_leaf = original_pointee;
            tinfo_t observed_leaf = pointee;
            size_t original_depth = 0;
            size_t observed_depth = 0;
            while (original_leaf.is_ptr()) {
                ++original_depth;
                original_leaf = original_leaf.get_pointed_object();
            }
            while (observed_leaf.is_ptr()) {
                ++observed_depth;
                observed_leaf = observed_leaf.get_pointed_object();
            }
            if (original_leaf.is_struct() || original_leaf.is_union() ||
                original_leaf.is_array() || original_leaf.is_func() ||
                original_depth > observed_depth ||
                (!original_pointee.empty() && original_size != BADSIZE &&
                 original_size > pointee.get_size())) {
                // A representation read does not erase a known aggregate or
                // callable interpretation, even through multiple pointers.
                // A shallower cast also cannot disprove deeper indirection.
                // Partial reads also cannot prove a smaller pointee extent.
                return pattern.original_type;
            }
        }
        tinfo_t result;
        result.create_ptr(pointee);
        return result;
    }

    // Partial evidence must not erase an existing pointer's indirection or
    // aggregate type. An untyped/integer base can still be identified as a
    // pointer until structure synthesis resolves its layout.
    if (pattern.original_type.is_ptr()) {
        return pattern.original_type;
    }
    tinfo_t void_type;
    void_type.create_simple_type(BTF_VOID);
    tinfo_t result;
    result.create_ptr(void_type);
    return result;
}

} // namespace structor::detail
