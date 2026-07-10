#include "structor/z3/field_candidates.hpp"
#include "structor/z3/array_constraints.hpp"
#include "structor/naming.hpp"
#include "structor/optimized_algorithms.hpp"
#include "structor/optimized_containers.hpp"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#ifndef STRUCTOR_TESTING
#include <pro.h>
#include <kernwin.hpp>
#endif

namespace structor::z3 {

namespace {
    // Helper for conditional logging
    inline void z3_log(const char* fmt, ...) {
#ifndef STRUCTOR_TESTING
        va_list va;
        va_start(va, fmt);
        vmsg(fmt, va);
        va_end(va);
#endif
    }

    struct MixedStrideField {
        uint32_t inner_offset = 0;
        uint32_t size = 0;
        // Index of representative type evidence in UnifiedAccessPattern.
        // Keeping only an integer makes worker-side augmentation independent
        // of non-THREAD_SAFE tinfo_t copy/destruction operations.
        int type_access_index = -1;
        qvector<int> access_indices;
    };

    struct MixedStrideKey {
        uint32_t inner_offset = 0;
        uint32_t size = 0;

        bool operator==(const MixedStrideKey& other) const noexcept {
            return inner_offset == other.inner_offset && size == other.size;
        }
    };

    struct MixedStrideKeyHash {
        size_t operator()(const MixedStrideKey& key) const noexcept {
            return (static_cast<size_t>(key.inner_offset) << 16) ^ key.size;
        }
    };

    std::unordered_set<uint32_t> collect_element_indices(const MixedStrideField& field,
                                                         const UnifiedAccessPattern& pattern,
                                                         sval_t base,
                                                         uint32_t stride,
                                                         std::optional<uint32_t> max_index = std::nullopt) {
        std::unordered_set<uint32_t> indices;
        if (stride == 0) {
            return indices;
        }

        for (int access_idx : field.access_indices) {
            if (access_idx < 0 || static_cast<size_t>(access_idx) >= pattern.all_accesses.size()) {
                continue;
            }

            const auto& access = pattern.all_accesses[static_cast<size_t>(access_idx)];
            if (access.offset < base) {
                continue;
            }

            const uint32_t idx = static_cast<uint32_t>((access.offset - base) / stride);
            if (max_index.has_value() && idx >= *max_index) {
                continue;
            }

            indices.insert(idx);
        }

        return indices;
    }

    bool covers_all_elements(const MixedStrideField& field,
                             const UnifiedAccessPattern& pattern,
                             sval_t base,
                             uint32_t stride,
                             uint32_t required_count) {
        if (required_count == 0) {
            return false;
        }

        const auto indices = collect_element_indices(field, pattern, base, stride, required_count);
        if (indices.size() < required_count) {
            return false;
        }

        for (uint32_t i = 0; i < required_count; ++i) {
            if (indices.count(i) == 0) {
                return false;
            }
        }

        return true;
    }

    size_t count_stable_repeated_fields(const qvector<MixedStrideField>& fields,
                                        const UnifiedAccessPattern& pattern,
                                        sval_t base,
                                        uint32_t stride,
                                        uint32_t required_count) {
        size_t stable = 0;
        for (const auto& field : fields) {
            if (covers_all_elements(field, pattern, base, stride, required_count)) {
                ++stable;
            }
        }
        return stable;
    }

    qvector<MixedStrideField> collapse_fields_by_inner_offset(
        const qvector<MixedStrideField>& fields,
        const UnifiedAccessPattern& pattern,
        sval_t base,
        uint32_t stride,
        uint32_t min_repeats,
        std::optional<uint32_t> max_index = std::nullopt) {
        std::unordered_map<uint32_t, MixedStrideField> merged;
        std::unordered_map<uint32_t, std::unordered_set<int>> merged_accesses;
        std::unordered_map<uint32_t, std::unordered_set<uint32_t>> merged_indices;

        for (const auto& field : fields) {
            const auto indices = collect_element_indices(field, pattern, base, stride, max_index);
            auto& collapsed = merged[field.inner_offset];
            collapsed.inner_offset = field.inner_offset;
            if (collapsed.size < field.size) {
                collapsed.size = field.size;
                if (field.type_access_index >= 0) {
                    collapsed.type_access_index = field.type_access_index;
                }
            } else if (collapsed.type_access_index < 0 &&
                       field.type_access_index >= 0) {
                collapsed.type_access_index = field.type_access_index;
            }

            auto& access_set = merged_accesses[field.inner_offset];
            access_set.insert(field.access_indices.begin(), field.access_indices.end());

            auto& index_set = merged_indices[field.inner_offset];
            index_set.insert(indices.begin(), indices.end());
        }

        qvector<MixedStrideField> result;
        result.reserve(merged.size());
        for (auto& [inner_offset, field] : merged) {
            auto indices_it = merged_indices.find(inner_offset);
            if (indices_it == merged_indices.end() || indices_it->second.size() < min_repeats) {
                continue;
            }

            auto accesses_it = merged_accesses.find(inner_offset);
            if (accesses_it != merged_accesses.end()) {
                field.access_indices.clear();
                for (int access_idx : accesses_it->second) {
                    field.access_indices.push_back(access_idx);
                }
                std::sort(field.access_indices.begin(), field.access_indices.end());
            }

            result.push_back(std::move(field));
        }

        std::sort(result.begin(), result.end(), [](const MixedStrideField& a, const MixedStrideField& b) {
            if (a.inner_offset != b.inner_offset) return a.inner_offset < b.inner_offset;
            return a.size > b.size;
        });
        return result;
    }

    bool is_dominated_by_struct_array(const FieldCandidate& array_candidate,
                                      const FieldCandidate& other) {
        if (array_candidate.kind != FieldCandidate::Kind::ArrayField ||
            array_candidate.type_category != TypeCategory::Struct) {
            return false;
        }

        if (other.kind != FieldCandidate::Kind::DirectAccess &&
            other.kind != FieldCandidate::Kind::ArrayField) {
            return false;
        }

        const bool compact_byte_array =
            other.kind == FieldCandidate::Kind::ArrayField &&
            other.type_category == TypeCategory::UInt8 &&
            other.array_stride.value_or(0) == 1 &&
            other.size <= 16;
        const bool scalar_array_field =
            other.kind == FieldCandidate::Kind::ArrayField &&
            other.type_category != TypeCategory::Struct;
        if (compact_byte_array || scalar_array_field) {
            return false;
        }

        if (array_candidate.offset > other.offset ||
            array_candidate.end_offset() < other.end_offset()) {
            return false;
        }

        if (other.kind == FieldCandidate::Kind::DirectAccess) {
            return other.offset == array_candidate.offset ||
                   (other.offset >= array_candidate.offset &&
                    other.offset < array_candidate.end_offset() &&
                    array_candidate.source_access_indices.size() > other.source_access_indices.size());
        }

        if (other.kind == FieldCandidate::Kind::ArrayField &&
            other.type_category == TypeCategory::Struct) {
            return array_candidate.offset < other.offset &&
                   array_candidate.end_offset() >= other.end_offset();
        }

        return array_candidate.offset < other.offset ||
               array_candidate.source_access_indices.size() > other.source_access_indices.size();
    }

    bool build_struct_type_from_groups(
        const qvector<MixedStrideField>& fields,
        const UnifiedAccessPattern& pattern,
        uint32_t stride,
        tinfo_t& out_type)
    {
        if (fields.size() < 2) {
            return false;
        }

        udt_type_data_t udt;
        udt.is_union = false;
        udt.total_size = stride;

        auto append_gap_member = [&](uint32_t offset, uint32_t size) {
            if (size == 0) {
                return;
            }

            udm_t gap;
            gap.offset = static_cast<uint64>(offset) * 8;
            gap.name = generate_field_name(offset, SemanticType::Unknown, size);

            if (size == 4) {
                gap.type.create_simple_type(BT_INT32 | BTMT_UNSIGNED);
            } else if (size == 2) {
                gap.type.create_simple_type(BT_INT16 | BTMT_UNSIGNED);
            } else if (size == 1) {
                gap.type.create_simple_type(BT_INT8 | BTMT_UNSIGNED);
            } else {
                tinfo_t byte_type;
                byte_type.create_simple_type(BT_INT8 | BTMT_UNSIGNED);
                gap.type.create_array(byte_type, size);
            }

            const asize_t member_size = gap.type.get_size();
            gap.size = member_size == BADSIZE ? size * 8 : member_size * 8;
            udt.push_back(gap);
        };

        uint32_t cursor = 0;

        const auto field_type = [&](const MixedStrideField& field)
            -> const tinfo_t* {
            if (field.type_access_index < 0 ||
                static_cast<std::size_t>(field.type_access_index) >=
                    pattern.all_accesses.size()) {
                return nullptr;
            }
            const tinfo_t& type = pattern.all_accesses[
                static_cast<std::size_t>(field.type_access_index)].inferred_type;
            return type.empty() ? nullptr : &type;
        };

        auto is_function_pointer_type = [](const tinfo_t* type) {
            if (type == nullptr || type->empty()) {
                return false;
            }
            if (type->is_func()) {
                return true;
            }
            if (type->is_funcptr()) {
                return true;
            }
            if (!type->is_ptr()) {
                return false;
            }
            tinfo_t pointed = type->get_pointed_object();
            return !pointed.empty() && pointed.is_func();
        };

        int func_field_count = 0;
        for (const auto& field : fields) {
            if (is_function_pointer_type(field_type(field))) {
                ++func_field_count;
            }
        }

        auto is_array_mergeable = [&](const MixedStrideField& a, const MixedStrideField& b) {
            if (a.inner_offset + a.size != b.inner_offset) {
                return false;
            }
            if (a.size != b.size) {
                return false;
            }
            const tinfo_t* a_type = field_type(a);
            const tinfo_t* b_type = field_type(b);
            if ((a_type == nullptr) != (b_type == nullptr)) {
                return false;
            }
            if (a_type != nullptr) {
                return a_type->equals_to(*b_type);
            }
            return true;
        };

        auto is_byte_field = [&](const MixedStrideField& field) {
            if (field.size != 1) {
                return false;
            }
            const tinfo_t* type = field_type(field);
            return type == nullptr || type->get_size() == 1;
        };

        for (size_t i = 0; i < fields.size(); ++i) {
            MixedStrideField merged = fields[i];
            uint32_t merged_count = 1;

            size_t j = i + 1;
            while (j < fields.size() && is_array_mergeable(merged, fields[j]) &&
                   is_byte_field(merged) && is_byte_field(fields[j])) {
                merged.size += fields[j].size;
                ++merged_count;
                ++j;
            }
            i = j - 1;

            if (merged.inner_offset > cursor) {
                append_gap_member(cursor, merged.inner_offset - cursor);
            }

            udm_t udm;
            udm.offset = static_cast<uint64>(merged.inner_offset) * 8;
            const tinfo_t* merged_type = field_type(merged);
            if (merged_count > 1) {
                tinfo_t unknown_type;
                udm.name = make_array_field_name(merged.inner_offset,
                                                 merged_type == nullptr
                                                     ? unknown_type
                                                     : *merged_type,
                                                 SemanticType::Unknown,
                                                 fields[i].size);
            } else if (is_function_pointer_type(merged_type)) {
                if (func_field_count == 1) {
                    udm.name = "callback";
                } else {
                    udm.name.sprnt("callback_%X", merged.inner_offset);
                }
            } else {
                udm.name = generate_field_name(merged.inner_offset,
                                               SemanticType::Unknown,
                                               merged.size);
            }

            if (merged_count > 1) {
                tinfo_t elem_type;
                if (merged_type != nullptr) {
                    elem_type = *merged_type;
                } else {
                    elem_type.create_simple_type(BT_INT8 | BTMT_USIGNED);
                }

                udm.type.create_array(elem_type, merged_count);
                udm.size = merged.size * 8;
            } else if (merged_type != nullptr) {
                udm.type = *merged_type;
                udm.size = merged_type->get_size() * 8;
            } else {
                tinfo_t byte_type;
                byte_type.create_simple_type(BT_INT8 | BTMT_USIGNED);
                if (merged.size > 1) {
                    udm.type.create_array(byte_type, merged.size);
                } else {
                    udm.type = byte_type;
                }
                udm.size = merged.size * 8;
            }
            udt.push_back(udm);

            cursor = std::max(cursor, merged.inner_offset + merged.size);
        }

        if (cursor < stride) {
            append_gap_member(cursor, stride - cursor);
        }

        if (!out_type.create_udt(udt)) {
            return false;
        }

        out_type.set_udt_pack(1);
        out_type.set_udt_alignment(1);
        return true;
    }

    double repeated_field_coverage_ratio(const qvector<MixedStrideField>& fields, uint32_t stride) {
        if (stride == 0 || fields.empty()) {
            return 0.0;
        }

        uint32_t covered = 0;
        for (const auto& field : fields) {
            covered += field.size;
        }

        return static_cast<double>(covered) / static_cast<double>(stride);
    }

    bool overlaps_scalar_array_field(const FieldCandidate& candidate,
                                     const qvector<FieldCandidate>& candidates) {
        if (candidate.kind != FieldCandidate::Kind::ArrayField ||
            candidate.type_category != TypeCategory::Struct) {
            return false;
        }

        for (const auto& other : candidates) {
            if (other.kind != FieldCandidate::Kind::ArrayField ||
                other.type_category == TypeCategory::Struct) {
                continue;
            }

            if (candidate.overlaps(other)) {
                return true;
            }
        }

        return false;
    }

    bool struct_array_depends_on_mixed_size_collapse(const UnifiedAccessPattern& pattern,
                                                     const FieldCandidate& candidate) {
        if (candidate.kind != FieldCandidate::Kind::ArrayField ||
            candidate.type_category != TypeCategory::Struct ||
            !candidate.array_stride.has_value() ||
            !candidate.array_element_count.has_value()) {
            return false;
        }

        const uint32_t stride = *candidate.array_stride;
        const uint32_t count = *candidate.array_element_count;
        if (stride == 0 || count < 3) {
            return false;
        }

        std::unordered_map<MixedStrideKey, MixedStrideField, MixedStrideKeyHash> by_inner;
        for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
            const auto& access = pattern.all_accesses[i];
            if (access.offset < candidate.offset) {
                continue;
            }

            sval_t rel = access.offset - candidate.offset;
            if (rel < 0) {
                continue;
            }

            uint32_t idx = static_cast<uint32_t>(rel / stride);
            if (idx >= count) {
                continue;
            }

            uint32_t inner = static_cast<uint32_t>(rel % stride);
            if (inner > stride || access.size > stride - inner) {
                continue;
            }

            MixedStrideKey key{inner, access.size};
            auto& field = by_inner[key];
            field.inner_offset = inner;
            field.size = access.size;
            if (field.type_access_index < 0 && !access.inferred_type.empty()) {
                field.type_access_index = static_cast<int>(i);
            }
            field.access_indices.push_back(static_cast<int>(i));
        }

        qvector<MixedStrideField> raw_fields;
        raw_fields.reserve(by_inner.size());
        for (auto& [key, field] : by_inner) {
            raw_fields.push_back(field);
        }

        const size_t stable_fields = count_stable_repeated_fields(
            raw_fields,
            pattern,
            candidate.offset,
            stride,
            count);
        qvector<MixedStrideField> repeated = collapse_fields_by_inner_offset(
            raw_fields,
            pattern,
            candidate.offset,
            stride,
            count);

        return repeated.size() > stable_fields;
    }

    bool has_mixed_scalar_array_semantics(const UnifiedAccessPattern& pattern,
                                          const qvector<int>& source_indices) {
        bool saw_pointer_like = false;
        bool saw_scalar_like = false;

        for (int idx : source_indices) {
            if (idx < 0 || static_cast<size_t>(idx) >= pattern.all_accesses.size()) {
                continue;
            }

            const auto& access = pattern.all_accesses[static_cast<size_t>(idx)];
            switch (access.semantic_type) {
                case SemanticType::Pointer:
                case SemanticType::FunctionPointer:
                case SemanticType::VTablePointer:
                    saw_pointer_like = true;
                    break;
                case SemanticType::Integer:
                case SemanticType::UnsignedInteger:
                case SemanticType::Float:
                case SemanticType::Double:
                    saw_scalar_like = true;
                    break;
                default:
                    break;
            }

            if (!access.inferred_type.empty()) {
                if (access.inferred_type.is_ptr() || access.inferred_type.is_funcptr()) {
                    saw_pointer_like = true;
                } else if (access.inferred_type.is_integral() || access.inferred_type.is_floating()) {
                    saw_scalar_like = true;
                }
            }
        }

        return saw_pointer_like && saw_scalar_like;
    }

    bool has_multiple_nonbyte_repeated_fields(const qvector<MixedStrideField>& fields) {
        int nonbyte_count = 0;
        for (const auto& field : fields) {
            if (field.size > 1) {
                ++nonbyte_count;
            }
        }
        return nonbyte_count >= 2;
    }

    bool overlaps_struct_array_prefix(const ArrayCandidate& array,
                                      const qvector<ArrayCandidate>& arrays) {
        if (array.needs_element_struct || array.element_count < 3 || array.stride == 0) {
            return false;
        }

        const sval_t array_begin = array.base_offset;
        const auto checked_array_end = array.checked_end_offset();
        if (!checked_array_end) {
            return true;
        }
        const sval_t array_end = *checked_array_end;
        for (const auto& other : arrays) {
            if (!other.needs_element_struct || other.base_offset <= array_begin) {
                continue;
            }

            if (other.base_offset >= array_end) {
                continue;
            }

            return true;
        }

        return false;
    }

    void augment_struct_array_candidate(ArrayCandidate& array, const UnifiedAccessPattern& pattern) {
        if (!array.needs_element_struct || array.stride == 0 || array.element_count < 3) {
            return;
        }

        constexpr double kMinStructElementCoverage = 0.75;

        struct BestAugmentation {
            sval_t base = 0;
            qvector<MixedStrideField> fields;
            int score = -1;
        } best;

        std::vector<BestAugmentation> shift_results(array.stride);
        algorithms::parallel_for_chunks(array.stride, 4, [&](size_t begin, size_t end) {
            for (size_t shift_index = begin; shift_index < end; ++shift_index) {
                const uint32_t shift = static_cast<uint32_t>(shift_index);
                if (array.base_offset <
                    std::numeric_limits<sval_t>::min() +
                        static_cast<sval_t>(shift)) {
                    continue;
                }
                const sval_t base =
                    array.base_offset - static_cast<sval_t>(shift);
                std::unordered_map<MixedStrideKey, MixedStrideField, MixedStrideKeyHash> groups;

                for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
                    const auto& access = pattern.all_accesses[i];
                    if (access.offset < base) {
                        continue;
                    }
                    const sval_t rel = access.offset - base;
                    if (rel < 0) {
                        continue;
                    }
                    uint32_t idx = static_cast<uint32_t>(rel / array.stride);
                    if (idx >= array.element_count) {
                        continue;
                    }
                    uint32_t inner = static_cast<uint32_t>(rel % array.stride);
                    if (inner > array.stride ||
                        access.size > array.stride - inner) {
                        continue;
                    }

                    MixedStrideKey key{inner, access.size};
                    auto& field = groups[key];
                    field.inner_offset = inner;
                    field.size = access.size;
                    if (field.type_access_index < 0 &&
                        !access.inferred_type.empty()) {
                        field.type_access_index = static_cast<int>(i);
                    }
                    field.access_indices.push_back(static_cast<int>(i));
                }

                qvector<MixedStrideField> raw_fields;
                raw_fields.reserve(groups.size());
                for (auto& [key, field] : groups) {
                    raw_fields.push_back(field);
                }

                const size_t stable_fields = count_stable_repeated_fields(
                    raw_fields,
                    pattern,
                    base,
                    array.stride,
                    array.element_count);
                const size_t min_stable_fields = base == 0 ? 2u : 1u;
                if (stable_fields < min_stable_fields) {
                    continue;
                }

                qvector<MixedStrideField> repeated = collapse_fields_by_inner_offset(
                    raw_fields,
                    pattern,
                    base,
                    array.stride,
                    std::min<uint32_t>(3, array.element_count));
                const int score = static_cast<int>(repeated.size());

                if (score < 2) {
                    continue;
                }

                std::sort(repeated.begin(), repeated.end(), [](const MixedStrideField& a, const MixedStrideField& b) {
                    if (a.inner_offset != b.inner_offset) return a.inner_offset < b.inner_offset;
                    return a.size > b.size;
                });

                auto& local_best = shift_results[shift_index];
                local_best.base = base;
                local_best.fields = std::move(repeated);
                local_best.score = score;
            }
        });

        for (auto& candidate : shift_results) {
            if (candidate.score < 2) {
                continue;
            }
            if (candidate.score > best.score ||
                (candidate.score == best.score && candidate.base < best.base)) {
                best = std::move(candidate);
            }
        }

        if (best.score < 2) {
            return;
        }

        if (!has_multiple_nonbyte_repeated_fields(best.fields)) {
            return;
        }

        if (repeated_field_coverage_ratio(best.fields, array.stride) < kMinStructElementCoverage) {
            return;
        }

        tinfo_t struct_type;
        if (!build_struct_type_from_groups(
                best.fields, pattern, array.stride, struct_type)) {
            return;
        }

        qvector<sval_t> augmented_offsets;
        bool valid_offsets = true;
        for (const auto& field : best.fields) {
            for (uint32_t i = 0; i < array.element_count; ++i) {
                const std::uint64_t delta =
                    static_cast<std::uint64_t>(field.inner_offset) +
                    static_cast<std::uint64_t>(i) * array.stride;
                if (delta > std::numeric_limits<std::uint32_t>::max()) {
                    valid_offsets = false;
                    break;
                }
                const auto member_offset = checked_interval_end(
                    best.base, static_cast<std::uint32_t>(delta));
                if (!member_offset) {
                    valid_offsets = false;
                    break;
                }
                augmented_offsets.push_back(*member_offset);
            }
            if (!valid_offsets) break;
        }
        if (!valid_offsets) {
            return;
        }
        std::sort(augmented_offsets.begin(), augmented_offsets.end());
        augmented_offsets.erase(
            std::unique(augmented_offsets.begin(), augmented_offsets.end()),
            augmented_offsets.end());

        array.base_offset = best.base;
        array.element_type = struct_type;
        array.member_offsets = std::move(augmented_offsets);
    }

    std::optional<FieldCandidate> build_struct_array_candidate(
        Z3Context& ctx,
        const UnifiedAccessPattern& pattern,
        sval_t base,
        uint32_t stride,
        uint32_t count)
    {
        constexpr double kMinStructElementCoverage = 0.75;

        if (count < 3 || stride == 0) {
            return std::nullopt;
        }

        std::unordered_map<MixedStrideKey, MixedStrideField, MixedStrideKeyHash> by_inner;
        qvector<int> used_indices;

        for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
            const auto& access = pattern.all_accesses[i];
            if (access.offset < base) {
                continue;
            }
            sval_t rel = access.offset - base;
            if (rel < 0) {
                continue;
            }
            uint32_t idx = static_cast<uint32_t>(rel / stride);
            if (idx >= count) {
                continue;
            }
            uint32_t inner = static_cast<uint32_t>(rel % stride);
            if (inner > stride || access.size > stride - inner) {
                continue;
            }

            MixedStrideKey key{inner, access.size};
            auto& field = by_inner[key];
            field.inner_offset = inner;
            field.size = access.size;
            if (field.type_access_index < 0 && !access.inferred_type.empty()) {
                field.type_access_index = static_cast<int>(i);
            }
            field.access_indices.push_back(static_cast<int>(i));
        }

        qvector<MixedStrideField> raw_fields;
        raw_fields.reserve(by_inner.size());
        for (auto& [key, field] : by_inner) {
            raw_fields.push_back(field);
        }

        qvector<MixedStrideField> repeated = collapse_fields_by_inner_offset(
            raw_fields,
            pattern,
            base,
            stride,
            3);

        if (repeated.size() < 2) {
            return std::nullopt;
        }

        if (!has_multiple_nonbyte_repeated_fields(repeated)) {
            return std::nullopt;
        }

        if (repeated_field_coverage_ratio(repeated, stride) < kMinStructElementCoverage) {
            return std::nullopt;
        }

        std::sort(repeated.begin(), repeated.end(), [](const MixedStrideField& a, const MixedStrideField& b) {
            if (a.access_indices.size() != b.access_indices.size()) {
                return a.access_indices.size() > b.access_indices.size();
            }
            if (a.inner_offset != b.inner_offset) {
                return a.inner_offset < b.inner_offset;
            }
            return a.size > b.size;
        });

        uint32_t effective_count = count;
        for (const auto& field : repeated) {
            std::unordered_set<uint32_t> indices = collect_element_indices(field, pattern, base, stride);
            uint32_t contiguous = 0;
            while (indices.count(contiguous) > 0) {
                ++contiguous;
            }
            effective_count = std::min(effective_count, contiguous);
        }

        if (effective_count < 3) {
            return std::nullopt;
        }

        const size_t stable_fields = count_stable_repeated_fields(
            raw_fields,
            pattern,
            base,
            stride,
            effective_count);
        const size_t min_stable_fields = base == 0 ? 2u : 1u;
        if (stable_fields < min_stable_fields) {
            return std::nullopt;
        }

        for (const auto& field : repeated) {
            for (int access_idx : field.access_indices) {
                const auto& access = pattern.all_accesses[access_idx];
                uint32_t idx = static_cast<uint32_t>((access.offset - base) / stride);
                if (idx < effective_count) {
                    used_indices.push_back(access_idx);
                }
            }
        }

        std::sort(repeated.begin(), repeated.end(), [](const MixedStrideField& a, const MixedStrideField& b) {
            return a.inner_offset < b.inner_offset;
        });

        repeated.erase(std::remove_if(repeated.begin(), repeated.end(), [&](const MixedStrideField& field) {
            std::unordered_set<uint32_t> indices = collect_element_indices(
                field, pattern, base, stride, effective_count);
            return indices.size() < effective_count;
        }), repeated.end());

        if (repeated.size() < 2) {
            return std::nullopt;
        }

        std::unordered_set<uint32_t> repeated_offsets;
        for (const auto& field : repeated) {
            repeated_offsets.insert(field.inner_offset);
        }

        std::unordered_map<uint32_t, std::unordered_set<uint32_t>> uncovered_offsets;
        for (const auto& access : pattern.all_accesses) {
            if (access.size <= 1 || access.offset < base) {
                continue;
            }

            const sval_t rel = access.offset - base;
            if (rel < 0) {
                continue;
            }

            const uint32_t idx = static_cast<uint32_t>(rel / stride);
            if (idx >= effective_count) {
                continue;
            }

            const uint32_t inner = static_cast<uint32_t>(rel % stride);
            if (repeated_offsets.count(inner) != 0) {
                continue;
            }

            uncovered_offsets[inner].insert(idx);
        }

        for (const auto& [inner, indices] : uncovered_offsets) {
            if (indices.size() >= std::min<uint32_t>(2, effective_count)) {
                return std::nullopt;
            }
        }

        // Reject spurious "struct arrays" that only explain roughly one
        // access per element. Real arrays-of-structs should expose multiple
        // inner members across repeated elements.
        if (used_indices.size() < static_cast<size_t>(effective_count) * 2) {
            return std::nullopt;
        }

        if (stride != 0 &&
            effective_count > std::numeric_limits<uint32_t>::max() / stride) {
            return std::nullopt;
        }

        tinfo_t elem_type;
        if (!build_struct_type_from_groups(
                repeated, pattern, stride, elem_type)) {
            return std::nullopt;
        }

        FieldCandidate candidate;
        candidate.offset = base;
        candidate.size = stride * effective_count;
        candidate.kind = FieldCandidate::Kind::ArrayField;
        candidate.type_category = TypeCategory::Struct;
        candidate.extended_type = ctx.type_encoder().extract_extended_info(elem_type);
        candidate.array_element_count = effective_count;
        candidate.array_stride = stride;
        candidate.confidence = TypeConfidence::Medium;
        candidate.source_access_indices = std::move(used_indices);
        return candidate;
    }
}

// ============================================================================
// FieldCandidateGenerator Implementation
// ============================================================================

FieldCandidateGenerator::FieldCandidateGenerator(
    Z3Context& ctx,
    const CandidateGenerationConfig& config)
    : ctx_(ctx)
    , config_(config) {}

qvector<FieldCandidate> FieldCandidateGenerator::generate(
    const UnifiedAccessPattern& pattern)
{
    qvector<FieldCandidate> candidates;
    next_id_ = 0;

    z3_log("[Structor/Z3] Generating field candidates from %zu accesses\n", pattern.all_accesses.size());

    if (pattern.all_accesses.empty()) {
        return candidates;
    }

    if (config_.max_accesses == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::Accesses, 0, pattern.all_accesses.size(),
            "candidate_generation",
            "configured access-evidence limit is zero");
    }
    if (config_.max_candidates == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::Candidates, 0, 0,
            "candidate_generation",
            "configured field-candidate limit is zero");
    }
    if (config_.max_array_elements == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::ArrayElements, 0, 0,
            "candidate_generation",
            "configured array-inference element limit is zero");
    }
    if (config_.max_structure_size == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::StructureSize, 0, 0,
            "candidate_generation",
            "configured structure-size limit is zero");
    }
    if (pattern.all_accesses.size() > config_.max_accesses) {
        throw ResourceLimitException(
            ResourceLimitKind::Accesses,
            config_.max_accesses,
            pattern.all_accesses.size(),
            "candidate_generation",
            "access evidence exceeds the configured synthesis limit");
    }

    sval_t evidence_origin = 0;
    sval_t evidence_end = 0;
    for (const auto& access : pattern.all_accesses) {
        evidence_origin = std::min(evidence_origin, access.offset);
        const auto end = checked_interval_end(access.offset, access.size);
        if (!end) {
            throw ResourceLimitException(
                ResourceLimitKind::StructureSize,
                config_.max_structure_size,
                std::numeric_limits<std::uint64_t>::max(),
                "candidate_generation",
                "access evidence interval overflows the signed offset domain");
        }
        evidence_end = std::max(evidence_end, *end);
    }
    const auto evidence_span = checked_interval_span(
        evidence_origin, evidence_end);
    if (!evidence_span || *evidence_span > config_.max_structure_size) {
        throw ResourceLimitException(
            ResourceLimitKind::StructureSize,
            config_.max_structure_size,
            evidence_span.value_or(std::numeric_limits<std::uint64_t>::max()),
            "candidate_generation",
            "recovered object span exceeds the configured structure-size limit");
    }

    // Pre-allocate: estimate ~1.5x accesses for direct + covering + arrays + padding
    const std::uint64_t estimated_candidates =
        static_cast<std::uint64_t>(pattern.all_accesses.size()) +
        static_cast<std::uint64_t>(pattern.all_accesses.size()) / 2 + 16;
    const size_t reserve_count = static_cast<size_t>(std::min<std::uint64_t>(
        config_.max_candidates, estimated_candidates));
    candidates.reserve(reserve_count);

    auto enforce_optional_budget = [&]() {
        qvector<FieldCandidate> mandatory;
        qvector<FieldCandidate> optional;
        mandatory.reserve(candidates.size());
        optional.reserve(candidates.size());
        for (auto& candidate : candidates) {
            if (candidate.kind == FieldCandidate::Kind::DirectAccess ||
                candidate.kind == FieldCandidate::Kind::UnionAlternative) {
                mandatory.push_back(std::move(candidate));
            } else if (candidate.within_array_element_limit(
                           config_.max_array_elements) &&
                       candidate.meets_optional_confidence_threshold(
                           config_.min_confidence_percent)) {
                optional.push_back(std::move(candidate));
            }
        }

        if (mandatory.size() > config_.max_candidates) {
            throw ResourceLimitException(
                ResourceLimitKind::Candidates,
                config_.max_candidates,
                mandatory.size(),
                "candidate_generation",
                "mandatory direct evidence exceeds the configured candidate limit");
        }

        finalize_candidates(optional);
        const size_t optional_budget = config_.max_candidates - mandatory.size();
        if (optional.size() > optional_budget) {
            optional.resize(optional_budget);
        }

        candidates = std::move(mandatory);
        for (auto& candidate : optional) {
            candidates.push_back(std::move(candidate));
        }
        finalize_candidates(candidates);
    };

    // Step 1: Generate direct access candidates
    generate_direct_candidates(pattern, candidates);
    size_t direct_count = candidates.size();
    z3_log("[Structor/Z3]   Direct access candidates: %zu\n", direct_count);

    // Step 2: Generate covering candidates (larger fields that cover multiple accesses)
    if (config_.generate_covering_candidates) {
        generate_covering_candidates(pattern, candidates);
        enforce_optional_budget();
        z3_log("[Structor/Z3]   Covering candidates: %zu\n", candidates.size() - direct_count);
    }

    size_t before_array = candidates.size();
    // Step 3: Generate array candidates
    if (config_.generate_array_candidates) {
        generate_array_candidates(pattern, candidates);
        enforce_optional_budget();
        z3_log("[Structor/Z3]   Array candidates: %zu\n", candidates.size() - before_array);
    }

    size_t before_padding = candidates.size();
    // Step 4: Generate padding candidates
    if (config_.generate_padding_candidates) {
        generate_padding_candidates(candidates, pattern.global_max_offset, candidates);
        enforce_optional_budget();
        z3_log("[Structor/Z3]   Padding candidates: %zu\n", candidates.size() - before_padding);
    }

    // Step 5: Prune candidates dominated by richer struct-array candidates.
    // Dominance checks are pure candidate-shape comparisons, so split the
    // expensive O(n^2) scan across workers without touching IDA or Z3 state.
    std::vector<uint8_t> dominated_flags(candidates.size(), 0);
    algorithms::parallel_for_chunks(candidates.size(), 128, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            bool dominated = false;

            if (candidates[i].kind == FieldCandidate::Kind::DirectAccess) {
                for (size_t j = 0; j < candidates.size(); ++j) {
                    if (i == j) {
                        continue;
                    }
                    const auto& other = candidates[j];
                    if (other.kind == FieldCandidate::Kind::ArrayField &&
                        other.type_category == TypeCategory::Struct &&
                        other.offset == candidates[i].offset &&
                        other.end_offset() >= candidates[i].end_offset()) {
                        dominated = true;
                        break;
                    }
                }
            }

            if (!dominated) {
                for (size_t j = 0; j < candidates.size(); ++j) {
                    if (i == j) {
                        continue;
                    }
                    if (is_dominated_by_struct_array(candidates[j], candidates[i])) {
                        dominated = true;
                        break;
                    }
                    if (candidates[i].kind == FieldCandidate::Kind::ArrayField &&
                        candidates[i].type_category != TypeCategory::Struct &&
                        candidates[j].kind == FieldCandidate::Kind::ArrayField &&
                        candidates[j].type_category == TypeCategory::Struct &&
                        candidates[j].offset > candidates[i].offset &&
                        candidates[j].offset < candidates[i].end_offset()) {
                        dominated = true;
                        break;
                    }
                }
            }

            dominated_flags[i] = dominated ? 1 : 0;
        }
    });

    qvector<FieldCandidate> pruned;
    pruned.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (dominated_flags[i] == 0) {
            pruned.push_back(std::move(candidates[i]));
        }
    }
    candidates = std::move(pruned);

    // Finalize: assign IDs and sort
    finalize_candidates(candidates);

    z3_log("[Structor/Z3]   Total candidates generated: %zu\n", candidates.size());
    
    // Log candidate summary by offset
    if (!candidates.empty()) {
        z3_log("[Structor/Z3]   Candidate summary:\n");
        for (const auto& cand : candidates) {
            const char* kind_str = "unknown";
            switch (cand.kind) {
                case FieldCandidate::Kind::DirectAccess: kind_str = "direct"; break;
                case FieldCandidate::Kind::CoveringField: kind_str = "covering"; break;
                case FieldCandidate::Kind::ArrayElement: kind_str = "array_elem"; break;
                case FieldCandidate::Kind::ArrayField: kind_str = "array"; break;
                case FieldCandidate::Kind::PaddingField: kind_str = "padding"; break;
                case FieldCandidate::Kind::UnionAlternative: kind_str = "union_alt"; break;
            }
            z3_log("[Structor/Z3]     [%d] offset=0x%llX size=%u type=%s kind=%s\n",
                   cand.id, static_cast<unsigned long long>(cand.offset), cand.size,
                   type_category_name(cand.type_category), kind_str);
        }
    }

    return candidates;
}

void FieldCandidateGenerator::generate_direct_candidates(
    const UnifiedAccessPattern& pattern,
    qvector<FieldCandidate>& candidates)
{
    for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
        const auto& access = pattern.all_accesses[i];

        if (!access.inferred_type.empty() &&
            (access.inferred_type.is_array() || access.inferred_type.is_struct())) {
            int covered_subaccesses = 0;
            const auto access_end = checked_interval_end(access.offset, access.size);
            if (!access_end) {
                continue;
            }
            for (size_t j = 0; j < pattern.all_accesses.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const auto& other = pattern.all_accesses[j];
                const auto other_end = checked_interval_end(other.offset, other.size);
                if (other_end && other.offset >= access.offset &&
                    *other_end <= *access_end &&
                    (other.size < access.size || other.offset != access.offset)) {
                    ++covered_subaccesses;
                }
            }
            if (covered_subaccesses >= 2) {
                continue;
            }
        }

        TypeCategory new_cat = infer_category(access);
        bool merged = false;

        for (auto& existing : candidates) {
            if (existing.offset != access.offset || existing.size != access.size) {
                continue;
            }

            bool compatible_evidence = true;
            for (int source_idx : existing.source_access_indices) {
                if (source_idx < 0 ||
                    static_cast<size_t>(source_idx) >= pattern.all_accesses.size()) {
                    continue;
                }
                if (!field_access_evidence_compatible(
                        pattern.all_accesses[static_cast<size_t>(source_idx)], access)) {
                    compatible_evidence = false;
                    break;
                }
            }

            if (compatible_evidence) {
                existing.source_access_indices.push_back(static_cast<int>(i));
                if (static_cast<int>(new_cat) > static_cast<int>(existing.type_category)) {
                    existing.type_category = new_cat;
                }
                if (!access.inferred_type.empty()) {
                    existing.extended_type = ctx_.type_encoder().extract_extended_info(access.inferred_type);
                }
                if (existing.primary_func_ea == BADADDR ||
                    (access.source_func_ea != BADADDR &&
                     access.source_func_ea < existing.primary_func_ea)) {
                    existing.primary_func_ea = access.source_func_ea;
                }
                merged = true;
                break;
            }
        }

        if (!merged) {
            if (candidates.size() >= config_.max_candidates) {
                throw ResourceLimitException(
                    ResourceLimitKind::Candidates,
                    config_.max_candidates,
                    candidates.size() + 1,
                    "direct_candidate_generation",
                    "mandatory direct evidence exceeds the configured candidate limit");
            }
            FieldCandidate candidate = create_from_access(access, static_cast<int>(i));
            bool has_same_range_alternative = false;
            for (auto& existing : candidates) {
                if (existing.offset != candidate.offset ||
                    existing.size != candidate.size) {
                    continue;
                }

                // Every evidence-backed interpretation of the same storage
                // range must participate in the union.  Marking only the
                // newly discovered interpretation lets the solver retain the
                // first observation as an ordinary scalar and discard the
                // later alternative.
                existing.kind = FieldCandidate::Kind::UnionAlternative;
                has_same_range_alternative = true;
            }
            if (has_same_range_alternative) {
                candidate.kind = FieldCandidate::Kind::UnionAlternative;
            }
            candidates.push_back(std::move(candidate));
        }
    }
}

void FieldCandidateGenerator::generate_covering_candidates(
    const UnifiedAccessPattern& pattern,
    qvector<FieldCandidate>& candidates)
{
    if (candidates.empty()) return;

    // Sort candidates by offset for analysis
    qvector<FieldCandidate> sorted_candidates = candidates;
    std::sort(sorted_candidates.begin(), sorted_candidates.end(),
        [](const FieldCandidate& a, const FieldCandidate& b) {
            return a.offset < b.offset;
        });

    // Find groups of adjacent small fields that could be covered by a larger field
    qvector<FieldCandidate> covering;

    size_t i = 0;
    while (i < sorted_candidates.size()) {
        // Look for sequence of small fields
        size_t j = i + 1;
        sval_t group_start = sorted_candidates[i].offset;
        sval_t group_end = sorted_candidates[i].end_offset();

        // Extend group while fields are adjacent or slightly gapped
        while (j < sorted_candidates.size()) {
            const auto& next = sorted_candidates[j];
            sval_t gap = next.offset - group_end;

            // Allow small gaps (padding)
            if (gap < 0 || gap > 4) break;

            group_end = next.end_offset();
            ++j;
        }

        // If we found multiple fields, create covering candidate
        if (j > i + 1) {
            uint32_t covering_size = static_cast<uint32_t>(group_end - group_start);

            if (covering_size <= config_.max_covering_size) {
                FieldCandidate cover;
                cover.offset = group_start;
                cover.size = covering_size;
                cover.kind = FieldCandidate::Kind::CoveringField;
                cover.type_category = TypeCategory::RawBytes;
                cover.confidence = TypeConfidence::Low;

                // Track which candidates this covers
                for (size_t k = i; k < j; ++k) {
                    for (int idx : sorted_candidates[k].source_access_indices) {
                        cover.source_access_indices.push_back(idx);
                    }
                }

                covering.push_back(std::move(cover));
            }
        }

        i = j;
    }

    // Add covering candidates
    for (auto& c : covering) {
        candidates.push_back(std::move(c));
    }
}

void FieldCandidateGenerator::generate_array_candidates(
    const UnifiedAccessPattern& pattern,
    qvector<FieldCandidate>& candidates)
{
    constexpr double kMinArrayCoverageRatio = 0.75;

    const ArrayDetectionConfig array_config = make_array_detection_config(
        config_.min_array_elements,
        config_.max_array_elements,
        config_.detect_symbolic_arrays,
        config_.max_array_stride);

    ArrayConstraintBuilder array_builder(ctx_, array_config);
    auto arrays = array_builder.detect_arrays(pattern.all_accesses);
    if (arrays.empty()) {
        return;
    }

    std::unordered_map<uint64_t, int> direct_index;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].kind != FieldCandidate::Kind::DirectAccess) {
            continue;
        }
        uint64_t key = (static_cast<uint64_t>(candidates[i].offset) << 32) |
                       static_cast<uint64_t>(candidates[i].size);
        direct_index[key] = static_cast<int>(i);
    }

        for (const auto& detected_array : arrays) {
            ArrayCandidate array = detected_array;
            if (array.element_count == 0 ||
                array.element_count > config_.max_array_elements ||
                !array.checked_total_size() ||
                !array.checked_end_offset()) {
                continue;
            }

            augment_struct_array_candidate(array, pattern);
            if (!array.checked_total_size() || !array.checked_end_offset()) {
                continue;
            }

        if (array.element_count == 0) {
            continue;
        }

        if (array.needs_element_struct) {
            std::unordered_map<MixedStrideKey, MixedStrideField, MixedStrideKeyHash> raw_groups;
            for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
                const auto& access = pattern.all_accesses[i];
                if (access.offset < array.base_offset) {
                    continue;
                }

                const sval_t rel = access.offset - array.base_offset;
                const uint32_t idx = static_cast<uint32_t>(rel / array.stride);
                if (idx >= array.element_count) {
                    continue;
                }

                const uint32_t inner = static_cast<uint32_t>(rel % array.stride);
                if (inner > array.stride ||
                    access.size > array.stride - inner) {
                    continue;
                }

                MixedStrideKey key{inner, access.size};
                auto& field = raw_groups[key];
                field.inner_offset = inner;
                field.size = access.size;
                if (field.type_access_index < 0 &&
                    !access.inferred_type.empty()) {
                    field.type_access_index = static_cast<int>(i);
                }
                field.access_indices.push_back(static_cast<int>(i));
            }

            qvector<MixedStrideField> raw_fields;
            raw_fields.reserve(raw_groups.size());
            for (auto& [key, field] : raw_groups) {
                raw_fields.push_back(field);
            }

            const size_t stable_fields = count_stable_repeated_fields(
                raw_fields,
                pattern,
                array.base_offset,
                array.stride,
                array.element_count);
            const size_t min_stable_fields = array.base_offset == 0 ? 2u : 1u;
            if (stable_fields < min_stable_fields) {
                continue;
            }
        }

        std::unordered_set<sval_t> member_offsets;
        for (sval_t off : array.member_offsets) {
            member_offsets.insert(off);
        }

        const double ratio = static_cast<double>(member_offsets.size()) /
                             static_cast<double>(array.element_count);
        if (ratio < kMinArrayCoverageRatio) {
            continue;
        }

        size_t elem_size = array.element_type.get_size();
        if (elem_size == BADSIZE || elem_size == 0) {
            elem_size = array.stride;
        }
        if (elem_size > std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        uint32_t access_size = static_cast<uint32_t>(elem_size);
        if (array.needs_element_struct && array.inner_access_size > 0) {
            access_size = array.inner_access_size;
        }

        bool conflicting_access = false;
        if (!array.needs_element_struct) {
            for (const auto& access : pattern.all_accesses) {
                if (!array.contains_offset(access.offset)) {
                    continue;
                }

                if (member_offsets.count(access.offset) > 0 && access.size == access_size) {
                    continue;
                }

                conflicting_access = true;
                break;
            }
        }

        if (conflicting_access) {
            continue;
        }

        if (overlaps_struct_array_prefix(array, arrays)) {
            continue;
        }

        FieldCandidate array_candidate;
        array_candidate.offset = array.base_offset;
        const auto total_size = array.checked_total_size();
        if (!total_size || array.element_count > config_.max_array_elements) {
            continue;
        }
        array_candidate.size = *total_size;
        array_candidate.kind = FieldCandidate::Kind::ArrayField;
        array_candidate.type_category = ctx_.type_encoder().categorize(array.element_type);
        array_candidate.extended_type = ctx_.type_encoder().extract_extended_info(array.element_type);
        array_candidate.array_element_count = array.element_count;
        array_candidate.array_stride = array.stride;
        array_candidate.confidence = array.confidence;

        for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
            const auto& access = pattern.all_accesses[i];
            if (member_offsets.count(access.offset) == 0) {
                continue;
            }
            if (access.size == access_size) {
                array_candidate.source_access_indices.push_back(static_cast<int>(i));
            }
        }

        for (sval_t off : array.member_offsets) {
            uint64_t key = (static_cast<uint64_t>(off) << 32) |
                           static_cast<uint64_t>(access_size);
            auto it = direct_index.find(key);
            if (it != direct_index.end()) {
                candidates[it->second].kind = FieldCandidate::Kind::ArrayElement;
            }
        }

        if (!array.needs_element_struct &&
            has_mixed_scalar_array_semantics(pattern, array_candidate.source_access_indices)) {
            continue;
        }

        candidates.push_back(std::move(array_candidate));
    }

    // Explicit repeated-anchor detection for arrays-of-structs. Start from a
    // repeated same-size anchor field (e.g. element.kind at offsets 8,20,32)
    // and then try to build a mixed-layout element struct around that stride.
    std::unordered_map<uint32_t, qvector<sval_t>> offsets_by_size;
    for (const auto& access : pattern.all_accesses) {
        if (access.size >= 2 && access.size <= 8) {
            offsets_by_size[access.size].push_back(access.offset);
        }
    }

    auto array_candidate_exists = [&](const FieldCandidate& candidate) {
        return std::any_of(candidates.begin(), candidates.end(), [&](const FieldCandidate& existing) {
            return existing.kind == FieldCandidate::Kind::ArrayField &&
                   existing.offset == candidate.offset &&
                   existing.size == candidate.size;
        });
    };

    for (auto& [size, offsets_for_size] : offsets_by_size) {
        std::sort(offsets_for_size.begin(), offsets_for_size.end());
        offsets_for_size.erase(std::unique(offsets_for_size.begin(), offsets_for_size.end()), offsets_for_size.end());

        if (offsets_for_size.size() < static_cast<size_t>(config_.min_array_elements)) {
            continue;
        }

        for (size_t i = 0; i + 2 < offsets_for_size.size(); ++i) {
            for (size_t j = i + 1; j < offsets_for_size.size() && j <= i + 8; ++j) {
                const uint32_t stride = static_cast<uint32_t>(offsets_for_size[j] - offsets_for_size[i]);
                if (stride < size || stride > 64) {
                    continue;
                }

                uint32_t count = 0;
                sval_t expected = offsets_for_size[i];
                while (std::binary_search(offsets_for_size.begin(), offsets_for_size.end(), expected)) {
                    ++count;
                    const auto next = checked_interval_end(expected, stride);
                    if (!next) {
                        break;
                    }
                    expected = *next;
                }

                if (count >= static_cast<uint32_t>(config_.min_array_elements)) {
                    if (count > config_.max_array_elements) {
                        continue;
                    }
                    auto candidate = build_struct_array_candidate(
                        ctx_, pattern, offsets_for_size[i], stride, count);
                    if (candidate.has_value() &&
                        struct_array_depends_on_mixed_size_collapse(pattern, *candidate) &&
                        overlaps_scalar_array_field(*candidate, candidates)) {
                        continue;
                    }
                    if (candidate.has_value() && !array_candidate_exists(*candidate)) {
                        candidates.push_back(std::move(*candidate));
                    }
                }
            }
        }
    }

    // Mixed-size repeated-stride detection for arrays of structs.
    if (pattern.all_accesses.size() >= 6) {
        std::unordered_set<uint32_t> stride_candidates;
        qvector<sval_t> offsets;
        offsets.reserve(pattern.all_accesses.size());
        for (const auto& access : pattern.all_accesses) {
            offsets.push_back(access.offset);
        }
        std::sort(offsets.begin(), offsets.end());
        const size_t max_lookahead = std::min<std::size_t>(offsets.size(), 16);
        for (size_t i = 0; i < offsets.size(); ++i) {
            for (size_t j = i + 1; j < offsets.size() && j <= i + max_lookahead; ++j) {
                sval_t diff = offsets[j] - offsets[i];
                if (diff >= 8 && diff <= 64) {
                    stride_candidates.insert(static_cast<uint32_t>(diff));
                }
            }
        }

        for (uint32_t stride : stride_candidates) {
            for (const auto& access : pattern.all_accesses) {
                std::unordered_set<uint32_t> indices;
                for (const auto& other : pattern.all_accesses) {
                    if (other.offset < access.offset) {
                        continue;
                    }
                    sval_t rel = other.offset - access.offset;
                    if (rel >= 0 && rel % stride == 0) {
                        indices.insert(static_cast<uint32_t>(rel / stride));
                    }
                }

                uint32_t count = 0;
                while (indices.count(count) > 0) {
                    ++count;
                }

                if (count > config_.max_array_elements) {
                    continue;
                }

                auto candidate = build_struct_array_candidate(ctx_, pattern, access.offset, stride, count);
                if (!candidate.has_value()) {
                    continue;
                }

                if (struct_array_depends_on_mixed_size_collapse(pattern, *candidate) &&
                    overlaps_scalar_array_field(*candidate, candidates)) {
                    continue;
                }

                if (!array_candidate_exists(*candidate)) {
                    candidates.push_back(std::move(*candidate));
                }
            }
        }
    }

    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const FieldCandidate& candidate) {
        if (candidate.kind != FieldCandidate::Kind::ArrayField ||
            candidate.type_category != TypeCategory::Struct ||
            !candidate.array_stride.has_value()) {
            return false;
        }

        const uint32_t stride = *candidate.array_stride;
        if (stride == 0) {
            return false;
        }

        return std::any_of(candidates.begin(), candidates.end(), [&](const FieldCandidate& other) {
            if (&other == &candidate ||
                other.kind != FieldCandidate::Kind::ArrayField ||
                other.type_category != TypeCategory::Struct ||
                !other.array_stride.has_value()) {
                return false;
            }

            if (other.size != candidate.size || *other.array_stride != stride) {
                return false;
            }

            if (other.offset <= candidate.offset ||
                other.offset >= candidate.offset + static_cast<sval_t>(stride)) {
                return false;
            }

            return other.source_access_indices.size() >= candidate.source_access_indices.size();
        });
    }), candidates.end());

    // Direct contiguous byte-run detection for compact tails such as
    // checksum arrays. Run this after struct-array candidates have been added
    // so byte tails do not swallow the last byte of a packed element.
    qvector<sval_t> byte_offsets;
    for (const auto& access : pattern.all_accesses) {
        if (access.size != 1) {
            continue;
        }

        const bool consumed_by_struct_array = std::any_of(candidates.begin(), candidates.end(),
            [&](const FieldCandidate& candidate) {
                return candidate.kind == FieldCandidate::Kind::ArrayField &&
                       candidate.type_category == TypeCategory::Struct &&
                       access.offset >= candidate.offset &&
                       access.offset < candidate.end_offset();
            });
        if (!consumed_by_struct_array) {
            byte_offsets.push_back(access.offset);
        }
    }

    std::sort(byte_offsets.begin(), byte_offsets.end());
    byte_offsets.erase(std::unique(byte_offsets.begin(), byte_offsets.end()), byte_offsets.end());

    size_t run_start = 0;
    while (run_start < byte_offsets.size()) {
        size_t run_end = run_start + 1;
        while (run_end < byte_offsets.size() && byte_offsets[run_end] == byte_offsets[run_end - 1] + 1) {
            ++run_end;
        }

        const size_t run_len = run_end - run_start;
        const uint32_t byte_tail_min =
            config_.min_array_elements > 2 ? 2 : config_.min_array_elements;
        if (run_len >= static_cast<size_t>(byte_tail_min) &&
            run_len <= config_.max_array_elements) {
            FieldCandidate byte_array;
            byte_array.offset = byte_offsets[run_start];
            byte_array.size = static_cast<uint32_t>(run_len);
            byte_array.kind = FieldCandidate::Kind::ArrayField;
            byte_array.type_category = TypeCategory::UInt8;
            byte_array.array_element_count = static_cast<uint32_t>(run_len);
            byte_array.array_stride = 1;
            byte_array.confidence = TypeConfidence::Medium;

            for (size_t i = 0; i < pattern.all_accesses.size(); ++i) {
                const auto& access = pattern.all_accesses[i];
                if (access.size == 1 &&
                    access.offset >= byte_array.offset &&
                    access.offset < byte_array.offset + static_cast<sval_t>(byte_array.size)) {
                    byte_array.source_access_indices.push_back(static_cast<int>(i));
                }
            }

            bool duplicate = std::any_of(candidates.begin(), candidates.end(), [&](const FieldCandidate& existing) {
                return existing.kind == FieldCandidate::Kind::ArrayField &&
                       existing.offset == byte_array.offset &&
                       existing.size == byte_array.size;
            });
            if (!duplicate) {
                candidates.push_back(std::move(byte_array));
            }
        }

        run_start = run_end;
    }
}

void FieldCandidateGenerator::generate_padding_candidates(
    const qvector<FieldCandidate>& existing_candidates,
    sval_t struct_end,
    qvector<FieldCandidate>& candidates)
{
    if (existing_candidates.empty()) return;

    // Get non-overlapping coverage ranges
    qvector<std::pair<sval_t, sval_t>> ranges;  // (start, end)

    for (const auto& c : existing_candidates) {
        if (c.kind == FieldCandidate::Kind::ArrayElement) {
            continue;  // Skip array elements (covered by ArrayField)
        }
        ranges.push_back({c.offset, c.end_offset()});
    }

    if (ranges.empty()) return;

    // Sort by start offset
    std::sort(ranges.begin(), ranges.end());

    // Merge overlapping ranges
    qvector<std::pair<sval_t, sval_t>> merged;
    merged.push_back(ranges[0]);

    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, ranges[i].second);
        } else {
            merged.push_back(ranges[i]);
        }
    }

    // Find gaps
    sval_t current_pos = 0;

    for (const auto& [start, end] : merged) {
        if (start > current_pos) {
            // Gap found - create padding
            FieldCandidate padding;
            padding.offset = current_pos;
            padding.size = static_cast<uint32_t>(start - current_pos);
            padding.kind = FieldCandidate::Kind::PaddingField;
            padding.type_category = TypeCategory::RawBytes;
            padding.confidence = TypeConfidence::Low;

            candidates.push_back(std::move(padding));
        }
        current_pos = std::max(current_pos, end);
    }

    // Final padding to struct end
    if (struct_end > current_pos) {
        FieldCandidate padding;
        padding.offset = current_pos;
        padding.size = static_cast<uint32_t>(struct_end - current_pos);
        padding.kind = FieldCandidate::Kind::PaddingField;
        padding.type_category = TypeCategory::RawBytes;
        padding.confidence = TypeConfidence::Low;

        candidates.push_back(std::move(padding));
    }
}

void FieldCandidateGenerator::finalize_candidates(qvector<FieldCandidate>& candidates) {
    for (auto& candidate : candidates) {
        std::sort(candidate.source_access_indices.begin(),
                  candidate.source_access_indices.end());
        candidate.source_access_indices.erase(
            std::unique(candidate.source_access_indices.begin(),
                        candidate.source_access_indices.end()),
            candidate.source_access_indices.end());
    }

    // Canonicalize candidate IDs after generation, including candidates
    // emitted through array-detection hash tables.  For an equal range, retain
    // evidence-backed candidates first, then order by the recovered type and
    // source-function provenance before generated shape.
    std::sort(candidates.begin(), candidates.end(),
        [](const FieldCandidate& a, const FieldCandidate& b) {
            if (a.offset != b.offset) return a.offset < b.offset;
            if (a.size != b.size) return a.size < b.size;

            const bool a_has_evidence = !a.source_access_indices.empty();
            const bool b_has_evidence = !b.source_access_indices.empty();
            if (a_has_evidence != b_has_evidence) return a_has_evidence;
            if (a.type_category != b.type_category) {
                return a.type_category < b.type_category;
            }
            if (a.extended_type.category != b.extended_type.category) {
                return a.extended_type.category < b.extended_type.category;
            }
            if (a.primary_func_ea != b.primary_func_ea) {
                return a.primary_func_ea < b.primary_func_ea;
            }
            if (a.kind != b.kind) return a.kind < b.kind;
            if (a.confidence != b.confidence) return a.confidence > b.confidence;
            if (a.array_element_count != b.array_element_count) {
                return a.array_element_count < b.array_element_count;
            }
            if (a.array_stride != b.array_stride) {
                return a.array_stride < b.array_stride;
            }
            if (a.extended_type.size != b.extended_type.size) {
                return a.extended_type.size < b.extended_type.size;
            }
            if (a.extended_type.pointee_category != b.extended_type.pointee_category) {
                return a.extended_type.pointee_category < b.extended_type.pointee_category;
            }
            if (a.extended_type.element_category != b.extended_type.element_category) {
                return a.extended_type.element_category < b.extended_type.element_category;
            }
            if (a.extended_type.element_count != b.extended_type.element_count) {
                return a.extended_type.element_count < b.extended_type.element_count;
            }
            if (a.extended_type.func_arg_count != b.extended_type.func_arg_count) {
                return a.extended_type.func_arg_count < b.extended_type.func_arg_count;
            }
            if (a.extended_type.udt_tid != b.extended_type.udt_tid) {
                return a.extended_type.udt_tid < b.extended_type.udt_tid;
            }
            return std::lexicographical_compare(
                a.source_access_indices.begin(), a.source_access_indices.end(),
                b.source_access_indices.begin(), b.source_access_indices.end());
        });

    // Assign IDs
    for (size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].id = static_cast<int>(i);
    }
}

TypeCategory FieldCandidateGenerator::infer_category(const FieldAccess& access) const {
    // Prefer explicit function pointer types from inference
    if (!access.inferred_type.empty()) {
        TypeCategory inferred = ctx_.type_encoder().categorize(access.inferred_type);
        if (inferred == TypeCategory::FuncPtr) {
            return inferred;
        }
    }

    // First check semantic type
    switch (access.semantic_type) {
        case SemanticType::Pointer:
            return TypeCategory::Pointer;
        case SemanticType::FunctionPointer:
        case SemanticType::VTablePointer:
            return TypeCategory::FuncPtr;
        case SemanticType::Float:
            return TypeCategory::Float32;
        case SemanticType::Double:
            return TypeCategory::Float64;
        default:
            break;
    }

    // Then check inferred type
    if (!access.inferred_type.empty()) {
        return ctx_.type_encoder().categorize(access.inferred_type);
    }

    // Fall back to size-based inference
    switch (access.size) {
        case 1:
            return TypeCategory::UInt8;
        case 2:
            return TypeCategory::UInt16;
        case 4:
            return TypeCategory::UInt32;
        case 8:
            // Could be uint64 or pointer
            if (get_ptr_size() == 8) {
                return TypeCategory::Pointer;  // Conservative assumption
            }
            return TypeCategory::UInt64;
        default:
            return TypeCategory::RawBytes;
    }
}

FieldCandidate FieldCandidateGenerator::create_from_access(
    const FieldAccess& access,
    int access_index)
{
    FieldCandidate candidate;
    candidate.offset = access.offset;
    candidate.size = access.size;
    candidate.kind = FieldCandidate::Kind::DirectAccess;
    candidate.type_category = infer_category(access);
    candidate.source_access_indices.push_back(access_index);
    candidate.primary_func_ea = access.source_func_ea;

    // Extract extended type info if available
    if (!access.inferred_type.empty()) {
        candidate.extended_type = ctx_.type_encoder().extract_extended_info(access.inferred_type);
    } else {
        candidate.extended_type.category = candidate.type_category;
        candidate.extended_type.size = access.size;
    }

    // Set confidence based on access type
    if (access.semantic_type != SemanticType::Unknown) {
        candidate.confidence = TypeConfidence::Medium;
    } else {
        candidate.confidence = TypeConfidence::Low;
    }

    return candidate;
}

qvector<qvector<int>> FieldCandidateGenerator::find_array_patterns(
    const qvector<FieldCandidate>& candidates) const
{
    qvector<qvector<int>> result;

    // Group candidates by size and type
    std::unordered_map<uint64_t, qvector<int>> size_type_groups;

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];

        // Skip non-direct-access candidates
        if (c.kind != FieldCandidate::Kind::DirectAccess) continue;

        // Key: (size, type_category)
        uint64_t key = (static_cast<uint64_t>(c.size) << 32) |
                       static_cast<uint64_t>(c.type_category);
        size_type_groups[key].push_back(static_cast<int>(i));
    }

    // For each group, check if offsets form arithmetic progression
    for (const auto& [key, indices] : size_type_groups) {
        if (indices.size() < config_.min_array_elements) continue;

        // Extract offsets
        qvector<std::pair<sval_t, int>> offset_idx;
        for (int idx : indices) {
            offset_idx.push_back({candidates[idx].offset, idx});
        }

        // Sort by offset
        std::sort(offset_idx.begin(), offset_idx.end());

        // Find longest arithmetic progression subsequence
        uint32_t size = static_cast<uint32_t>(key >> 32);
        qvector<int> current_group;

        for (size_t i = 0; i < offset_idx.size(); ++i) {
            if (current_group.empty()) {
                current_group.push_back(offset_idx[i].second);
                continue;
            }

            // Check if this extends current progression
            const auto expected_offset = checked_interval_end(
                candidates[current_group.back()].offset, size);
            if (expected_offset && offset_idx[i].first == *expected_offset) {
                current_group.push_back(offset_idx[i].second);
            } else {
                // Break in progression
                if (current_group.size() >= config_.min_array_elements) {
                    result.push_back(current_group);
                }
                current_group.clear();
                current_group.push_back(offset_idx[i].second);
            }
        }

        // Don't forget the last group
        if (current_group.size() >= config_.min_array_elements) {
            result.push_back(current_group);
        }
    }

    return result;
}

bool FieldCandidateGenerator::is_arithmetic_progression(
    const qvector<sval_t>& offsets,
    uint32_t expected_stride) const
{
    if (offsets.size() < 2) return true;

    for (size_t i = 1; i < offsets.size(); ++i) {
        const auto actual_stride = checked_interval_span(
            offsets[i - 1], offsets[i]);
        if (!actual_stride || *actual_stride != expected_stride) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// OverlapAnalysis Implementation
// ============================================================================

OverlapAnalysis FieldCandidateGenerator::analyze_overlaps(
    const qvector<FieldCandidate>& candidates) const
{
    OverlapAnalysis result;

    // OPTIMIZATION: Use sweep line for large sets, O(n²) for small sets
    const size_t n = candidates.size();
    
    if (n >= 64) {
        // Use sweep line algorithm - O(n log n + k)
        std::vector<algorithms::Interval> intervals;
        intervals.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            const auto end = candidates[i].checked_end_offset();
            if (!end) {
                continue;
            }
            intervals.emplace_back(
                candidates[i].offset,
                *end,
                static_cast<int32_t>(candidates[i].id)
            );
        }
        
        auto overlapping = algorithms::find_overlapping_pairs(intervals);
        
        for (const auto& [id1, id2] : overlapping) {
            result.overlapping_pairs.push_back({id1, id2});
        }
    } else {
        // Use O(n²) for small sets - lower constant factors
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (candidates[i].overlaps(candidates[j])) {
                    result.overlapping_pairs.push_back({
                        candidates[i].id,
                        candidates[j].id
                    });
                }
            }
        }
    }

    if (result.overlapping_pairs.empty()) {
        return result;
    }

    // OPTIMIZATION: Use optimized FlatUnionFind
    FlatUnionFind uf;
    
    // Unite overlapping candidates
    for (const auto& [id1, id2] : result.overlapping_pairs) {
        uf.unite_by_id(id1, id2);
    }

    // Collect groups - use root as key to group overlapping candidates
    std::unordered_map<size_t, qvector<int>> groups;
    for (const auto& [id1, id2] : result.overlapping_pairs) {
        size_t root = uf.find_by_id(id1);
        // Both id1 and id2 have same root since they were united
        
        // Track both IDs under the root
        auto& group = groups[root];
        if (std::find(group.begin(), group.end(), id1) == group.end()) {
            group.push_back(id1);
        }
        if (std::find(group.begin(), group.end(), id2) == group.end()) {
            group.push_back(id2);
        }
    }

    for (auto& [root, members] : groups) {
        if (members.size() > 1) {
            std::sort(members.begin(), members.end());
            result.overlap_groups.push_back(std::move(members));
        }
    }

    return result;
}

} // namespace structor::z3
