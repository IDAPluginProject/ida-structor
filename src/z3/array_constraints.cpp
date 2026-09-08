#include "structor/z3/array_constraints.hpp"
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_set>

namespace structor::z3 {

namespace {

bool array_candidate_less(const ArrayCandidate& a, const ArrayCandidate& b,
                          TypeEncoder& encoder) {
    if (a.base_offset != b.base_offset) return a.base_offset < b.base_offset;
    if (a.stride != b.stride) return a.stride < b.stride;
    if (a.element_count != b.element_count) return a.element_count < b.element_count;
    if (a.needs_element_struct != b.needs_element_struct) {
        return a.needs_element_struct < b.needs_element_struct;
    }
    const auto a_category = encoder.categorize(a.element_type);
    const auto b_category = encoder.categorize(b.element_type);
    if (a_category != b_category) return a_category < b_category;
    const auto a_pointee = encoder.categorize(a.element_type.get_pointed_object());
    const auto b_pointee = encoder.categorize(b.element_type.get_pointed_object());
    if (a_pointee != b_pointee) return a_pointee < b_pointee;
    qstring a_decl, b_decl;
    a.element_type.print(&a_decl);
    b.element_type.print(&b_decl);
    return std::string(a_decl.c_str()) < std::string(b_decl.c_str());
}

std::optional<uint32_t> checked_element_count(
    sval_t base,
    sval_t last,
    uint32_t stride,
    uint32_t maximum) {
    if (stride == 0) {
        return std::nullopt;
    }
    const auto span = checked_interval_span(base, last);
    if (!span) {
        return std::nullopt;
    }
    const std::uint64_t count = *span / stride + 1;
    if (count == 0 || count > maximum ||
        count > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(count);
}

std::optional<sval_t> checked_subtract_size(
    sval_t value,
    uint32_t amount) {
    if (value < std::numeric_limits<sval_t>::min() +
                    static_cast<sval_t>(amount)) {
        return std::nullopt;
    }
    return value - static_cast<sval_t>(amount);
}

double coverage_ratio(const qvector<sval_t>& offsets, sval_t base, uint32_t stride) {
    if (offsets.empty() || stride == 0) {
        return 0.0;
    }

    const auto expected = checked_element_count(
        base, offsets.back(), stride, std::numeric_limits<uint32_t>::max());
    if (!expected) {
        return 0.0;
    }

    return static_cast<double>(offsets.size()) / static_cast<double>(*expected);
}

bool has_excessive_gap(const qvector<sval_t>& offsets, uint32_t stride, int max_gap_ratio) {
    if (offsets.size() < 2 || stride == 0) {
        return false;
    }

    const std::uint64_t max_gap =
        static_cast<std::uint64_t>(std::max(1, max_gap_ratio)) * stride;
    for (size_t i = 1; i < offsets.size(); ++i) {
        const auto gap = checked_interval_span(offsets[i - 1], offsets[i]);
        if (!gap || *gap > max_gap) {
            return true;
        }
    }

    return false;
}

bool is_weak_single_field_struct_array(const ArrayCandidate& candidate,
                                       const qvector<const FieldAccess*>& accesses) {
    if (!candidate.needs_element_struct || candidate.element_count >= 4 || accesses.empty()) {
        return false;
    }

    std::unordered_set<uint32_t> inner_offsets;
    for (const auto* access : accesses) {
        if (access->offset < candidate.base_offset) {
            continue;
        }
        const sval_t rel = access->offset - candidate.base_offset;
        inner_offsets.insert(static_cast<uint32_t>(rel % candidate.stride));
    }

    return inner_offsets.size() <= 1 && accesses.size() == candidate.element_count;
}

} // namespace

// ============================================================================
// ArrayConstraintBuilder Implementation
// ============================================================================

ArrayConstraintBuilder::ArrayConstraintBuilder(
    Z3Context& ctx,
    const ArrayDetectionConfig& config)
    : ctx_(ctx)
    , config_(config) {
    config_.min_elements = std::max(config_.min_elements, uint32_t{2});
    config_.max_gap_ratio = std::max(config_.max_gap_ratio, 1);
    config_.max_elements = std::min(
        config_.max_elements, ctx_.config().max_array_elements);
}

qvector<ArrayCandidate> ArrayConstraintBuilder::detect_arrays(
    const qvector<FieldAccess>& accesses)
{
    qvector<ArrayCandidate> candidates;
    stats_ = DetectionStats();

    if (accesses.size() < static_cast<size_t>(config_.min_elements)) {
        return candidates;
    }

    // Group accesses by size
    auto size_groups = group_by_size(accesses);

    struct SizeGroupPlan {
        uint32_t size = 0;
        qvector<const FieldAccess*> group;
        qvector<sval_t> offsets;
        std::optional<std::pair<sval_t, uint32_t>> hinted_result;
        std::optional<std::pair<sval_t, uint32_t>> ap_result;
        bool eligible = false;
    };

    std::vector<SizeGroupPlan> plans;
    plans.reserve(size_groups.size());
    for (auto& [size, group] : size_groups) {
        SizeGroupPlan plan;
        plan.size = size;
        plan.group = group;
        plans.push_back(std::move(plan));
    }
    std::sort(plans.begin(), plans.end(), [](const SizeGroupPlan& a, const SizeGroupPlan& b) {
        return a.size < b.size;
    });

    algorithms::parallel_for_chunks(plans.size(), 2, [&](size_t begin, size_t end) {
        for (size_t plan_idx = begin; plan_idx < end; ++plan_idx) {
            auto& plan = plans[plan_idx];
            if (plan.group.size() < config_.min_elements) {
                continue;
            }

            plan.offsets.reserve(plan.group.size());
            for (const auto* access : plan.group) {
                plan.offsets.push_back(access->offset);
            }
            std::sort(plan.offsets.begin(), plan.offsets.end());
            plan.offsets.erase(std::unique(plan.offsets.begin(), plan.offsets.end()), plan.offsets.end());

            if (plan.offsets.size() < config_.min_elements) {
                continue;
            }

            plan.eligible = true;
            auto stride_hint = extract_stride_hint(plan.group);
            if (stride_hint.has_value()) {
                plan.hinted_result = find_progression_with_stride(plan.offsets, plan.group, *stride_hint);
            }
            plan.ap_result = find_arithmetic_progression(plan.offsets);

        }
    });

    // Materialize candidates serially: this updates stats and may create IDA types.
    for (const auto& plan : plans) {
        if (!plan.eligible) {
            continue;
        }

        const auto append_progression = [&](
            const std::optional<std::pair<sval_t, uint32_t>>& progression) {
            if (!progression) return false;
            const auto [base, stride] = *progression;
            const auto count = checked_element_count(
                base, plan.offsets.back(), stride, config_.max_elements);
            if (!count) return false;
            auto candidate = create_candidate(base, stride, *count, plan.group);
            if (!candidate || is_weak_single_field_struct_array(*candidate, plan.group)) {
                return false;
            }
            candidates.push_back(std::move(*candidate));
            stats_.arrays_found++;
            stats_.elements_covered += static_cast<int>(plan.offsets.size());
            return true;
        };

        // Whole-group hypotheses retain precedence when valid. A failed type
        // or extent hypothesis must not discard independent observed runs.
        if (append_progression(plan.hinted_result) ||
            append_progression(plan.ap_result)) {
            continue;
        }

        if (config_.use_symbolic_indices) {
            auto symbolic_result = detect_symbolic_array(plan.group);
            if (symbolic_result.has_value()) {
                candidates.push_back(std::move(*symbolic_result));
                stats_.arrays_found++;
                stats_.symbolic_detections++;
                continue;
            }
        }

        auto runs = detect_contiguous_typed_runs(plan.group, plan.size);
        for (auto& candidate : runs) {
            stats_.arrays_found++;
            stats_.elements_covered += static_cast<int>(candidate.member_offsets.size());
            candidates.push_back(std::move(candidate));
        }
    }

    // Merge overlapping arrays
    merge_overlapping_arrays(candidates);

    // Sort by base offset
    std::sort(candidates.begin(), candidates.end(),
        [&](const ArrayCandidate& a, const ArrayCandidate& b) {
            return array_candidate_less(a, b, ctx_.type_encoder());
        });

    return candidates;
}

std::unordered_map<uint32_t, qvector<const FieldAccess*>>
ArrayConstraintBuilder::group_by_size(const qvector<FieldAccess>& accesses) {
    std::unordered_map<uint32_t, qvector<const FieldAccess*>> groups;

    for (const auto& access : accesses) {
        groups[access.size].push_back(&access);
    }

    return groups;
}

std::optional<std::pair<sval_t, uint32_t>>
ArrayConstraintBuilder::find_arithmetic_progression(const qvector<sval_t>& offsets) {
    if (offsets.size() < 2) return std::nullopt;

    // Calculate stride as GCD of all differences
    qvector<uint32_t> diffs;
    for (size_t i = 1; i < offsets.size(); ++i) {
        const auto diff = checked_interval_span(offsets[i - 1], offsets[i]);
        if (!diff || *diff == 0 ||
            *diff > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        diffs.push_back(static_cast<uint32_t>(*diff));
    }

    uint32_t stride = gcd_vector(diffs);
    if (stride == 0 || stride > config_.max_stride) {
        return std::nullopt;
    }

    // Verify all offsets fit the pattern: offset[i] = base + i * stride
    sval_t base = offsets[0];

    for (const auto& offset : offsets) {
        const auto relative = checked_interval_span(base, offset);
        if (!relative || *relative % stride != 0) {
            return std::nullopt;  // Doesn't fit the pattern
        }
    }

    // Check for gaps (missing elements)
    const auto expected_count = checked_element_count(
        offsets.front(), offsets.back(), stride, config_.max_elements);
    if (!expected_count) {
        return std::nullopt;
    }

    if (has_excessive_gap(offsets, stride, config_.max_gap_ratio)) {
        return std::nullopt;
    }

    // Allow some gaps based on config
    double ratio = static_cast<double>(offsets.size()) / *expected_count;
    if (ratio < 1.0 / config_.max_gap_ratio) {
        return std::nullopt;  // Too sparse
    }

    return std::make_pair(base, stride);
}

std::optional<uint32_t> ArrayConstraintBuilder::extract_stride_hint(
    const qvector<const FieldAccess*>& accesses) const
{
    std::optional<uint32_t> hint;

    for (const auto* access : accesses) {
        if (!access || !access->array_stride_hint.has_value()) {
            continue;
        }

        uint32_t value = *access->array_stride_hint;
        if (value == 0 || value > config_.max_stride) {
            continue;
        }

        if (!hint.has_value()) {
            hint = value;
        } else if (*hint != value) {
            return std::nullopt;
        }
    }

    return hint;
}

std::optional<std::pair<sval_t, uint32_t>> ArrayConstraintBuilder::find_progression_with_stride(
    const qvector<sval_t>& offsets,
    const qvector<const FieldAccess*>& accesses,
    uint32_t stride_hint) const
{
    if (stride_hint == 0 || stride_hint > config_.max_stride) {
        return std::nullopt;
    }
    if (offsets.size() < 2) {
        return std::nullopt;
    }

    uint32_t inner_offset = 0;
    if (!check_struct_element_pattern(accesses, stride_hint, inner_offset)) {
        sval_t base = offsets.front();
        for (const auto& offset : offsets) {
            const auto relative = checked_interval_span(base, offset);
            if (!relative || *relative % stride_hint != 0) {
                return std::nullopt;
            }
        }
        inner_offset = 0;
    }

    const auto base_value = checked_subtract_size(offsets.front(), inner_offset);
    if (!base_value) {
        return std::nullopt;
    }
    const sval_t base = *base_value;
    for (const auto& offset : offsets) {
        const auto relative = checked_interval_span(base, offset);
        if (!relative || *relative % stride_hint != 0) {
            return std::nullopt;
        }
    }

    const auto expected_count = checked_element_count(
        base, offsets.back(), stride_hint, config_.max_elements);
    if (!expected_count) {
        return std::nullopt;
    }

    if (has_excessive_gap(offsets, stride_hint, config_.max_gap_ratio)) {
        return std::nullopt;
    }

    double ratio = static_cast<double>(offsets.size()) / *expected_count;
    if (ratio < 1.0 / config_.max_gap_ratio) {
        return std::nullopt;
    }

    return std::make_pair(base, stride_hint);
}

bool ArrayConstraintBuilder::verify_type_consistency(
    const qvector<const FieldAccess*>& accesses)
{
    if (accesses.empty()) return true;

    const uint32_t first_size = accesses[0]->size;
    tinfo_t seen_type;

    for (const auto* access : accesses) {
        if (access->size != first_size) {
            return false;
        }

        const tinfo_t type = normalized_element_type(*access);
        if (type.empty()) continue;
        if (type.get_size() != first_size) return false;
        if (!seen_type.empty() && !seen_type.equals_to(type)) return false;
        seen_type = type;
    }

    return true;
}

tinfo_t ArrayConstraintBuilder::normalized_element_type(const FieldAccess& access) {
    if (access.inferred_type.empty()) {
        if (access.semantic_type == SemanticType::Unknown) return tinfo_t();
        return ctx_.type_encoder().decode(
            semantic_to_category(static_cast<int>(access.semantic_type)), access.size);
    }
    if (access.inferred_type.get_size() == access.size) return access.inferred_type;

    // Ctree expression promotion does not change a memory access's width.
    const auto extended = ctx_.type_encoder().extract_extended_info(access.inferred_type);
    auto category = ctx_.type_encoder().categorize(access.inferred_type);
    if (TypeEncoder::is_integer(category) &&
        (access.semantic_type == SemanticType::Integer ||
         access.semantic_type == SemanticType::UnsignedInteger)) {
        // Integer promotion can turn an unsigned narrow load into signed int.
        // Recover its storage signedness from the access semantics; a concrete
        // exact-width type above remains authoritative when views conflict.
        category = semantic_to_category(static_cast<int>(access.semantic_type));
    }
    return ctx_.type_encoder().decode(
        category, access.size, &extended);
}

qvector<ArrayCandidate> ArrayConstraintBuilder::detect_contiguous_typed_runs(
    const qvector<const FieldAccess*>& accesses,
    uint32_t stride)
{
    qvector<ArrayCandidate> result;
    if (stride == 0 || stride > config_.max_stride) return result;

    struct TypeRunGroup {
        tinfo_t type;
        qvector<const FieldAccess*> accesses;
    };
    std::vector<TypeRunGroup> groups;
    for (const auto* access : accesses) {
        const tinfo_t type = normalized_element_type(*access);
        auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& item) {
            return !config_.require_consistent_types ||
                (type.empty() ? item.type.empty()
                              : !item.type.empty() && type.equals_to(item.type));
        });
        if (group == groups.end()) {
            groups.push_back(TypeRunGroup{type, {}});
            group = std::prev(groups.end());
        }
        group->accesses.push_back(access);
    }

    for (auto& group : groups) {
        if (group.accesses.size() < config_.min_elements) continue;
        std::sort(group.accesses.begin(), group.accesses.end(), [](const auto* a, const auto* b) {
            return a->offset < b->offset;
        });
        size_t begin = 0;
        while (begin < group.accesses.size()) {
            size_t end = begin + 1;
            uint32_t count = 1;
            while (end < group.accesses.size()) {
                const auto gap = checked_interval_span(
                    group.accesses[end - 1]->offset, group.accesses[end]->offset);
                if (!gap || (*gap != 0 && *gap != stride)) break;
                count += *gap != 0;
                ++end;
            }
            if (count >= config_.min_elements && count <= config_.max_elements) {
                qvector<const FieldAccess*> run;
                run.reserve(end - begin);
                for (size_t i = begin; i < end; ++i) run.push_back(group.accesses[i]);
                auto candidate = create_candidate(run.front()->offset, stride, count, run);
                if (candidate) result.push_back(std::move(*candidate));
            }
            begin = end;
        }
    }
    return result;
}

void ArrayConstraintBuilder::merge_overlapping_arrays(qvector<ArrayCandidate>& candidates) {
    if (candidates.size() <= 1) return;

    // Sort by base offset
    std::sort(candidates.begin(), candidates.end(),
        [&](const ArrayCandidate& a, const ArrayCandidate& b) {
            return array_candidate_less(a, b, ctx_.type_encoder());
        });

    qvector<ArrayCandidate> merged;
    merged.reserve(candidates.size());  // Reserve to avoid reallocations
    merged.push_back(candidates[0]);

    for (size_t i = 1; i < candidates.size(); ++i) {
        ArrayCandidate& last = merged.back();
        const ArrayCandidate& curr = candidates[i];

        // Check for overlap
        const auto last_end = last.checked_end_offset();
        const auto curr_end = curr.checked_end_offset();

        const auto origin_delta = checked_interval_span(last.base_offset, curr.base_offset);
        if (last_end && curr_end && origin_delta && curr.base_offset < *last_end &&
            curr.stride == last.stride && last.stride != 0 &&
            *origin_delta % last.stride == 0 &&
            curr.needs_element_struct == last.needs_element_struct &&
            curr.inner_access_offset == last.inner_access_offset &&
            curr.inner_access_size == last.inner_access_size &&
            curr.element_type.equals_to(last.element_type)) {
            // Merge: extend the last array
            const sval_t new_end = std::max(*last_end, *curr_end);
            const auto merged_span = checked_interval_span(
                last.base_offset, new_end);
            if (!merged_span ||
                *merged_span > std::numeric_limits<uint32_t>::max() ||
                *merged_span % last.stride != 0 ||
                *merged_span / last.stride > config_.max_elements) {
                merged.push_back(curr);
                continue;
            }
            last.element_count = static_cast<uint32_t>(
                *merged_span / last.stride);

            // Merge member offsets using hash-based deduplication
            // Build hash set of existing offsets for O(1) lookup
            std::unordered_set<sval_t> existing_offsets(
                last.member_offsets.begin(), 
                last.member_offsets.end()
            );
            
            // Add new offsets that don't exist
            for (sval_t off : curr.member_offsets) {
                if (existing_offsets.insert(off).second) {
                    last.member_offsets.push_back(off);
                }
            }
            std::sort(last.member_offsets.begin(), last.member_offsets.end());
            for (const auto& evidence : curr.member_evidence) {
                last.member_evidence.push_back(evidence);
            }
            std::stable_sort(last.member_evidence.begin(), last.member_evidence.end(),
                [](const auto& a, const auto& b) { return a.offset < b.offset; });
        } else {
            // No overlap, keep separate
            merged.push_back(curr);
        }
    }

    candidates = std::move(merged);
}

std::optional<ArrayCandidate> ArrayConstraintBuilder::create_candidate(
    sval_t base,
    uint32_t stride,
    uint32_t count,
    const qvector<const FieldAccess*>& accesses)
{
    if (accesses.empty() || stride == 0 || stride > config_.max_stride ||
        count < config_.min_elements || count > config_.max_elements) {
        return std::nullopt;
    }

    ArrayCandidate candidate;
    candidate.base_offset = base;
    candidate.stride = stride;
    candidate.element_count = count;

    const auto candidate_end = candidate.checked_end_offset();
    if (!candidate_end) {
        return std::nullopt;
    }

    // Each observed access must fit completely within one element and within
    // the emitted array. Deduplicate observation addresses before applying the
    // evidence threshold: repeated reads do not imply additional elements.
    std::unordered_set<sval_t> observed_offsets;
    for (const auto* access : accesses) {
        if (!access || access->size == 0 || access->size > stride) {
            return std::nullopt;
        }
        const auto relative = checked_interval_span(base, access->offset);
        const auto access_end = checked_interval_end(access->offset, access->size);
        if (!relative || *relative % stride != 0 ||
            !access_end || *access_end > *candidate_end) {
            return std::nullopt;
        }
        observed_offsets.insert(access->offset);
    }
    if (observed_offsets.size() < config_.min_elements ||
        (config_.require_consistent_types && !verify_type_consistency(accesses))) {
        return std::nullopt;
    }

    for (const auto offset : observed_offsets) candidate.member_offsets.push_back(offset);
    std::sort(candidate.member_offsets.begin(), candidate.member_offsets.end());
    for (const auto* access : accesses) {
        candidate.member_evidence.push_back(ArrayAccessEvidence{
            access->offset, access->size, access->inferred_type, access->semantic_type});
    }
    std::stable_sort(candidate.member_evidence.begin(), candidate.member_evidence.end(),
        [](const auto& a, const auto& b) { return a.offset < b.offset; });

    // Prefer actual type evidence when the first observation was untyped.
    if (!accesses.empty()) {
        const FieldAccess* first = accesses[0];
        for (const auto* access : accesses) {
            const tinfo_t type = normalized_element_type(*access);
            if (!type.empty()) {
                first = access;
                candidate.element_type = type;
                break;
            }
        }
        if (candidate.element_type.empty()) {
            candidate.element_type = ctx_.type_encoder().decode(
                TypeCategory::Unknown,
                first->size);
        }

        // Check if stride > access_size (struct element pattern)
        if (stride > first->size) {
            candidate.needs_element_struct = true;
            const auto relative = checked_interval_span(base, first->offset);
            if (!relative) {
                return std::nullopt;
            }
            candidate.inner_access_offset = static_cast<uint32_t>(
                *relative % stride);
            candidate.inner_access_size = first->size;
            stats_.struct_element_arrays++;

            // Create synthetic element type
            if (config_.detect_arrays_of_structs) {
                candidate.element_type = create_element_struct_type(
                    stride,
                    candidate.inner_access_offset,
                    candidate.element_type
                );
            }
        }
    }

    if (!candidate.is_valid_c_array()) {
        return std::nullopt;
    }

    // Determine confidence
    if (count >= 5 && !candidate.needs_element_struct) {
        candidate.confidence = TypeConfidence::High;
    } else if (count >= 3) {
        candidate.confidence = TypeConfidence::Medium;
    } else {
        candidate.confidence = TypeConfidence::Low;
    }

    return candidate;
}

std::optional<ArrayCandidate> ArrayConstraintBuilder::detect_symbolic_array(
    const qvector<const FieldAccess*>& accesses)
{
    if (accesses.size() < static_cast<size_t>(config_.min_elements)) {
        return std::nullopt;
    }

    // Use Z3 to solve for: offset[i] = base + index[i] * stride
    // Variables: base (Int), stride (Int)
    // For each access, we want: (access.offset - base) % stride == inner_offset

    return solve_stride_z3(accesses);
}

std::optional<ArrayCandidate> ArrayConstraintBuilder::solve_stride_z3(
    const qvector<const FieldAccess*>& accesses)
{
    if (accesses.empty()) return std::nullopt;

    // Extract offsets
    qvector<sval_t> offsets;
    for (const auto* access : accesses) {
        if (!access) return std::nullopt;
        offsets.push_back(access->offset);
    }
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    if (offsets.size() < config_.min_elements) {
        return std::nullopt;
    }

    auto stride_hint = extract_stride_hint(accesses);
    if (stride_hint.has_value()) {
        auto hinted = find_progression_with_stride(offsets, accesses, *stride_hint);
        if (hinted.has_value()) {
            auto [base, stride] = *hinted;
            const auto count = checked_element_count(
                base, offsets.back(), stride, config_.max_elements);
            if (count && *count >= static_cast<uint32_t>(config_.min_elements)) {
                auto candidate = create_candidate(base, stride, *count, accesses);
                if (candidate && !is_weak_single_field_struct_array(*candidate, accesses)) {
                    return candidate;
                }
            }
        }
    }

    // Calculate GCD stride
    uint32_t stride = calculate_gcd_stride(offsets);
    if (stride == 0 || stride > config_.max_stride) {
        return std::nullopt;
    }

    // Try to detect if accesses are at consistent inner offset
    uint32_t inner_offset = 0;
    if (!check_struct_element_pattern(accesses, stride, inner_offset)) {
        // Fall back to stride = access size
        stride = accesses[0]->size;
        inner_offset = 0;
    }

    const auto base_value = checked_subtract_size(offsets[0], inner_offset);
    if (!base_value) {
        return std::nullopt;
    }
    const sval_t base = *base_value;
    const sval_t last = offsets.back();
    const auto count = checked_element_count(
        base, last, stride, config_.max_elements);

    if (!count || *count < static_cast<uint32_t>(config_.min_elements)) {
        return std::nullopt;
    }

    if (has_excessive_gap(offsets, stride, config_.max_gap_ratio)) {
        return std::nullopt;
    }

    if (coverage_ratio(offsets, base, stride) < 1.0 / config_.max_gap_ratio) {
        return std::nullopt;
    }

    auto candidate = create_candidate(base, stride, *count, accesses);
    if (!candidate || is_weak_single_field_struct_array(*candidate, accesses)) {
        return std::nullopt;
    }

    return candidate;
}

uint32_t ArrayConstraintBuilder::calculate_gcd_stride(const qvector<sval_t>& offsets) const {
    if (offsets.size() < 2) return 0;

    qvector<uint32_t> diffs;
    for (size_t i = 1; i < offsets.size(); ++i) {
        const auto diff = checked_interval_span(offsets[i - 1], offsets[i]);
        if (diff && *diff > 0 &&
            *diff <= std::numeric_limits<uint32_t>::max()) {
            diffs.push_back(static_cast<uint32_t>(*diff));
        }
    }

    if (diffs.empty()) return 0;
    return gcd_vector(diffs);
}

bool ArrayConstraintBuilder::check_struct_element_pattern(
    const qvector<const FieldAccess*>& accesses,
    uint32_t stride,
    uint32_t& inner_offset) const
{
    if (accesses.empty() || stride == 0) return false;

    // Calculate inner offset from first access
    sval_t min_offset = accesses[0]->offset;
    for (const auto* access : accesses) {
        min_offset = std::min(min_offset, access->offset);
    }

    const auto first_relative = checked_interval_span(
        min_offset, accesses[0]->offset);
    if (!first_relative) {
        return false;
    }
    uint32_t first_inner = static_cast<uint32_t>(*first_relative % stride);

    // Check all accesses have the same inner offset
    for (const auto* access : accesses) {
        const auto relative = checked_interval_span(min_offset, access->offset);
        if (!relative) {
            return false;
        }
        uint32_t this_inner = static_cast<uint32_t>(*relative % stride);
        if (this_inner != first_inner) {
            return false;
        }
    }

    inner_offset = first_inner;
    return true;
}

tinfo_t ArrayConstraintBuilder::create_element_struct_type(
    uint32_t stride,
    uint32_t inner_offset,
    const tinfo_t& inner_type)
{
    // Create synthetic element struct:
    //   struct __element_N {
    //       char __pad_0[inner_offset];  // if inner_offset > 0
    //       <inner_type> accessed_field;
    //       char __pad_1[stride - inner_offset - inner_type.size()];
    //   };

    const size_t raw_inner_size = inner_type.empty() ? 4 : inner_type.get_size();
    if (stride == 0 || raw_inner_size == BADSIZE ||
        raw_inner_size > std::numeric_limits<uint32_t>::max()) {
        return tinfo_t();
    }
    const uint32_t inner_size = static_cast<uint32_t>(raw_inner_size);
    if (inner_offset > stride || inner_size > stride - inner_offset) {
        return tinfo_t();
    }
    uint32_t trailing_pad = stride - inner_offset - inner_size;

    // Generate unique name
    qstring name;
    name.sprnt("__array_elem_%u_%u", stride, inner_offset);

    // Create struct type
    tinfo_t struct_type;
    udt_type_data_t udt;

    // Leading padding
    if (inner_offset > 0) {
        udm_t pad_field;
        pad_field.name = "__pad_0";
        pad_field.offset = 0;
        pad_field.size = static_cast<std::uint64_t>(inner_offset) * 8;  // bits

        tinfo_t byte_type;
        byte_type.create_simple_type(BT_INT8 | BTMT_CHAR);
        tinfo_t pad_array;
        pad_array.create_array(byte_type, inner_offset);
        pad_field.type = pad_array;

        udt.push_back(pad_field);
    }

    // Accessed field
    udm_t accessed_field;
    accessed_field.name = "value";
    accessed_field.offset = static_cast<std::uint64_t>(inner_offset) * 8;
    accessed_field.size = static_cast<std::uint64_t>(inner_size) * 8;
    accessed_field.type = inner_type.empty() ?
        tinfo_t() : inner_type;

    if (accessed_field.type.empty()) {
        // Default to uint32_t
        accessed_field.type.create_simple_type(BT_INT32 | BTMT_UNSIGNED);
    }

    udt.push_back(accessed_field);

    // Trailing padding
    if (trailing_pad > 0) {
        udm_t trail_field;
        trail_field.name = "__pad_1";
        trail_field.offset =
            static_cast<std::uint64_t>(inner_offset + inner_size) * 8;
        trail_field.size = static_cast<std::uint64_t>(trailing_pad) * 8;

        tinfo_t byte_type;
        byte_type.create_simple_type(BT_INT8 | BTMT_CHAR);
        tinfo_t pad_array;
        pad_array.create_array(byte_type, trailing_pad);
        trail_field.type = pad_array;

        udt.push_back(trail_field);
    }

    // Finalize struct
    udt.total_size = stride;
    udt.pack = 1;  // Packed

    struct_type.create_udt(udt, BTF_STRUCT);

    return struct_type;
}

} // namespace structor::z3
