#include "structor/z3/layout_constraints.hpp"
#include "structor/naming.hpp"
#include "structor/optimized_algorithms.hpp"
#include "structor/optimized_containers.hpp"
#include "structor/simd.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <unordered_set>

#ifndef STRUCTOR_TESTING
#include <pro.h>
#include <kernwin.hpp>
#endif

namespace structor::z3 {

namespace {

ArrayDetectionConfig make_bounded_array_config(
    const LayoutConstraintConfig& config) {
    return make_array_detection_config(
        config.min_array_elements,
        config.max_array_elements,
        config.detect_symbolic_arrays,
        config.max_array_stride);
}

bool is_redundant_aggregate_access(const UnifiedAccessPattern* pattern, const FieldAccess& access) {
    if (!pattern || access.inferred_type.empty()) {
        return false;
    }

    if ((!access.inferred_type.is_array() && !access.inferred_type.is_struct()) || access.size <= 8) {
        return false;
    }

    int nested = 0;
    const auto access_end = checked_interval_end(access.offset, access.size);
    if (!access_end) {
        return false;
    }
    for (const auto& other : pattern->all_accesses) {
        if (&other == &access) {
            continue;
        }
        const auto other_end = checked_interval_end(other.offset, other.size);
        if (other_end && other.offset >= access.offset && *other_end <= *access_end &&
            (other.size < access.size || other.offset != access.offset)) {
            ++nested;
        }
    }

    return nested >= 2;
}

} // namespace

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

    inline int clamp_weight(int value, int min_val, int max_val) {
        return std::max(min_val, std::min(max_val, value));
    }

    inline int access_weight(const FieldCandidate& cand, int base_weight) {
        if (base_weight <= 0) return 0;
        int count = static_cast<int>(cand.source_access_indices.size());
        int multiplier = clamp_weight(count, 1, 10);
        return base_weight * multiplier;
    }

    inline int padding_weight(uint32_t size, int base_weight) {
        if (base_weight <= 0) return 0;
        int multiplier = clamp_weight(static_cast<int>((size + 3) / 4), 1, 10);
        return base_weight * multiplier;
    }

    inline int candidate_specificity_score(const FieldCandidate& cand) {
        int score = 0;

        switch (cand.kind) {
            case FieldCandidate::Kind::ArrayField: score += 40; break;
            case FieldCandidate::Kind::DirectAccess: score += 5; break;
            case FieldCandidate::Kind::ArrayElement: score += 2; break;
            default: break;
        }

        switch (cand.type_category) {
            case TypeCategory::Struct: score += 25; break;
            case TypeCategory::Union: score += 20; break;
            case TypeCategory::Array: score += 15; break;
            case TypeCategory::RawBytes:
            case TypeCategory::Unknown:
                score -= 10;
                break;
            default:
                score += 5;
                break;
        }

        score += clamp_weight(static_cast<int>(cand.source_access_indices.size()), 0, 16) * 3;
        if (cand.array_element_count.has_value()) {
            score += clamp_weight(static_cast<int>(*cand.array_element_count), 0, 8) * 2;
        }

        return score;
    }

    inline bool should_prefer_candidate(const FieldCandidate& preferred, const FieldCandidate& other) {
        const bool compact_byte_array =
            other.kind == FieldCandidate::Kind::ArrayField &&
            other.type_category == TypeCategory::UInt8 &&
            other.array_stride.value_or(0) == 1 &&
            other.size <= 16;
        if (compact_byte_array) {
            return false;
        }

        const bool preferred_struct_array =
            preferred.kind == FieldCandidate::Kind::ArrayField &&
            preferred.type_category == TypeCategory::Struct;
        const bool preferred_scalar_array =
            preferred.kind == FieldCandidate::Kind::ArrayField &&
            preferred.type_category != TypeCategory::Struct;
        const bool other_scalar_array =
            other.kind == FieldCandidate::Kind::ArrayField &&
            other.type_category != TypeCategory::Struct;
        const bool other_struct_array =
            other.kind == FieldCandidate::Kind::ArrayField &&
            other.type_category == TypeCategory::Struct;

        if (preferred_struct_array && other_scalar_array) {
            return false;
        }

        if (preferred_scalar_array && other_struct_array) {
            return true;
        }

        if (!preferred.overlaps(other)) {
            return false;
        }

        if (!(preferred.contains(other) || other.contains(preferred))) {
            return false;
        }

        const int preferred_score = candidate_specificity_score(preferred);
        const int other_score = candidate_specificity_score(other);
        if (preferred_score == other_score) {
            return preferred.offset < other.offset;
        }

        return preferred_score > other_score;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

SynthField field_from_candidate(
    const FieldCandidate& candidate,
    TypeEncoder& type_encoder,
    const qvector<FieldAccess>* access_list)
{
    SynthField field;
    field.offset = candidate.offset;
    field.size = candidate.size;

    SemanticType semantic = static_cast<SemanticType>(category_to_semantic(candidate.type_category));
    if (candidate.kind == FieldCandidate::Kind::ArrayField) {
        semantic = SemanticType::Array;
    } else if (candidate.type_category == TypeCategory::Struct) {
        semantic = SemanticType::NestedStruct;
    }

    // Decode array candidates at element width, not total array width.
    // Otherwise a 4-byte run of four byte accesses becomes uint32_t[4].
    const uint32_t decode_size =
        candidate.kind == FieldCandidate::Kind::ArrayField &&
                candidate.array_stride.has_value()
            ? *candidate.array_stride
            : candidate.size;

    // Decode type
    field.type = type_encoder.decode(
        candidate.type_category,
        decode_size,
        &candidate.extended_type
    );

    // Set semantic type
    if (TypeEncoder::is_integer(candidate.type_category)) {
        semantic = TypeEncoder::is_signed_int(candidate.type_category)
            ? SemanticType::Integer : SemanticType::UnsignedInteger;
    } else if (TypeEncoder::is_floating(candidate.type_category)) {
        semantic = candidate.size == 4 ? SemanticType::Float : SemanticType::Double;
    }

    if (access_list) {
        for (int idx : candidate.source_access_indices) {
            if (idx >= 0 && static_cast<size_t>(idx) < access_list->size()) {
                const FieldAccess& access = access_list->at(static_cast<size_t>(idx));
                field.source_accesses.push_back(access);
                if (semantic != SemanticType::Array &&
                    semantic != SemanticType::NestedStruct &&
                    semantic_priority(access.semantic_type) > semantic_priority(semantic)) {
                    semantic = access.semantic_type;
                }
            }
        }
    }

    field.semantic = semantic;

    if (!field.type.empty() && field.type.is_integral()) {
        if (field.semantic == SemanticType::UnsignedInteger &&
            !field.type.is_unsigned()) {
            (void)field.type.change_sign(type_unsigned);
        } else if (field.semantic == SemanticType::Integer &&
                   field.type.is_unsigned()) {
            (void)field.type.change_sign(type_signed);
        }
    }

    // Handle arrays
    if (candidate.kind == FieldCandidate::Kind::ArrayField &&
        candidate.array_element_count.has_value()) {
        const tinfo_t elem_type = field.type;
        const size_t elem_size = elem_type.get_size();
        tinfo_t array_type;
        array_type.create_array(field.type, *candidate.array_element_count);
        field.type = array_type;
        field.is_array = true;
        field.array_count = *candidate.array_element_count;
        field.semantic = SemanticType::Array;
        const uint32_t stride = candidate.array_stride.value_or(decode_size);
        if (stride != 0 &&
            *candidate.array_element_count <=
                std::numeric_limits<uint32_t>::max() / stride) {
            field.size = stride * *candidate.array_element_count;
        }
        field.name = make_array_field_name(candidate.offset,
                                           elem_type,
                                           semantic,
                                           static_cast<std::uint32_t>(elem_size == BADSIZE ? candidate.size : elem_size));
        field.naming.kind = GeneratedNameKind::ArrayField;
        field.naming.origin = NameOrigin::GeneratedFallback;
        field.naming.confidence = NameConfidence::Medium;
    } else {
        if (field.semantic == SemanticType::NestedStruct) {
            field.name = make_substruct_field_name(candidate.offset);
            field.naming.kind = GeneratedNameKind::SubStructField;
            field.naming.origin = NameOrigin::GeneratedFallback;
            field.naming.confidence = NameConfidence::Medium;
        } else {
            field.name = generate_field_name(candidate.offset, field.semantic, candidate.size);
            field.naming.kind = GeneratedNameKind::Field;
            field.naming.origin = NameOrigin::GeneratedFallback;
            field.naming.confidence = NameConfidence::Medium;
        }
    }

    return field;
}

bool candidates_compatible_for_union(
    const FieldCandidate& a,
    const FieldCandidate& b)
{
    // Must have the same offset
    if (a.offset != b.offset) return false;

    // Size should be the same or one contains the other
    if (a.size != b.size && !a.contains(b) && !b.contains(a)) {
        return false;
    }

    return true;
}

// ============================================================================
// LayoutConstraintBuilder Implementation
// ============================================================================

LayoutConstraintBuilder::LayoutConstraintBuilder(
    Z3Context& ctx,
    const LayoutConstraintConfig& config)
    : ctx_(ctx)
    , config_(config)
    , array_builder_(ctx, make_bounded_array_config(config))
    , constraint_tracker_(ctx.ctx())
    , solver_(ctx.make_solver()) {}

void LayoutConstraintBuilder::build_constraints(
    const UnifiedAccessPattern& pattern,
    const qvector<FieldCandidate>& candidates)
{
    auto start_time = std::chrono::steady_clock::now();

    if (config_.max_accesses == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::Accesses, 0, pattern.all_accesses.size(),
            "constraint_build",
            "configured access-evidence limit is zero");
    }
    if (config_.max_candidates == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::Candidates, 0, candidates.size(),
            "constraint_build",
            "configured field-candidate limit is zero");
    }
    if (config_.max_fields == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::Fields, 0, 0,
            "constraint_build",
            "configured materialized-field limit is zero");
    }
    if (config_.max_array_elements == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::ArrayElements, 0, 0,
            "constraint_build",
            "configured array-inference element limit is zero");
    }
    if (config_.max_struct_size == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::StructureSize, 0, 0,
            "constraint_build",
            "configured structure-size limit is zero");
    }
    if (config_.max_constraint_pairs == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::ConstraintPairs, 0, 0,
            "constraint_build",
            "configured constraint-relation limit is zero");
    }
    if (config_.max_union_alternatives == 0) {
        throw ResourceLimitException(
            ResourceLimitKind::UnionAlternatives, 0, 0,
            "constraint_build",
            "configured per-union alternative limit is zero");
    }
    if (!is_valid_abi_alignment(
            static_cast<std::int64_t>(config_.default_alignment))) {
        throw std::invalid_argument(
            "layout default alignment must be a non-zero power of two representable by SynthOptions::alignment");
    }
    if (pattern.all_accesses.size() > config_.max_accesses) {
        throw ResourceLimitException(
            ResourceLimitKind::Accesses,
            config_.max_accesses,
            pattern.all_accesses.size(),
            "constraint_build",
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
                config_.max_struct_size,
                std::numeric_limits<std::uint64_t>::max(),
                "constraint_build",
                "access evidence interval overflows the signed offset domain");
        }
        evidence_end = std::max(evidence_end, *end);
    }
    const auto evidence_span = checked_interval_span(
        evidence_origin, evidence_end);
    if (!evidence_span || *evidence_span > config_.max_struct_size) {
        throw ResourceLimitException(
            ResourceLimitKind::StructureSize,
            config_.max_struct_size,
            evidence_span.value_or(std::numeric_limits<std::uint64_t>::max()),
            "constraint_build",
            "recovered object span exceeds the configured structure-size limit");
    }

    // Oversized arrays are optional aggregate interpretations.  Omit those
    // candidates before pair construction while retaining direct scalar
    // evidence, which remains mandatory and truthfully coverable.
    qvector<FieldCandidate> bounded_candidates;
    bounded_candidates.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (!candidate.within_array_element_limit(
                config_.max_array_elements)) {
            continue;
        }

        const auto end = candidate.checked_end_offset();
        const auto span = end && candidate.offset >= evidence_origin
            ? checked_interval_span(evidence_origin, *end)
            : std::nullopt;
        if (!span || *span > config_.max_struct_size) {
            throw ResourceLimitException(
                ResourceLimitKind::StructureSize,
                config_.max_struct_size,
                span.value_or(std::numeric_limits<std::uint64_t>::max()),
                "constraint_build",
                "field candidate interval exceeds the configured structure domain");
        }
        bounded_candidates.push_back(candidate);
    }

    if (bounded_candidates.size() > config_.max_candidates) {
        throw ResourceLimitException(
            ResourceLimitKind::Candidates,
            config_.max_candidates,
            bounded_candidates.size(),
            "constraint_build",
            "field candidates exceed the configured solver limit");
    }

    std::vector<UnionAlternativeDescriptor> union_descriptors;
    union_descriptors.reserve(bounded_candidates.size());
    for (const auto& candidate : bounded_candidates) {
        union_descriptors.push_back({
            static_cast<std::int64_t>(candidate.offset),
            candidate.size,
            candidate.kind == FieldCandidate::Kind::UnionAlternative});
    }
    if (config_.allow_unions) {
        const auto largest_union = largest_mandatory_union_cluster(
            union_descriptors);
        if (largest_union.count > config_.max_union_alternatives) {
            throw ResourceLimitException(
                ResourceLimitKind::UnionAlternatives,
                config_.max_union_alternatives,
                largest_union.count,
                "constraint_build",
                "mandatory storage interpretations exceed the per-union alternative limit");
        }
    }

    const uint64_t candidate_count = bounded_candidates.size();
    const auto relation_count = checked_layout_relation_count(
        candidate_count, config_.max_fields, config_.allow_unions);
    if (!relation_count ||
        *relation_count > config_.max_constraint_pairs) {
        throw ResourceLimitException(
            ResourceLimitKind::ConstraintPairs,
            config_.max_constraint_pairs,
            relation_count.value_or(
                std::numeric_limits<std::uint64_t>::max()),
            "constraint_build",
            "candidate-pair and union-cardinality relations exceed the configured constraint limit");
    }

    z3_log("[Structor/Z3] Building constraints for %zu accesses, %zu field candidates\n",
           pattern.all_accesses.size(), bounded_candidates.size());

    pattern_ = &pattern;
    candidates_ = std::move(bounded_candidates);

    // Reset state
    field_vars_.clear();
    arrays_.clear();
    union_resolutions_.clear();
    packing_var_.reset();
    inferred_packing_.reset();
    statistics_ = {};
    solver_.reset();
    constraint_tracker_.clear();

    // Detect arrays first
    if (config_.detect_arrays) {
        arrays_ = array_builder_.detect_arrays(pattern.all_accesses);
    }
    if (!arrays_.empty()) {
        z3_log("[Structor/Z3] Detected %zu potential arrays\n", arrays_.size());
    }

    // Create field variables
    create_field_variables();

    // Add constraints in order of importance
    add_coverage_constraints();      // HARD
    add_size_bound_constraints();    // HARD

    add_non_overlap_constraints();   // SOFT (union option)
    add_alignment_constraints();     // SOFT
    add_type_constraints();          // SOFT
    add_type_preference_constraints(); // SOFT (prefer typed over raw_bytes)
    add_array_constraints();         // SOFT

    // Add optimization objectives
    add_optimization_objectives();

    auto end_time = std::chrono::steady_clock::now();
    statistics_.constraint_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    statistics_.total_constraints = static_cast<unsigned>(constraint_tracker_.total_constraints());
    statistics_.hard_constraints = static_cast<unsigned>(constraint_tracker_.hard_constraint_count());
    statistics_.soft_constraints = static_cast<unsigned>(constraint_tracker_.soft_constraint_count());

    z3_log("[Structor/Z3] Built %u constraints (%u hard, %u soft) in %lldms\n",
           statistics_.total_constraints,
           statistics_.hard_constraints,
           statistics_.soft_constraints,
           static_cast<long long>(statistics_.constraint_build_time.count()));
}

void LayoutConstraintBuilder::create_field_variables() {
    auto& ctx = ctx_.ctx();

    // A C/C++ union's alternatives share one storage origin. Assign a stable
    // group id per candidate offset instead of asking Z3 to discover an
    // arbitrary graph coloring. This removes group-label symmetry and permits
    // linear per-storage cardinality constraints.
    std::map<sval_t, int> union_group_by_offset;
    for (const auto& candidate : candidates_) {
        union_group_by_offset.emplace(candidate.offset, 0);
    }
    int next_union_group = 0;
    for (auto& [offset, group] : union_group_by_offset) {
        (void)offset;
        group = next_union_group++;
    }
    std::vector<CanonicalUnionConstraintVariables> canonical_union_variables;
    canonical_union_variables.reserve(candidates_.size());

    z3_log("[Structor/Z3] Creating field variables for %zu candidates\n", candidates_.size());

    // Create packing variable if needed
    const qvector<uint32_t> packing_options = normalized_packing_options();
    if (config_.model_packing && !packing_options.empty()) {
        packing_var_ = ctx.int_const("__packing");

        // Constrain packing to valid options (hard constraint, not tracked)
        ::z3::expr_vector options(ctx);
        for (uint32_t p : packing_options) {
            options.push_back(*packing_var_ == static_cast<int>(p));
        }

        ConstraintProvenance prov;
        prov.description = "Packing value constraint";
        prov.is_soft = false;
        prov.kind = ConstraintProvenance::Kind::Other;
        constraint_tracker_.add_hard(solver_, ::z3::mk_or(options), prov);
        z3_log("[Structor/Z3]   Added packing constraint with %zu options\n",
               packing_options.size());
    }

    // Create variables for each candidate
    for (size_t i = 0; i < candidates_.size(); ++i) {
        const auto& cand = candidates_[i];

        FieldVariables fv(ctx);
        fv.candidate_id = static_cast<int>(i);

        // Create named variables for this candidate
        qstring prefix;
        prefix.sprnt("f%zu_", i);

        fv.selected = ctx.bool_const((prefix + "sel").c_str());
        fv.offset = ctx_.int_val(static_cast<int64_t>(cand.offset));  // Fixed
        fv.size = ctx_.uint_val(cand.size);                      // Fixed
        fv.type = ctx.int_val(static_cast<int>(cand.type_category));  // Fixed
        fv.is_array = ctx.bool_val(cand.is_array());
        fv.array_count = ctx.int_val(cand.array_element_count.value_or(1));
        fv.is_union_member = ctx.bool_const((prefix + "union").c_str());
        fv.union_group = ctx.int_const((prefix + "ugrp").c_str());

        const int deterministic_union_group =
            union_group_by_offset.at(cand.offset);
        const UnionConstraintVariables union_variables{
            fv.selected, fv.is_union_member, fv.union_group};
        canonical_union_variables.push_back({
            union_variables,
            static_cast<std::uint32_t>(deterministic_union_group)});

        // Selected union alternatives at the same byte offset use one
        // canonical group id. Nonmembers carry the -1 sentinel. This domain
        // also canonicalizes every unselected candidate as a nonmember.
        {
            ConstraintProvenance prov;
            prov.description.sprnt(
                "Canonical union group for field %zu at offset 0x%llX",
                i,
                static_cast<unsigned long long>(cand.offset));
            prov.is_soft = false;
            prov.kind = ConstraintProvenance::Kind::Other;
            constraint_tracker_.add_hard(
                solver_,
                canonical_union_group_domain_constraint(
                    union_variables,
                    static_cast<std::uint32_t>(deterministic_union_group)),
                prov);
        }

        if (!config_.allow_unions) {
            ConstraintProvenance prov;
            prov.description.sprnt("Union membership disabled for field %zu", i);
            prov.is_soft = false;
            prov.kind = ConstraintProvenance::Kind::Other;
            constraint_tracker_.add_hard(solver_, !fv.is_union_member, prov);
        }

        // Soft constraint: prefer NOT being a union member. Evidence-backed
        // alternatives are handled by hard constraints in add_type_constraints;
        // penalizing them here only forces an avoidable relaxation round.
        if (cand.kind != FieldCandidate::Kind::UnionAlternative) {
            int weight = access_weight(cand, config_.weight_prefer_non_union);
            if (weight > 0) {
                ConstraintProvenance prov;
                prov.description.sprnt("Prefer non-union for field %zu", i);
                prov.is_soft = true;
                prov.kind = ConstraintProvenance::Kind::Other;
                prov.weight = weight;
                constraint_tracker_.add_soft(solver_, !fv.is_union_member, prov, weight);
            }
        }

        // Soft constraint: penalize selecting padding fields
        if (cand.kind == FieldCandidate::Kind::PaddingField) {
            int weight = padding_weight(cand.size, config_.weight_minimize_padding);
            if (weight > 0) {
                ConstraintProvenance prov;
                prov.description.sprnt("Penalize padding at 0x%llX", static_cast<unsigned long long>(cand.offset));
                prov.is_soft = true;
                prov.kind = ConstraintProvenance::Kind::Other;
                prov.weight = weight;
                constraint_tracker_.add_soft(solver_, !fv.selected, prov, weight);
            }
        }

        field_vars_.push_back(fv);
    }

    if (config_.allow_unions && !field_vars_.empty()) {
        const auto cardinality_constraints =
            canonical_union_alternative_limit_constraints(
                ctx,
                canonical_union_variables,
                static_cast<std::uint32_t>(union_group_by_offset.size()),
                config_.max_union_alternatives);
        for (size_t group = 0;
             group < cardinality_constraints.size(); ++group) {
            ConstraintProvenance prov;
            prov.description.sprnt(
                "Storage union group %zu has at most %u selected alternatives",
                group,
                config_.max_union_alternatives);
            prov.is_soft = false;
            prov.kind = ConstraintProvenance::Kind::Other;
            constraint_tracker_.add_hard(
                solver_,
                cardinality_constraints[group],
                prov);
        }
    }
}

void LayoutConstraintBuilder::add_coverage_constraints() {
    auto& ctx = ctx_.ctx();

    z3_log("[Structor/Z3] Adding coverage constraints for %zu accesses\n", pattern_->all_accesses.size());
    int uncovered_count = 0;

    // Pre-compute candidate bounds for faster coverage checking
    const size_t num_candidates = candidates_.size();
    const size_t num_accesses = pattern_->all_accesses.size();
    std::vector<TypeCategory> access_categories(num_accesses, TypeCategory::Unknown);
    for (size_t i = 0; i < num_accesses; ++i) {
        const auto& access = pattern_->all_accesses[i];
        if (!access.inferred_type.empty()) {
            access_categories[i] = ctx_.type_encoder().categorize(access.inferred_type);
        } else {
            access_categories[i] = semantic_to_category(static_cast<int>(access.semantic_type));
        }
    }

    std::vector<uint8_t> redundant_accesses(num_accesses, 0);
    for (size_t i = 0; i < num_accesses; ++i) {
        redundant_accesses[i] = is_redundant_aggregate_access(pattern_, pattern_->all_accesses[i]) ? 1 : 0;
    }

    std::vector<std::vector<int32_t>> coverage_map(num_accesses);
    const auto is_padding_like_candidate = [](const FieldCandidate& cand) {
        return cand.kind == FieldCandidate::Kind::PaddingField ||
               cand.type_category == TypeCategory::RawBytes;
    };

    auto covers_by_shape = [&](const FieldCandidate& cand,
                               const FieldAccess& access,
                               TypeCategory access_cat,
                               size_t access_index) {
        if (cand.kind == FieldCandidate::Kind::ArrayField &&
            cand.type_category != TypeCategory::Struct &&
            std::find(cand.source_access_indices.begin(), cand.source_access_indices.end(),
                      static_cast<int>(access_index)) == cand.source_access_indices.end()) {
            return false;
        }
        if ((cand.type_category == TypeCategory::Array ||
             cand.type_category == TypeCategory::Struct ||
             cand.type_category == TypeCategory::Union) &&
            cand.kind == FieldCandidate::Kind::DirectAccess) {
            return cand.offset == access.offset && cand.size == access.size;
        }

        if ((cand.kind == FieldCandidate::Kind::DirectAccess ||
             cand.kind == FieldCandidate::Kind::ArrayElement ||
             cand.kind == FieldCandidate::Kind::UnionAlternative) &&
            cand.offset == access.offset && cand.size == access.size &&
            cand.type_category != TypeCategory::RawBytes) {
            if (access_cat != TypeCategory::Unknown && access_cat != TypeCategory::RawBytes &&
                access_cat != cand.type_category &&
                !types_compatible(cand.type_category, access_cat)) {
                return false;
            }
        }

        return cand.offset <= access.offset &&
               cand.offset + static_cast<sval_t>(cand.size) >=
               access.offset + static_cast<sval_t>(access.size);
    };

    auto candidate_covers_access_fast = [&](const FieldCandidate& candidate,
                                            const FieldAccess& access,
                                            TypeCategory access_cat,
                                            size_t access_index) {
        if (!covers_by_shape(candidate, access, access_cat, access_index)) {
            return false;
        }

        const bool has_non_padding_evidence =
            access.access_type == AccessType::Call ||
            access.access_type == AccessType::AddressTaken ||
            access.is_call_argument;

        if (!has_non_padding_evidence || !is_padding_like_candidate(candidate)) {
            return true;
        }

        for (const auto& other : candidates_) {
            if (is_padding_like_candidate(other)) {
                continue;
            }
            if (covers_by_shape(other, access, access_cat, access_index)) {
                return false;
            }
        }

        return true;
    };
    algorithms::parallel_for_chunks(num_accesses, 32, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            if (redundant_accesses[i] != 0) {
                continue;
            }

            const auto& access = pattern_->all_accesses[i];
            auto& covering = coverage_map[i];
            covering.reserve(std::min<size_t>(num_candidates, 16));

            for (size_t j = 0; j < num_candidates; ++j) {
                const auto& cand = candidates_[field_vars_[j].candidate_id];
                if (candidate_covers_access_fast(cand, access, access_categories[i], i)) {
                    covering.push_back(static_cast<int32_t>(j));
                }
            }
        }
    });
    
    // Prefetch candidate data
    if (num_candidates > 0) {
        simd::prefetch_range(&candidates_[0], 
            std::min(num_candidates * sizeof(FieldCandidate), size_t{simd::kCacheLine * 8}));
    }

    for (size_t i = 0; i < num_accesses; ++i) {
        const auto& access = pattern_->all_accesses[i];

        if (redundant_accesses[i] != 0) {
            continue;
        }

        // Build: OR of all candidates that cover this access
        ::z3::expr_vector covering(ctx);
        for (int32_t candidate_idx : coverage_map[i]) {
            covering.push_back(field_vars_[static_cast<size_t>(candidate_idx)].selected);
        }

        if (covering.empty()) {
            // No candidate covers this access - this is a problem
            // Add a false constraint to force UNSAT with useful core
            ConstraintProvenance prov;
            prov.insn_ea = access.insn_ea;
            prov.access_idx = static_cast<int>(i);
            prov.description.sprnt("Access at 0x%llX (offset 0x%llX size %u) has no covering field",
                static_cast<unsigned long long>(access.insn_ea),
                static_cast<unsigned long long>(access.offset),
                access.size);
            prov.is_soft = false;
            prov.kind = ConstraintProvenance::Kind::Coverage;
            prov.weight = config_.weight_coverage;

            z3_log("[Structor/Z3]   WARNING: Access %zu at offset 0x%llX size %u has NO covering candidates!\n",
                   i, static_cast<unsigned long long>(access.offset), access.size);
            constraint_tracker_.add_hard(solver_, ctx.bool_val(false), prov);
            ++uncovered_count;
            continue;
        }

        // At least one covering field must be selected
        ::z3::expr coverage = ::z3::mk_or(covering);

        ConstraintProvenance prov;
        prov.insn_ea = access.insn_ea;
        prov.access_idx = static_cast<int>(i);
        prov.description.sprnt("Access at offset 0x%llX size %u must be covered",
            static_cast<unsigned long long>(access.offset), access.size);
        prov.is_soft = false;
        prov.kind = ConstraintProvenance::Kind::Coverage;
        prov.weight = config_.weight_coverage;

        constraint_tracker_.add_hard(solver_, coverage, prov);
        ++statistics_.coverage_constraints;
    }
    
    z3_log("[Structor/Z3]   Added %u coverage constraints (%d uncovered accesses)\n", 
           statistics_.coverage_constraints, uncovered_count);
}

void LayoutConstraintBuilder::add_non_overlap_constraints() {
    z3_log("[Structor/Z3] Adding non-overlap constraints (allow_unions=%s)\n", 
           config_.allow_unions ? "true" : "false");
    int overlap_count = 0;

    // OPTIMIZATION: Use O(n log n) sweep line algorithm for large candidate sets
    // instead of O(n²) pairwise comparison
    const size_t n = field_vars_.size();
    
    if (n >= 64) {
        // Use sweep line for large sets - O(n log n + k) where k is overlapping pairs
        std::vector<algorithms::Interval> intervals;
        intervals.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            const auto& c = candidates_[field_vars_[i].candidate_id];
            intervals.emplace_back(
                c.offset,
                c.offset + static_cast<int64_t>(c.size),
                static_cast<int32_t>(i)
            );
        }
        
        auto overlapping_pairs = algorithms::find_overlapping_pairs(intervals);
        
        z3_log("[Structor/Z3]   Using sweep line: found %zu overlapping pairs from %zu candidates\n",
               overlapping_pairs.size(), n);
        
        // Process overlapping pairs
        for (const auto& [idx_i, idx_j] : overlapping_pairs) {
            const auto& fv1 = field_vars_[idx_i];
            const auto& fv2 = field_vars_[idx_j];
            const auto& c1 = candidates_[fv1.candidate_id];
            const auto& c2 = candidates_[fv2.candidate_id];
            
            ++overlap_count;
            
            if (config_.allow_unions) {
                ::z3::expr non_overlap =
                    (fv1.offset + ctx_.uint_val(c1.size) <= fv2.offset) ||
                    (fv2.offset + ctx_.uint_val(c2.size) <= fv1.offset);

                ::z3::expr same_union =
                    fv1.is_union_member && fv2.is_union_member &&
                    (fv1.union_group == fv2.union_group) &&
                    (fv1.union_group >= 0);

                ::z3::expr constraint = ::z3::implies(
                    fv1.selected && fv2.selected,
                    non_overlap || same_union
                );

                ConstraintProvenance prov;
                prov.description.sprnt("Non-overlap or union at 0x%llX",
                    static_cast<unsigned long long>(c1.offset));
                prov.is_soft = false;
                prov.kind = ConstraintProvenance::Kind::NonOverlap;
                constraint_tracker_.add_hard(solver_, constraint, prov);
            } else {
                ::z3::expr non_overlap =
                    (fv1.offset + ctx_.uint_val(c1.size) <= fv2.offset) ||
                    (fv2.offset + ctx_.uint_val(c2.size) <= fv1.offset);

                ::z3::expr constraint = ::z3::implies(
                    fv1.selected && fv2.selected,
                    non_overlap
                );

                ConstraintProvenance prov;
                prov.description.sprnt("Non-overlap at 0x%llX",
                    static_cast<unsigned long long>(c1.offset));
                prov.is_soft = false;
                prov.kind = ConstraintProvenance::Kind::NonOverlap;

                constraint_tracker_.add_hard(solver_, constraint, prov);
            }
        }
        
    } else {
        // Small set - use O(n²) which has lower constant factors
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const auto& fv1 = field_vars_[i];
                const auto& fv2 = field_vars_[j];
                const auto& c1 = candidates_[fv1.candidate_id];
                const auto& c2 = candidates_[fv2.candidate_id];

                bool could_overlap = c1.overlaps(c2);

                if (!could_overlap) {
                    continue;
                }
                ++overlap_count;

                if (config_.allow_unions) {
                    ::z3::expr non_overlap =
                        (fv1.offset + ctx_.uint_val(c1.size) <= fv2.offset) ||
                        (fv2.offset + ctx_.uint_val(c2.size) <= fv1.offset);

                    ::z3::expr same_union =
                        fv1.is_union_member && fv2.is_union_member &&
                        (fv1.union_group == fv2.union_group) &&
                        (fv1.union_group >= 0);

                    ::z3::expr constraint = ::z3::implies(
                        fv1.selected && fv2.selected,
                        non_overlap || same_union
                    );

                    ConstraintProvenance prov;
                    prov.description.sprnt("Non-overlap or union at 0x%llX",
                        static_cast<unsigned long long>(c1.offset));
                    prov.is_soft = false;
                    prov.kind = ConstraintProvenance::Kind::NonOverlap;
                    constraint_tracker_.add_hard(solver_, constraint, prov);
                } else {
                    ::z3::expr non_overlap =
                        (fv1.offset + ctx_.uint_val(c1.size) <= fv2.offset) ||
                        (fv2.offset + ctx_.uint_val(c2.size) <= fv1.offset);

                    ::z3::expr constraint = ::z3::implies(
                        fv1.selected && fv2.selected,
                        non_overlap
                    );

                    ConstraintProvenance prov;
                    prov.description.sprnt("Non-overlap at 0x%llX",
                        static_cast<unsigned long long>(c1.offset));
                    prov.is_soft = false;
                    prov.kind = ConstraintProvenance::Kind::NonOverlap;

                    constraint_tracker_.add_hard(solver_, constraint, prov);
                }
            }
        }
    }
    
    z3_log("[Structor/Z3]   Added %d non-overlap constraints for overlapping candidate pairs\n", overlap_count);
}

void LayoutConstraintBuilder::add_alignment_constraints() {
    auto& ctx = ctx_.ctx();

    z3_log("[Structor/Z3] Adding alignment constraints\n");
    int misaligned_count = 0;

    for (const auto& fv : field_vars_) {
        const auto& cand = candidates_[fv.candidate_id];
        uint32_t natural_align = std::min(
            ctx_.type_encoder().natural_alignment(cand.type_category),
            cand.alignment());
        natural_align = std::max<uint32_t>(1, natural_align);

        // Effective alignment = min(natural_align, packing)
        ::z3::expr effective_align = config_.model_packing && packing_var_
            ? ::z3::ite(ctx.int_val(static_cast<int>(natural_align)) < *packing_var_,
                        ctx.int_val(static_cast<int>(natural_align)),
                        *packing_var_)
            : ctx.int_val(static_cast<int>(natural_align));

        // Soft constraint: offset % effective_align == 0
        // This must respect modeled packing. A packed struct can legitimately
        // contain fields that are misaligned with respect to their natural size.
        bool always_aligned = (cand.offset % natural_align) == 0;

        if (!always_aligned) {
            // Only add constraint if misaligned
            ::z3::expr offset_val = ctx_.int_val(static_cast<int64_t>(cand.offset));
            ::z3::expr constraint = ::z3::implies(
                fv.selected,
                ::z3::mod(offset_val, effective_align) == 0);

            ConstraintProvenance prov;
            prov.description.sprnt("Alignment of field at 0x%llX (need %u, candidate %d)",
                static_cast<unsigned long long>(cand.offset), natural_align, fv.candidate_id);
            prov.is_soft = true;
            prov.kind = ConstraintProvenance::Kind::Alignment;
            prov.weight = config_.weight_alignment;

            constraint_tracker_.add_soft(solver_, constraint, prov, config_.weight_alignment);
            ++statistics_.alignment_constraints;
            ++misaligned_count;
        }
    }
    
    z3_log("[Structor/Z3]   Added %d alignment constraints for misaligned candidates\n", misaligned_count);
}

void LayoutConstraintBuilder::add_type_constraints() {
    // Add soft constraints for type consistency between overlapping candidates
    // that might end up in the same union

    z3_log("[Structor/Z3] Adding type consistency constraints\n");
    auto& ctx = ctx_.ctx();
    int type_constraint_count = 0;

    // Preserve an observed type interpretation through either its scalar
    // candidate or an array carrying every source observation for that view.
    // Requiring the scalar itself prevents equivalent array-union layouts.
    std::unordered_map<std::size_t, ::z3::expr> storage_view_cache;
    const auto selected_storage_view = [&](std::size_t scalar_index) {
        if (const auto found = storage_view_cache.find(scalar_index);
            found != storage_view_cache.end()) {
            return found->second;
        }
        ::z3::expr_vector alternatives(ctx);
        alternatives.push_back(field_vars_[scalar_index].selected);
        const auto& scalar = candidates_[field_vars_[scalar_index].candidate_id];
        for (std::size_t i = 0; i < field_vars_.size(); ++i) {
            const auto& candidate = candidates_[field_vars_[i].candidate_id];
            if (candidate.replaces_scalar_evidence(scalar)) {
                alternatives.push_back(field_vars_[i].selected);
            }
        }
        auto selected = ::z3::mk_or(alternatives);
        storage_view_cache.emplace(scalar_index, selected);
        return selected;
    };

    std::vector<size_t> by_offset(field_vars_.size());
    for (size_t i = 0; i < by_offset.size(); ++i) {
        by_offset[i] = i;
    }

    std::stable_sort(by_offset.begin(), by_offset.end(), [&](size_t lhs, size_t rhs) {
        const auto& c1 = candidates_[field_vars_[lhs].candidate_id];
        const auto& c2 = candidates_[field_vars_[rhs].candidate_id];
        return c1.offset < c2.offset;
    });

    size_t group_begin = 0;
    while (group_begin < by_offset.size()) {
        const auto& first = candidates_[field_vars_[by_offset[group_begin]].candidate_id];
        size_t group_end = group_begin + 1;
        while (group_end < by_offset.size()) {
            const auto& next = candidates_[field_vars_[by_offset[group_end]].candidate_id];
            if (next.offset != first.offset) {
                break;
            }
            ++group_end;
        }

        for (size_t left = group_begin; left < group_end; ++left) {
            for (size_t right = left + 1; right < group_end; ++right) {
                const size_t i = by_offset[left];
                const size_t j = by_offset[right];
                const auto& c1 = candidates_[field_vars_[i].candidate_id];
                const auto& c2 = candidates_[field_vars_[j].candidate_id];

                // Check type compatibility
                bool compatible = types_compatible(c1.type_category, c2.type_category);
                for (int lhs_idx : c1.source_access_indices) {
                    if (lhs_idx < 0 ||
                        static_cast<size_t>(lhs_idx) >= pattern_->all_accesses.size()) {
                        continue;
                    }
                    for (int rhs_idx : c2.source_access_indices) {
                        if (rhs_idx < 0 ||
                            static_cast<size_t>(rhs_idx) >= pattern_->all_accesses.size()) {
                            continue;
                        }
                        if (!field_access_evidence_compatible(
                                pattern_->all_accesses[static_cast<size_t>(lhs_idx)],
                                pattern_->all_accesses[static_cast<size_t>(rhs_idx)])) {
                            compatible = false;
                            break;
                        }
                    }
                    if (!compatible) {
                        break;
                    }
                }
                ++type_constraint_count;

                if (!compatible) {
                    const bool direct_evidence_alternatives =
                        c1.offset == c2.offset && c1.size == c2.size &&
                        !c1.source_access_indices.empty() &&
                        !c2.source_access_indices.empty() &&
                        (c1.kind == FieldCandidate::Kind::DirectAccess ||
                         c1.kind == FieldCandidate::Kind::ArrayElement ||
                         c1.kind == FieldCandidate::Kind::UnionAlternative) &&
                        (c2.kind == FieldCandidate::Kind::DirectAccess ||
                         c2.kind == FieldCandidate::Kind::ArrayElement ||
                         c2.kind == FieldCandidate::Kind::UnionAlternative);

                    if (direct_evidence_alternatives) {
                        ConstraintProvenance prov;
                        prov.description.sprnt(
                            "Mandatory incompatible storage views at 0x%llX: %s vs %s",
                            static_cast<unsigned long long>(c1.offset),
                            type_category_name(c1.type_category),
                            type_category_name(c2.type_category));
                        prov.is_soft = false;
                        prov.kind = ConstraintProvenance::Kind::TypeMatch;

                        if (!config_.allow_unions) {
                            constraint_tracker_.add_hard(
                                solver_, ctx.bool_val(false), prov);
                        } else {
                            // The hard non-overlap constraints require the
                            // selected overlapping views to share a union.
                            const ::z3::expr mandatory_union =
                                selected_storage_view(i) && selected_storage_view(j);
                            constraint_tracker_.add_hard(
                                solver_, mandatory_union, prov);
                        }
                        ++statistics_.type_constraints;
                        continue;
                    }

                    if (config_.allow_unions) {
                        continue;
                    }

                    ConstraintProvenance prov;
                    prov.description.sprnt("Type consistency at 0x%llX: %s vs %s",
                        static_cast<unsigned long long>(c1.offset),
                        type_category_name(c1.type_category),
                        type_category_name(c2.type_category));
                    const auto& weight_source = (c1.source_access_indices.size() >= c2.source_access_indices.size())
                        ? c1 : c2;
                    int weight = access_weight(weight_source, config_.weight_type_consistency);

                    prov.is_soft = true;
                    prov.kind = ConstraintProvenance::Kind::TypeMatch;
                    prov.weight = weight;

                    // Prefer not selecting both incompatible types
                    ::z3::expr constraint = !(field_vars_[i].selected && field_vars_[j].selected);

                    constraint_tracker_.add_soft(solver_, constraint, prov, weight);
                    ++statistics_.type_constraints;
                }
            }
        }

        group_begin = group_end;
    }
    
    z3_log("[Structor/Z3]   Added %u type consistency constraints (checked %d pairs)\n", 
           statistics_.type_constraints, type_constraint_count);
}

void LayoutConstraintBuilder::add_type_preference_constraints() {
    // Add soft constraints preferring typed fields over raw_bytes/unknown
    // When two overlapping candidates exist, prefer the one with a more specific type
    
    int preference_count = 0;
    const size_t n = field_vars_.size();
    
    // OPTIMIZATION: Use sweep-line for large candidate sets
    if (n >= 64) {
        // Build interval list for sweep-line overlap detection
        std::vector<algorithms::Interval> intervals;
        intervals.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            const auto& c = candidates_[field_vars_[i].candidate_id];
            intervals.emplace_back(
                c.offset,
                c.offset + static_cast<int64_t>(c.size),
                static_cast<int32_t>(i)
            );
        }
        
        auto overlapping_pairs = algorithms::find_overlapping_pairs(intervals);
        
        for (const auto& [idx_i, idx_j] : overlapping_pairs) {
            const auto& c1 = candidates_[field_vars_[idx_i].candidate_id];
            const auto& c2 = candidates_[field_vars_[idx_j].candidate_id];
            
            bool c1_is_raw = (c1.type_category == TypeCategory::RawBytes || 
                              c1.type_category == TypeCategory::Unknown);
            bool c2_is_raw = (c2.type_category == TypeCategory::RawBytes || 
                              c2.type_category == TypeCategory::Unknown);
            
            if (c1_is_raw && !c2_is_raw) {
                ConstraintProvenance prov;
                prov.description.sprnt("Prefer typed field at 0x%llX over raw at 0x%llX",
                    static_cast<unsigned long long>(c2.offset),
                    static_cast<unsigned long long>(c1.offset));
                int weight = access_weight(c2, 1);
                prov.is_soft = true;
                prov.kind = ConstraintProvenance::Kind::TypeMatch;
                prov.weight = weight;
                
                constraint_tracker_.add_soft(solver_, !field_vars_[idx_i].selected, prov, weight);
                ++preference_count;
            }
            else if (c2_is_raw && !c1_is_raw) {
                ConstraintProvenance prov;
                prov.description.sprnt("Prefer typed field at 0x%llX over raw at 0x%llX",
                    static_cast<unsigned long long>(c1.offset),
                    static_cast<unsigned long long>(c2.offset));
                int weight = access_weight(c1, 1);
                prov.is_soft = true;
                prov.kind = ConstraintProvenance::Kind::TypeMatch;
                prov.weight = weight;
                
                constraint_tracker_.add_soft(solver_, !field_vars_[idx_j].selected, prov, weight);
                ++preference_count;
            } else if (!c1_is_raw && !c2_is_raw) {
                const FieldCandidate* preferred = nullptr;
                size_t penalize_idx = 0;

                if (should_prefer_candidate(c1, c2)) {
                    preferred = &c1;
                    penalize_idx = idx_j;
                } else if (should_prefer_candidate(c2, c1)) {
                    preferred = &c2;
                    penalize_idx = idx_i;
                }

                if (preferred) {
                    ConstraintProvenance prov;
                    prov.description.sprnt("Prefer richer aggregate at 0x%llX over overlap at 0x%llX",
                        static_cast<unsigned long long>(preferred->offset),
                        static_cast<unsigned long long>(candidates_[field_vars_[penalize_idx].candidate_id].offset));
                    int weight = access_weight(*preferred, std::max(2, config_.weight_prefer_arrays));
                    prov.is_soft = true;
                    prov.kind = ConstraintProvenance::Kind::TypeMatch;
                    prov.weight = weight;

                    constraint_tracker_.add_soft(solver_, !field_vars_[penalize_idx].selected, prov, weight);
                    ++preference_count;
                }
            }
        }
    } else {
        // Small set - use O(n²) with lower constant factors
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const auto& c1 = candidates_[field_vars_[i].candidate_id];
                const auto& c2 = candidates_[field_vars_[j].candidate_id];
                
                // Only for overlapping candidates
                if (!c1.overlaps(c2)) continue;
                
                // Determine which one is more specifically typed
                bool c1_is_raw = (c1.type_category == TypeCategory::RawBytes || 
                                  c1.type_category == TypeCategory::Unknown);
                bool c2_is_raw = (c2.type_category == TypeCategory::RawBytes || 
                                  c2.type_category == TypeCategory::Unknown);
                
                if (c1_is_raw && !c2_is_raw) {
                    // Prefer c2 (typed) over c1 (raw)
                    ConstraintProvenance prov;
                    prov.description.sprnt("Prefer typed field at 0x%llX over raw at 0x%llX",
                        static_cast<unsigned long long>(c2.offset),
                        static_cast<unsigned long long>(c1.offset));
                    int weight = access_weight(c2, 1);
                    prov.is_soft = true;
                    prov.kind = ConstraintProvenance::Kind::TypeMatch;
                    prov.weight = weight;
                    
                    // Prefer the typed candidate by penalizing selecting raw when typed exists
                    constraint_tracker_.add_soft(solver_, !field_vars_[i].selected, prov, weight);
                    ++preference_count;
                }
                else if (c2_is_raw && !c1_is_raw) {
                    // Prefer c1 (typed) over c2 (raw)
                    ConstraintProvenance prov;
                    prov.description.sprnt("Prefer typed field at 0x%llX over raw at 0x%llX",
                        static_cast<unsigned long long>(c1.offset),
                        static_cast<unsigned long long>(c2.offset));
                    int weight = access_weight(c1, 1);
                    prov.is_soft = true;
                    prov.kind = ConstraintProvenance::Kind::TypeMatch;
                    prov.weight = weight;
                    
                    constraint_tracker_.add_soft(solver_, !field_vars_[j].selected, prov, weight);
                    ++preference_count;
                } else if (!c1_is_raw && !c2_is_raw) {
                    const FieldCandidate* preferred = nullptr;
                    size_t penalize_idx = 0;

                    if (should_prefer_candidate(c1, c2)) {
                        preferred = &c1;
                        penalize_idx = j;
                    } else if (should_prefer_candidate(c2, c1)) {
                        preferred = &c2;
                        penalize_idx = i;
                    }

                    if (preferred) {
                        ConstraintProvenance prov;
                        prov.description.sprnt("Prefer richer aggregate at 0x%llX over overlap at 0x%llX",
                            static_cast<unsigned long long>(preferred->offset),
                            static_cast<unsigned long long>(candidates_[field_vars_[penalize_idx].candidate_id].offset));
                        int weight = access_weight(*preferred, std::max(2, config_.weight_prefer_arrays));
                        prov.is_soft = true;
                        prov.kind = ConstraintProvenance::Kind::TypeMatch;
                        prov.weight = weight;

                        constraint_tracker_.add_soft(solver_, !field_vars_[penalize_idx].selected, prov, weight);
                        ++preference_count;
                    }
                }
            }
        }
    }
    
    if (preference_count > 0) {
        z3_log("[Structor/Z3]   Added %d type preference constraints (prefer typed over raw)\n", 
               preference_count);
    }
}

void LayoutConstraintBuilder::add_size_bound_constraints() {
    sval_t evidence_origin = 0;
    if (pattern_) {
        for (const auto& access : pattern_->all_accesses) {
            evidence_origin = std::min(evidence_origin, access.offset);
        }
    }

    for (size_t i = 0; i < candidates_.size(); ++i) {
        const auto& candidate = candidates_[i];
        bool in_bounds = candidate.offset >= evidence_origin;
        uint64_t span = std::numeric_limits<uint64_t>::max();
        if (in_bounds &&
            candidate.offset <= std::numeric_limits<sval_t>::max() -
                                    static_cast<sval_t>(candidate.size)) {
            const sval_t end =
                candidate.offset + static_cast<sval_t>(candidate.size);
            span = static_cast<uint64_t>(end) -
                   static_cast<uint64_t>(evidence_origin);
            in_bounds = span <= config_.max_struct_size;
        }

        ConstraintProvenance prov;
        prov.description.sprnt(
            "Structure bound for candidate %zu (span=%llu, max=%u)",
            i,
            static_cast<unsigned long long>(span),
            config_.max_struct_size);
        prov.is_soft = false;
        prov.kind = ConstraintProvenance::Kind::SizeMatch;
        constraint_tracker_.add_hard(
            solver_,
            ::z3::implies(field_vars_[i].selected,
                          ctx_.ctx().bool_val(in_bounds)),
            prov);
    }
}

void LayoutConstraintBuilder::add_array_constraints() {
    // Bind preferences to the actual typed candidate and its source evidence.
    // Distinct views can have identical base offsets, strides, and lengths.
    for (std::size_t i = 0; i < field_vars_.size(); ++i) {
        const auto& array = candidates_[field_vars_[i].candidate_id];
        if (array.kind != FieldCandidate::Kind::ArrayField) continue;
        for (std::size_t j = 0; j < field_vars_.size(); ++j) {
            const auto& element = candidates_[field_vars_[j].candidate_id];
            if (array.replaces_scalar_evidence(element) ||
                (array.type_category == TypeCategory::Struct &&
                 element.kind == FieldCandidate::Kind::ArrayElement &&
                 array.covers_source_evidence(element))) {
                ::z3::expr constraint = ::z3::implies(
                    field_vars_[i].selected,
                    !field_vars_[j].selected
                );

                ConstraintProvenance prov;
                prov.description.sprnt("Prefer array over elements at 0x%llX",
                    static_cast<unsigned long long>(array.offset));
                prov.is_soft = true;
                prov.kind = ConstraintProvenance::Kind::ArrayDetection;
                prov.weight = config_.weight_prefer_arrays;

                constraint_tracker_.add_soft(solver_, constraint, prov,
                    config_.weight_prefer_arrays);
            }
        }
    }
}

void LayoutConstraintBuilder::add_optimization_objectives() {
    // Minimize total number of selected fields (soft)
    auto& ctx = ctx_.ctx();

    ::z3::expr_vector selected(ctx);
    for (const auto& fv : field_vars_) {
        selected.push_back(::z3::ite(fv.selected, ctx.int_val(1), ctx.int_val(0)));
    }

    if (!selected.empty()) {
        // We can't directly add optimization objectives to a regular solver
        // This would need z3::optimize, but we're using solver for tracking
        // Instead, we add soft constraints that penalize selecting too many fields
    }
}

qvector<uint32_t> LayoutConstraintBuilder::normalized_packing_options() const {
    qvector<uint32_t> result;
    if (!config_.model_packing || config_.default_alignment == 0) {
        return result;
    }

    for (uint32_t value : config_.packing_options) {
        const bool is_power_of_two = value != 0 && (value & (value - 1)) == 0;
        if (!is_power_of_two || value > config_.default_alignment) {
            continue;
        }
        if (std::find(result.begin(), result.end(), value) == result.end()) {
            result.push_back(value);
        }
    }

    // The ABI-default cap is a real domain value even when it is larger than
    // IDA's explicit #pragma-pack choices.  Without it, alignment=32 would
    // force every naturally aligned structure into an artificial pack(16).
    if (is_valid_packing_value(config_.default_alignment) &&
        std::find(result.begin(), result.end(), config_.default_alignment) ==
            result.end()) {
        result.push_back(config_.default_alignment);
    }

    std::sort(result.begin(), result.end(), std::greater<uint32_t>());
    return result;
}

std::optional<::z3::model> LayoutConstraintBuilder::canonicalize_packing_model(
    const ::z3::model& model,
    const ::z3::expr_vector& active_assumptions)
{
    inferred_packing_.reset();
    if (!packing_var_) {
        return model;
    }

    const qvector<uint32_t> options = normalized_packing_options();
    if (options.empty()) {
        return std::nullopt;
    }

    qvector<uint32_t> physically_valid;
    for (uint32_t packing : options) {
        bool valid = true;
        for (const auto& fv : field_vars_) {
            if (!get_bool_value(model, fv.selected)) {
                continue;
            }

            const auto& candidate = candidates_[fv.candidate_id];
            const uint32_t natural_alignment = std::max<uint32_t>(
                1, std::min(
                    ctx_.type_encoder().natural_alignment(candidate.type_category),
                    candidate.alignment()));
            const uint32_t effective_alignment =
                std::min(natural_alignment, packing);
            if (candidate.offset % static_cast<sval_t>(effective_alignment) != 0) {
                valid = false;
                break;
            }
        }
        if (valid) {
            physically_valid.push_back(packing);
        }
    }

    if (physically_valid.empty()) {
        return std::nullopt;
    }

    // Pin all decisions from the original model. This makes the packing
    // canonicalization a metadata refinement, never a second layout choice.
    solver_.push();
    for (const auto& fv : field_vars_) {
        solver_.add(fv.selected == ctx_.ctx().bool_val(get_bool_value(model, fv.selected)));
        solver_.add(
            fv.is_union_member ==
            ctx_.ctx().bool_val(get_bool_value(model, fv.is_union_member)));
        solver_.add(
            fv.union_group ==
            ctx_.ctx().int_val(static_cast<int>(get_int_value(model, fv.union_group))));
    }

    std::optional<::z3::model> canonical_model;
    for (uint32_t packing : physically_valid) {
        solver_.push();
        solver_.add(*packing_var_ == static_cast<int>(packing));
        if (check_solver_with_deadline(active_assumptions) == ::z3::sat) {
            canonical_model = solver_.get_model();
            inferred_packing_ = packing;
            solver_.pop();
            break;
        }
        solver_.pop();
    }
    solver_.pop();

    return canonical_model;
}

::z3::check_result LayoutConstraintBuilder::check_solver_with_deadline(
    const ::z3::expr_vector& assumptions) {
    last_unknown_reason_.clear();
    if (solve_deadline_) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= *solve_deadline_) {
            solve_deadline_exhausted_ = true;
            last_unknown_reason_ = "aggregate solver deadline exceeded";
            return ::z3::unknown;
        }

        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            *solve_deadline_ - now);
        const unsigned timeout_ms = static_cast<unsigned>(std::max<int64_t>(
            1, remaining.count() + 1));
        ::z3::params params(ctx_.ctx());
        params.set("timeout", timeout_ms);
        if (ctx_.config().max_memory_mb != 0) {
            params.set("max_memory", ctx_.config().max_memory_mb);
        }
        if (ctx_.config().produce_unsat_cores) {
            params.set("unsat_core", true);
        }
        solver_.set(params);
    }

    ++statistics_.solve_iterations;
    const auto result = solver_.check(assumptions);
    if (result == ::z3::unknown) {
        last_unknown_reason_ = Z3Context::get_unknown_reason(solver_);
        if (solve_deadline_ &&
            std::chrono::steady_clock::now() >= *solve_deadline_) {
            solve_deadline_exhausted_ = true;
            last_unknown_reason_ = "aggregate solver deadline exceeded";
        }
    }
    return result;
}

::z3::check_result LayoutConstraintBuilder::check_optimizer_with_deadline(
    ::z3::optimize& optimizer) {
    last_unknown_reason_.clear();
    if (solve_deadline_) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= *solve_deadline_) {
            solve_deadline_exhausted_ = true;
            last_unknown_reason_ = "aggregate solver deadline exceeded";
            return ::z3::unknown;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            *solve_deadline_ - now);
        ::z3::params params(ctx_.ctx());
        params.set("timeout", static_cast<unsigned>(std::max<int64_t>(
            1, remaining.count() + 1)));
        optimizer.set(params);
    }

    ++statistics_.solve_iterations;
    const auto result = optimizer.check();
    if (result == ::z3::unknown) {
        if (const char* reason = Z3_optimize_get_reason_unknown(
                ctx_.ctx(), optimizer)) {
            last_unknown_reason_ = reason;
        }
        if (solve_deadline_ &&
            std::chrono::steady_clock::now() >= *solve_deadline_) {
            solve_deadline_exhausted_ = true;
            last_unknown_reason_ = "aggregate solver deadline exceeded";
        }
    }
    return result;
}

Z3Result LayoutConstraintBuilder::solve_weighted_maxsmt(
    std::chrono::steady_clock::time_point start_time) {
    auto optimizer = ctx_.make_optimizer();
    optimizer.add(solver_.assertions());

    const ::z3::expr_vector hard_literals =
        constraint_tracker_.add_hard_literals_to_optimizer(optimizer);

    const ::z3::expr_vector soft_literals = constraint_tracker_.get_soft_literals();
    for (unsigned i = 0; i < soft_literals.size(); ++i) {
        const ConstraintProvenance* provenance =
            constraint_tracker_.get_provenance(soft_literals[i]);
        const unsigned weight = static_cast<unsigned>(std::max(
            1, provenance ? provenance->weight : 1));
        optimizer.add_soft(soft_literals[i], weight);
    }

    // Deterministic secondary objectives are evaluated only after the exact
    // weighted preference optimum.
    auto& ctx = ctx_.ctx();
    ::z3::expr selected_count = ctx.int_val(0);
    ::z3::expr union_count = ctx.int_val(0);
    ::z3::expr canonical_rank = ctx.int_val(0);
    ::z3::expr canonical_weight = ctx.int_val(1);
    for (size_t i = 0; i < field_vars_.size(); ++i) {
        const auto& fv = field_vars_[i];
        selected_count = selected_count +
            ::z3::ite(fv.selected, ctx.int_val(1), ctx.int_val(0));
        union_count = union_count +
            ::z3::ite(fv.selected && fv.is_union_member,
                      ctx.int_val(1), ctx.int_val(0));
        canonical_rank = canonical_rank +
            ::z3::ite(fv.selected,
                      canonical_weight,
                      ctx.int_val(0));
        canonical_weight = canonical_weight * ctx.int_val(2);
    }
    optimizer.minimize(selected_count);
    optimizer.minimize(union_count);
    optimizer.minimize(canonical_rank);

    const auto check = check_optimizer_with_deadline(optimizer);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    statistics_.solve_time = elapsed;

    if (check == ::z3::unknown) {
        const std::string reason = last_unknown_reason_.empty()
            ? "weighted Max-SMT optimizer returned unknown"
            : last_unknown_reason_;
        return Z3Result::make_unknown(
            reason.c_str(),
            elapsed);
    }
    if (check == ::z3::unsat) {
        auto core = constraint_tracker_.analyze_unsat_core(optimizer.unsat_core());
        return Z3Result::make_unsat(std::move(core), elapsed);
    }

    ::z3::model optimized_model = optimizer.get_model();
    auto canonical_model = canonicalize_packing_model(
        optimized_model, hard_literals);
    if (!canonical_model) {
        const auto final_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        const std::string reason = last_unknown_reason_.empty()
            ? "unable to derive a canonical packing assignment"
            : last_unknown_reason_;
        return Z3Result::make_unknown(
            reason.c_str(),
            final_elapsed);
    }

    qvector<ConstraintProvenance> dropped;
    for (unsigned i = 0; i < soft_literals.size(); ++i) {
        if (get_bool_value(optimized_model, soft_literals[i])) {
            continue;
        }
        if (const ConstraintProvenance* provenance =
                constraint_tracker_.get_provenance(soft_literals[i])) {
            ConstraintProvenance copy = *provenance;
            copy.tracking_literal = soft_literals[i];
            dropped.push_back(std::move(copy));
        }
    }

    statistics_.relaxations_performed = static_cast<unsigned>(dropped.size());
    detect_union_groups(*canonical_model);
    const auto final_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    statistics_.solve_time = final_elapsed;

    Z3Result result = dropped.empty()
        ? Z3Result::make_sat(std::move(*canonical_model), final_elapsed)
        : Z3Result::make_sat_relaxed(
              std::move(*canonical_model), std::move(dropped), final_elapsed);
    result.iterations = statistics_.solve_iterations;
    result.constraints_relaxed = statistics_.relaxations_performed;
    return result;
}

Z3Result LayoutConstraintBuilder::solve() {
    auto start_time = std::chrono::steady_clock::now();

    if (config_.enable_maxsmt && ctx_.config().max_memory_mb != 0) {
        return Z3Result::make_unknown(
            "configured memory limit cannot be enforced by the MaxSMT optimizer",
            std::chrono::milliseconds{0});
    }

    solve_deadline_.reset();
    solve_deadline_exhausted_ = false;
    last_unknown_reason_.clear();
    if (ctx_.config().timeout_ms != 0) {
        solve_deadline_ = start_time +
            std::chrono::milliseconds(ctx_.config().timeout_ms);
    }

    if (config_.enable_maxsmt) {
        return solve_weighted_maxsmt(start_time);
    }

    z3_log("[Structor/Z3] Solving constraints...\n");
    z3_log("[Structor/Z3] Solver assertions: %u\n", solver_.assertions().size());

    // Build assumptions from all tracking literals
    // All constraints are guarded by implications: tracking_lit => constraint
    // By assuming all tracking_lits are true, we activate all constraints
    ::z3::expr_vector assumptions = constraint_tracker_.get_all_literals();
    z3_log("[Structor/Z3] Assumptions (tracking literals): %u\n", assumptions.size());

    // First attempt: solve with all constraints assumed active
    auto result = check_solver_with_deadline(assumptions);

    auto end_time = std::chrono::steady_clock::now();
    auto solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    statistics_.solve_time = solve_time;
    if (result == ::z3::sat) {
        z3_log("[Structor/Z3] SAT - solution found in %lldms\n",
               static_cast<long long>(solve_time.count()));

        auto canonical_model = canonicalize_packing_model(solver_.get_model(), assumptions);
        if (!canonical_model) {
            const char* reason = last_unknown_reason_.empty()
                ? "unable to derive a canonical packing assignment"
                : last_unknown_reason_.c_str();
            return Z3Result::make_unknown(
                reason, solve_time);
        }
        ::z3::model model = std::move(*canonical_model);
        if (inferred_packing_) {
            z3_log("[Structor/Z3] Canonical packing: %u\n", *inferred_packing_);
        }

        // Detect union groups
        detect_union_groups(model);
        if (!union_resolutions_.empty()) {
            z3_log("[Structor/Z3] Detected %zu union groups\n", union_resolutions_.size());
        }

        Z3Result sat_result = Z3Result::make_sat(std::move(model), solve_time);
        sat_result.iterations = statistics_.solve_iterations;
        return sat_result;
    }
    else if (result == ::z3::unsat) {
        if (!config_.relax_on_unsat || config_.max_relaxation_iterations == 0) {
            return Z3Result::make_unsat(
                constraint_tracker_.analyze_unsat_core(solver_.unsat_core()),
                solve_time);
        }
        z3_log("[Structor/Z3] UNSAT - attempting relaxation...\n");
        // Try relaxation
        return solve_with_relaxation();
    }
    else {
        // Get the actual reason for unknown
        std::string reason = last_unknown_reason_.empty()
            ? Z3Context::get_unknown_reason(solver_)
            : last_unknown_reason_;
        z3_log("[Structor/Z3] UNKNOWN result after %lldms: %s\n",
               static_cast<long long>(solve_time.count()), reason.c_str());
        
        qstring reason_msg;
        reason_msg.sprnt("solver returned unknown: %s", reason.c_str());
        return Z3Result::make_unknown(reason_msg.c_str(), solve_time);
    }
}

Z3Result LayoutConstraintBuilder::solve_with_relaxation() {
    auto start_time = std::chrono::steady_clock::now();
    qvector<ConstraintProvenance> dropped_constraints;

    // Build the set of active assumptions (tracking literals)
    // Start with all tracking literals active
    std::unordered_set<std::string> active_assumptions;
    ::z3::expr_vector all_literals = constraint_tracker_.get_all_literals();
    for (unsigned i = 0; i < all_literals.size(); ++i) {
        active_assumptions.insert(all_literals[i].to_string());
    }

    const uint32_t max_iterations = config_.max_relaxation_iterations;

    z3_log("[Structor/Z3] Starting constraint relaxation (max %u iterations)\n",
           max_iterations);
    z3_log("[Structor/Z3] Initial assumptions: %zu\n", active_assumptions.size());

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // Get UNSAT core from the last check (which used assumptions)
        auto core = solver_.unsat_core();
        auto core_provenances = constraint_tracker_.analyze_unsat_core(core);

        z3_log("[Structor/Z3] Iteration %d: UNSAT core has %u constraints (%zu analyzed)\n",
               iteration + 1, core.size(), core_provenances.size());

        // Log all constraints in the core
        z3_log("[Structor/Z3]   Core contents:\n");
        for (const auto& prov : core_provenances) {
            z3_log("[Structor/Z3]     [%s w=%d] %s (lit=%s)\n",
                   prov.is_soft ? "SOFT" : "HARD",
                   prov.weight,
                   prov.description.c_str(),
                   prov.tracking_literal ? prov.tracking_literal->to_string().c_str() : "none");
        }

        // Find soft constraints in the core (prioritize by weight - lower weight = relax first)
        qvector<ConstraintProvenance> relaxable;
        for (const auto& prov : core_provenances) {
            if (prov.is_soft && prov.tracking_literal) {
                // Only include if it's still in our active assumptions
                std::string lit_str = prov.tracking_literal->to_string();
                if (active_assumptions.count(lit_str)) {
                    relaxable.push_back(prov);
                }
            }
        }

        if (relaxable.empty()) {
            // All core constraints are hard - truly unsatisfiable
            z3_log("[Structor/Z3] Relaxation failed - all core constraints are hard\n");
            auto end_time = std::chrono::steady_clock::now();
            auto solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            return Z3Result::make_unsat(std::move(core_provenances), solve_time);
        }

        // Sort by weight (ascending) - relax lowest weight constraints first
        std::sort(relaxable.begin(), relaxable.end(),
            [](const ConstraintProvenance& a, const ConstraintProvenance& b) {
                if (a.weight != b.weight) {
                    return a.weight < b.weight;
                }
                const std::string a_literal = a.tracking_literal
                    ? a.tracking_literal->to_string() : std::string();
                const std::string b_literal = b.tracking_literal
                    ? b.tracking_literal->to_string() : std::string();
                return a_literal < b_literal;
            });

        // Relax the lowest-weight soft constraint by removing from assumptions
        const auto& to_relax = relaxable[0];
        dropped_constraints.push_back(to_relax);

        z3_log("[Structor/Z3] Relaxing constraint (weight=%d): %s\n",
               to_relax.weight, to_relax.description.c_str());
        z3_log("[Structor/Z3]   Tracking literal: %s\n",
               to_relax.tracking_literal->to_string().c_str());

        // Remove this tracking literal from active assumptions
        active_assumptions.erase(to_relax.tracking_literal->to_string());

        ++statistics_.relaxations_performed;

        // Build new assumptions vector
        ::z3::expr_vector current_assumptions(ctx_.ctx());
        for (unsigned i = 0; i < all_literals.size(); ++i) {
            if (active_assumptions.count(all_literals[i].to_string())) {
                current_assumptions.push_back(all_literals[i]);
            }
        }

        z3_log("[Structor/Z3]   Remaining assumptions: %u\n", current_assumptions.size());

        // Re-solve with reduced assumptions
        auto result = check_solver_with_deadline(current_assumptions);

        if (result == ::z3::unknown) {
            auto end_time = std::chrono::steady_clock::now();
            auto solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            Z3Result unknown = Z3Result::make_unknown(
                last_unknown_reason_.empty()
                    ? Z3Context::get_unknown_reason(solver_).c_str()
                    : last_unknown_reason_.c_str(),
                solve_time);
            unknown.dropped_constraints = std::move(dropped_constraints);
            unknown.iterations = statistics_.solve_iterations;
            return unknown;
        }

        if (result == ::z3::sat) {
            z3_log("[Structor/Z3] Relaxation succeeded after dropping %zu constraints\n",
                   dropped_constraints.size());

            auto canonical_model = canonicalize_packing_model(
                solver_.get_model(), current_assumptions);
            if (!canonical_model) {
                auto end_time = std::chrono::steady_clock::now();
                auto solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time);
                Z3Result unknown = Z3Result::make_unknown(
                    last_unknown_reason_.empty()
                        ? "unable to derive a canonical packing assignment"
                        : last_unknown_reason_.c_str(),
                    solve_time);
                unknown.dropped_constraints = std::move(dropped_constraints);
                return unknown;
            }
            ::z3::model model = std::move(*canonical_model);

            // Detect union groups
            detect_union_groups(model);

            auto end_time = std::chrono::steady_clock::now();
            auto solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Return SAT with dropped constraints info
            Z3Result sat_result = Z3Result::make_sat_relaxed(
                std::move(model), std::move(dropped_constraints), solve_time);
            sat_result.iterations = statistics_.solve_iterations;
            sat_result.constraints_relaxed = statistics_.relaxations_performed;
            return sat_result;
        }
        // If still UNSAT, continue relaxing
    }

    // Reached max iterations without satisfiability
    auto end_time = std::chrono::steady_clock::now();
    auto solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Return UNSAT with all dropped constraints for diagnostics
    Z3Result unsat_result = Z3Result::make_unsat(
        constraint_tracker_.analyze_unsat_core(solver_.unsat_core()), solve_time);
    unsat_result.dropped_constraints = std::move(dropped_constraints);
    unsat_result.iterations = statistics_.solve_iterations;
    unsat_result.constraints_relaxed = statistics_.relaxations_performed;
    return unsat_result;
}

qvector<ConstraintProvenance> LayoutConstraintBuilder::extract_mus() {
    // Extract minimal unsatisfiable subset
    auto core = solver_.unsat_core();
    return constraint_tracker_.analyze_unsat_core(core);
}

SynthStruct LayoutConstraintBuilder::extract_struct(const ::z3::model& model) {
    auto start_time = std::chrono::steady_clock::now();

    z3_log("[Structor/Z3] Extracting structure from Z3 model...\n");

    SynthStruct result;

    // Collect selected fields
    qvector<std::pair<int, FieldCandidate>> selected_fields;

    for (const auto& fv : field_vars_) {
        if (get_bool_value(model, fv.selected)) {
            selected_fields.push_back({fv.candidate_id, candidates_[fv.candidate_id]});
        }
    }

    z3_log("[Structor/Z3]   Model selected %zu of %zu candidates\n", 
           selected_fields.size(), field_vars_.size());

    // Sort by offset
    std::sort(selected_fields.begin(), selected_fields.end(),
        [](const auto& a, const auto& b) {
            if (a.second.offset != b.second.offset) {
                return a.second.offset < b.second.offset;
            }
            if (a.second.size != b.second.size) {
                return a.second.size > b.second.size;
            }
            if (a.second.type_category != b.second.type_category) {
                return static_cast<int>(a.second.type_category) <
                       static_cast<int>(b.second.type_category);
            }
            return a.first < b.first;
        });
    
    // Log selected fields
    for (const auto& [cand_id, cand] : selected_fields) {
        z3_log("[Structor/Z3]   Selected: candidate %d at offset 0x%llX size %u type %s\n",
               cand_id, static_cast<unsigned long long>(cand.offset), cand.size,
               type_category_name(cand.type_category));
    }

    // Build fields, handling unions
    std::unordered_set<int> processed_union_groups;

    z3_log("[Structor/Z3]   Processing selected candidates for union detection:\n");
    for (const auto& [cand_id, candidate] : selected_fields) {
        const auto& fv = field_vars_[cand_id];

        // Check if part of a union
        bool is_union = get_bool_value(model, fv.is_union_member);
        int union_group = static_cast<int>(get_int_value(model, fv.union_group));

        z3_log("[Structor/Z3]     Candidate %d (offset=0x%llX, size=%u): is_union=%s, union_group=%d\n",
               cand_id, static_cast<unsigned long long>(candidate.offset), candidate.size,
               is_union ? "true" : "false", union_group);

        if (is_union && union_group >= 0) {
            // Check if we've already processed this union group
            if (processed_union_groups.count(union_group)) {
                z3_log("[Structor/Z3]       -> Skipping (union group %d already processed)\n", union_group);
                continue;
            }
            processed_union_groups.insert(union_group);

            // Find all members of this union group
            qvector<int> union_members;
            for (size_t i = 0; i < field_vars_.size(); ++i) {
                const auto& other_fv = field_vars_[i];
                if (get_bool_value(model, other_fv.selected) &&
                    get_bool_value(model, other_fv.is_union_member) &&
                    get_int_value(model, other_fv.union_group) == union_group) {
                    union_members.push_back(static_cast<int>(i));
                }
            }

            z3_log("[Structor/Z3]       -> Creating union with %zu members\n", union_members.size());
            for (int member_idx : union_members) {
                const auto& member_cand = candidates_[field_vars_[member_idx].candidate_id];
                z3_log("[Structor/Z3]         Union member: idx=%d, offset=0x%llX, size=%u\n",
                       member_idx, static_cast<unsigned long long>(member_cand.offset), member_cand.size);
            }

            if (union_members.size() <= 1) {
                z3_log("[Structor/Z3]       -> Single-member union; treating as regular field\n");
                SynthField field = field_from_candidate(candidate, ctx_.type_encoder(),
                                                         pattern_ ? &pattern_->all_accesses : nullptr);
                z3_log("[Structor/Z3]       -> Created field: name='%s', offset=0x%llX, size=%u\n",
                       field.name.c_str(), static_cast<unsigned long long>(field.offset), field.size);
                result.fields.push_back(std::move(field));
                continue;
            }

            // Create union field
            SynthField union_field = create_union_field(union_members, model);
            z3_log("[Structor/Z3]       -> Created union field: name='%s', offset=0x%llX, size=%u\n",
                   union_field.name.c_str(), static_cast<unsigned long long>(union_field.offset), union_field.size);
            result.fields.push_back(std::move(union_field));
        } else {
            // Regular field
            z3_log("[Structor/Z3]       -> Adding as regular field\n");
            SynthField field = field_from_candidate(candidate, ctx_.type_encoder(),
                                                     pattern_ ? &pattern_->all_accesses : nullptr);
            z3_log("[Structor/Z3]       -> Created field: name='%s', offset=0x%llX, size=%u\n",
                   field.name.c_str(), static_cast<unsigned long long>(field.offset), field.size);
            result.fields.push_back(std::move(field));
        }
    }

    std::sort(result.fields.begin(), result.fields.end(),
              [](const SynthField& a, const SynthField& b) {
                  if (a.offset != b.offset) return a.offset < b.offset;
                  if (a.size != b.size) return a.size > b.size;
                  if (a.is_union_candidate != b.is_union_candidate) {
                      return a.is_union_candidate;
                  }
                  return std::string(a.name.c_str()) < std::string(b.name.c_str());
              });

    // Set struct properties
    if (!result.fields.empty()) {
        result.size = static_cast<uint32_t>(
            result.fields.back().offset + result.fields.back().size);
    }

    result.alignment = config_.default_alignment;
    if (inferred_packing_) {
        result.alignment = std::min(result.alignment, *inferred_packing_);
    }

    auto end_time = std::chrono::steady_clock::now();
    statistics_.extraction_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Fill gaps between fields with padding
    if (config_.fill_gaps_with_padding && !result.fields.empty()) {
        qvector<SynthField> fields_with_padding;
        sval_t current_end = 0;

        for (const auto& field : result.fields) {
            // Check for gap before this field
            if (field.offset > current_end) {
                uint32_t gap_size = static_cast<uint32_t>(field.offset - current_end);
                z3_log("[Structor/Z3] Filling gap at 0x%llX (size %u) with padding\n",
                       static_cast<unsigned long long>(current_end), gap_size);
                fields_with_padding.push_back(SynthField::create_padding(current_end, gap_size));
            }
            fields_with_padding.push_back(field);
            current_end = std::max(
                current_end,
                field.offset + static_cast<sval_t>(field.size));
        }

        result.fields = std::move(fields_with_padding);
        
        // Recalculate size
        if (!result.fields.empty()) {
            result.size = static_cast<uint32_t>(
                result.fields.back().offset + result.fields.back().size);
        }
    }

    z3_log("[Structor/Z3] Extracted structure: %zu fields, %u bytes, alignment %u\n",
           result.fields.size(), result.size, result.alignment);
    
    // Dump all fields for debugging
    z3_log("[Structor/Z3] Final field list:\n");
    for (size_t i = 0; i < result.fields.size(); ++i) {
        const auto& f = result.fields[i];
        z3_log("[Structor/Z3]   [%zu] name='%s', offset=0x%llX, size=%u, is_union=%s%s\n",
               i, f.name.c_str(), static_cast<unsigned long long>(f.offset),
               f.size, f.is_union_candidate ? "yes" : "no",
               f.is_padding ? " [padding]" : "");
    }

    return result;
}

void LayoutConstraintBuilder::detect_union_groups(const ::z3::model& model) {
    union_resolutions_.clear();

    std::unordered_map<int, qvector<int>> groups;

    for (size_t i = 0; i < field_vars_.size(); ++i) {
        const auto& fv = field_vars_[i];

        if (get_bool_value(model, fv.selected) &&
            get_bool_value(model, fv.is_union_member)) {
            int group = static_cast<int>(get_int_value(model, fv.union_group));
            if (group >= 0) {
                groups[group].push_back(static_cast<int>(i));
            }
        }
    }

    for (const auto& [group_id, members] : groups) {
        if (members.size() <= 1) continue;

        UnionResolution resolution;
        resolution.union_id = group_id;
        resolution.member_candidate_ids = members;

        // Calculate union offset and size
        sval_t min_offset = SVAL_MAX;
        sval_t max_end = std::numeric_limits<sval_t>::lowest();

        for (int idx : members) {
            const auto& cand = candidates_[field_vars_[idx].candidate_id];
            min_offset = std::min(min_offset, cand.offset);
            max_end = std::max(max_end, cand.offset + static_cast<sval_t>(cand.size));
        }

        resolution.offset = min_offset;
        resolution.size = static_cast<uint32_t>(max_end - min_offset);

        // Create alternative fields
        for (int idx : members) {
            const auto& cand = candidates_[field_vars_[idx].candidate_id];
            resolution.alternatives.push_back(field_from_candidate(cand, ctx_.type_encoder(),
                                                                    pattern_ ? &pattern_->all_accesses : nullptr));
        }

        union_resolutions_.push_back(std::move(resolution));
    }
}

SynthField LayoutConstraintBuilder::create_union_field(
    const qvector<int>& overlapping_ids,
    const ::z3::model& model)
{
    (void)model;
    SynthField union_field;
    union_field.is_union_candidate = true;

    // Calculate union bounds
    sval_t min_offset = SVAL_MAX;
    sval_t max_end = std::numeric_limits<sval_t>::lowest();
    uint32_t max_size = 0;

    z3_log("[Structor/Z3] create_union_field: %zu overlapping candidates\n", overlapping_ids.size());
    for (int idx : overlapping_ids) {
        const auto& cand = candidates_[field_vars_[idx].candidate_id];
        z3_log("[Structor/Z3]   Member idx=%d, cand_id=%d, offset=0x%llX, size=%u\n",
               idx, field_vars_[idx].candidate_id,
               static_cast<unsigned long long>(cand.offset), cand.size);
        min_offset = std::min(min_offset, cand.offset);
        max_end = std::max(max_end, cand.offset + static_cast<sval_t>(cand.size));
        max_size = std::max(max_size, cand.size);
    }

    // Validate: if no valid members, use offset 0
    if (min_offset == SVAL_MAX || max_end == std::numeric_limits<sval_t>::lowest()) {
        z3_log("[Structor/Z3] WARNING: Invalid min_offset=0x%llX, using 0\n",
               static_cast<unsigned long long>(min_offset));
        min_offset = 0;
        max_end = static_cast<sval_t>(max_size == 0 ? 1 : max_size);
    }
    if (max_end < min_offset) {
        z3_log("[Structor/Z3] WARNING: Invalid max_end=0x%llX (< min_offset), using min+8\n",
               static_cast<unsigned long long>(max_end));
        max_end = min_offset + static_cast<sval_t>(max_size == 0 ? 1 : max_size);
    }

    union_field.offset = min_offset;
    union_field.size = static_cast<uint32_t>(max_end - min_offset);
    set_generated_name(union_field.name,
                       union_field.naming,
                       qstring().sprnt("union_%s", make_offset_suffix(min_offset).c_str()),
                       GeneratedNameKind::UnionField,
                       NameConfidence::High);
    z3_log("[Structor/Z3] Created union: offset=0x%llX, size=%u, name='%s'\n",
           static_cast<unsigned long long>(union_field.offset), union_field.size,
           union_field.name.c_str());
    union_field.semantic = SemanticType::Unknown;

    // Create union type
    union_field.union_members.clear();

    // For now, use the largest member's type as the representative field type,
    // but preserve all alternatives so persistence can materialize a real union.
    qvector<int> ordered_ids = overlapping_ids;
    std::sort(ordered_ids.begin(), ordered_ids.end(), [&](int lhs_idx, int rhs_idx) {
        const auto& lhs = candidates_[field_vars_[lhs_idx].candidate_id];
        const auto& rhs = candidates_[field_vars_[rhs_idx].candidate_id];
        if (lhs.offset != rhs.offset) return lhs.offset < rhs.offset;
        if (lhs.size != rhs.size) return lhs.size > rhs.size;
        if (lhs.type_category != rhs.type_category) {
            return static_cast<int>(lhs.type_category) < static_cast<int>(rhs.type_category);
        }
        if (lhs.primary_func_ea != rhs.primary_func_ea) {
            return lhs.primary_func_ea < rhs.primary_func_ea;
        }
        return lhs.id < rhs.id;
    });

    const FieldCandidate* largest = nullptr;
    std::unordered_map<std::string, size_t> member_name_counts;
    std::unordered_set<int> recorded_accesses;
    for (int idx : ordered_ids) {
        const auto& cand = candidates_[field_vars_[idx].candidate_id];
        SynthField alt = field_from_candidate(cand, ctx_.type_encoder(),
                                              pattern_ ? &pattern_->all_accesses : nullptr);
        SynthField::UnionMember member;
        member.name = alt.name;
        const std::string base_name = member.name.c_str();
        const size_t duplicate_index = member_name_counts[base_name]++;
        if (duplicate_index != 0) {
            member.name.sprnt("%s_alt%zu", base_name.c_str(), duplicate_index);
        }
        member.naming = alt.naming;
        member.offset = cand.offset - min_offset;
        member.size = alt.size;
        member.type = alt.type;
        member.comment = alt.comment;
        union_field.union_members.push_back(std::move(member));

        for (int access_idx : cand.source_access_indices) {
            if (!pattern_ || access_idx < 0 ||
                static_cast<size_t>(access_idx) >= pattern_->all_accesses.size() ||
                !recorded_accesses.insert(access_idx).second) {
                continue;
            }
            union_field.source_accesses.push_back(
                pattern_->all_accesses[static_cast<size_t>(access_idx)]);
        }

        if (!largest || cand.size > largest->size ||
            (cand.size == largest->size &&
             static_cast<int>(cand.type_category) <
                 static_cast<int>(largest->type_category))) {
            largest = &cand;
        }
    }

    if (largest) {
        union_field.type = ctx_.type_encoder().decode(
            largest->type_category,
            largest->size,
            &largest->extended_type
        );
    }

    return union_field;
}

SynthField LayoutConstraintBuilder::create_raw_bytes_field(
    sval_t offset,
    uint32_t size)
{
    return SynthField::create_padding(offset, size);
}

bool LayoutConstraintBuilder::get_bool_value(
    const ::z3::model& model,
    const ::z3::expr& e) const
{
    try {
        ::z3::expr val = model.eval(e, true);
        return val.is_true();
    } catch (...) {
        return false;
    }
}

int64_t LayoutConstraintBuilder::get_int_value(
    const ::z3::model& model,
    const ::z3::expr& e) const
{
    try {
        ::z3::expr val = model.eval(e, true);
        if (val.is_numeral()) {
            return val.get_numeral_int64();
        }
    } catch (...) {
    }
    return 0;
}

bool LayoutConstraintBuilder::candidate_covers_access(
    const FieldCandidate& candidate,
    const FieldAccess& access) const
{
    const auto is_padding_like_candidate = [](const FieldCandidate& cand) {
        return cand.kind == FieldCandidate::Kind::PaddingField ||
               cand.type_category == TypeCategory::RawBytes;
    };

    const auto covers_by_shape = [&](const FieldCandidate& cand) {
        if (cand.kind == FieldCandidate::Kind::ArrayField &&
            cand.type_category != TypeCategory::Struct) {
            const bool supported = pattern_ && std::any_of(
                cand.source_access_indices.begin(), cand.source_access_indices.end(),
                [&](int index) {
                    if (index < 0 || static_cast<size_t>(index) >= pattern_->all_accesses.size()) {
                        return false;
                    }
                    const auto& source = pattern_->all_accesses[static_cast<size_t>(index)];
                    return ArrayAccessEvidence{source.offset, source.size,
                        source.inferred_type, source.semantic_type}.matches(access);
                });
            if (!supported) return false;
        }
        if ((cand.type_category == TypeCategory::Array ||
             cand.type_category == TypeCategory::Struct ||
             cand.type_category == TypeCategory::Union) &&
            cand.kind == FieldCandidate::Kind::DirectAccess) {
            return cand.offset == access.offset && cand.size == access.size;
        }

        if ((cand.kind == FieldCandidate::Kind::DirectAccess ||
             cand.kind == FieldCandidate::Kind::ArrayElement ||
             cand.kind == FieldCandidate::Kind::UnionAlternative) &&
            cand.offset == access.offset && cand.size == access.size &&
            cand.type_category != TypeCategory::RawBytes) {
            TypeCategory access_cat = TypeCategory::Unknown;
            if (!access.inferred_type.empty()) {
                access_cat = ctx_.type_encoder().categorize(access.inferred_type);
            } else {
                access_cat = semantic_to_category(static_cast<int>(access.semantic_type));
            }

            if (access_cat != TypeCategory::Unknown && access_cat != TypeCategory::RawBytes &&
                access_cat != cand.type_category &&
                !types_compatible(cand.type_category, access_cat)) {
                return false;
            }
        }

        return cand.offset <= access.offset &&
               cand.offset + static_cast<sval_t>(cand.size) >=
               access.offset + static_cast<sval_t>(access.size);
    };

    if (!covers_by_shape(candidate)) {
        return false;
    }

    const bool has_non_padding_evidence =
        access.access_type == AccessType::Call ||
        access.access_type == AccessType::AddressTaken ||
        access.is_call_argument;

    if (!has_non_padding_evidence || !is_padding_like_candidate(candidate)) {
        return true;
    }

    for (const auto& other : candidates_) {
        if (is_padding_like_candidate(other)) {
            continue;
        }
        if (covers_by_shape(other)) {
            return false;
        }
    }

    return true;
}

} // namespace structor::z3
