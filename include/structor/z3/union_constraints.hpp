#pragma once

#include <z3++.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace structor::z3 {

/// Solver variables needed to assign one candidate to a union group.
struct UnionConstraintVariables {
    ::z3::expr selected;
    ::z3::expr is_union_member;
    ::z3::expr union_group;
};

/// One solver variable tuple paired with its deterministic storage-origin
/// group. The group is derived from the candidate offset before solving.
struct CanonicalUnionConstraintVariables {
    UnionConstraintVariables variables;
    std::uint32_t canonical_group = 0;
};

struct UnionAlternativeDescriptor {
    std::int64_t offset = 0;
    std::uint32_t size = 0;
    bool mandatory = false;
};

struct UnionAlternativeClusterSummary {
    std::uint32_t count = 0;
    std::int64_t offset = 0;
    std::uint32_t size = 0;
};

/// Find the largest exact-range cluster of mandatory storage
/// interpretations.  Optional candidates do not consume the evidence cap.
/// Complexity: O(C log C) time and O(C) space.
[[nodiscard]] inline UnionAlternativeClusterSummary
largest_mandatory_union_cluster(
    std::span<const UnionAlternativeDescriptor> descriptors)
{
    std::vector<UnionAlternativeDescriptor> mandatory;
    mandatory.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        if (descriptor.mandatory) {
            mandatory.push_back(descriptor);
        }
    }
    std::sort(mandatory.begin(), mandatory.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.offset != rhs.offset) return lhs.offset < rhs.offset;
        return lhs.size < rhs.size;
    });

    UnionAlternativeClusterSummary result;
    std::size_t begin = 0;
    while (begin < mandatory.size()) {
        std::size_t end = begin + 1;
        while (end < mandatory.size() &&
               mandatory[end].offset == mandatory[begin].offset &&
               mandatory[end].size == mandatory[begin].size) {
            ++end;
        }
        const auto count = static_cast<std::uint32_t>(end - begin);
        if (count > result.count) {
            result = {count, mandatory[begin].offset, mandatory[begin].size};
        }
        begin = end;
    }
    return result;
}

/// Return the number of representable union-group identifiers. A solved union
/// becomes one materialized root field, so max_fields is the semantic group
/// bound; the candidate count supplies the tighter instance-specific bound.
///
/// Complexity: O(1) time and O(1) space.
[[nodiscard]] inline constexpr std::uint32_t union_group_capacity(
    std::size_t candidate_count,
    std::uint32_t max_fields) noexcept
{
    const std::size_t z3_int_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    const std::size_t capacity = std::min(
        candidate_count,
        std::min(static_cast<std::size_t>(max_fields), z3_int_limit));
    return static_cast<std::uint32_t>(capacity);
}

/// Count unordered candidate-pair relations without overflowing uint64_t.
[[nodiscard]] inline constexpr std::optional<std::uint64_t>
checked_candidate_pair_relation_count(std::uint64_t candidate_count) noexcept
{
    if (candidate_count < 2) {
        return std::uint64_t{0};
    }
    std::uint64_t lhs = candidate_count;
    std::uint64_t rhs = candidate_count - 1;
    if ((lhs & 1U) == 0) {
        lhs /= 2;
    } else {
        rhs /= 2;
    }
    if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
        return std::nullopt;
    }
    return lhs * rhs;
}

/// Count candidate-to-storage-group occupancy relations. Canonical groups are
/// keyed by field offset, so every candidate occurs in exactly one relation.
[[nodiscard]] inline constexpr std::optional<std::uint64_t>
checked_union_cardinality_relation_count(
    std::uint64_t candidate_count,
    std::uint32_t max_fields,
    bool allow_unions) noexcept
{
    (void)max_fields;
    if (!allow_unions || candidate_count == 0) {
        return std::uint64_t{0};
    }
    return candidate_count;
}

/// Total statically constructed binary/cardinality relation budget.
[[nodiscard]] inline constexpr std::optional<std::uint64_t>
checked_layout_relation_count(
    std::uint64_t candidate_count,
    std::uint32_t max_fields,
    bool allow_unions) noexcept
{
    const auto pairs =
        checked_candidate_pair_relation_count(candidate_count);
    const auto union_relations = checked_union_cardinality_relation_count(
        candidate_count, max_fields, allow_unions);
    if (!pairs || !union_relations ||
        *pairs > std::numeric_limits<std::uint64_t>::max() -
                     *union_relations) {
        return std::nullopt;
    }
    return *pairs + *union_relations;
}

/// Constrain the sentinel/group domain for one candidate. Non-members use -1;
/// selected members use the group fixed by their storage origin. Unselected
/// candidates are canonicalized as non-members, eliminating group-label and
/// unselected-state symmetry.
[[nodiscard]] inline ::z3::expr canonical_union_group_domain_constraint(
    const UnionConstraintVariables& variables,
    std::uint32_t canonical_group)
{
    const ::z3::expr non_member_domain =
        !variables.is_union_member && variables.union_group == -1;
    const ::z3::expr member_domain =
        variables.selected && variables.is_union_member &&
        variables.union_group == static_cast<int>(canonical_group);
    return non_member_domain || member_domain;
}

/// Build one hard cardinality constraint per canonical storage-origin group.
/// Every candidate is visited only in its precomputed group; unselected
/// candidates and non-members do not consume the alternatives cap.
///
/// Complexity: O(C+G) time, O(C+G) auxiliary space, and O(C+G) Z3 AST nodes
/// for C candidate variables and G canonical groups.
[[nodiscard]] inline std::vector<::z3::expr>
canonical_union_alternative_limit_constraints(
    ::z3::context& ctx,
    std::span<const CanonicalUnionConstraintVariables> variables,
    std::uint32_t canonical_group_count,
    std::uint32_t max_union_alternatives)
{
    std::vector<std::vector<std::size_t>> candidates_by_group(
        canonical_group_count);
    for (std::size_t i = 0; i < variables.size(); ++i) {
        const std::uint32_t group = variables[i].canonical_group;
        if (group >= canonical_group_count) {
            throw std::out_of_range(
                "canonical union group is outside the declared group count");
        }
        candidates_by_group[group].push_back(i);
    }

    std::vector<::z3::expr> constraints;
    constraints.reserve(canonical_group_count);
    for (std::uint32_t group = 0; group < canonical_group_count; ++group) {
        ::z3::expr selected_member_count = ctx.int_val(0);
        for (const std::size_t candidate_index : candidates_by_group[group]) {
            const auto& variable = variables[candidate_index].variables;
            const ::z3::expr in_group =
                variable.selected && variable.is_union_member &&
                variable.union_group == static_cast<int>(group);
            selected_member_count = selected_member_count +
                ::z3::ite(in_group, ctx.int_val(1), ctx.int_val(0));
        }
        constraints.push_back(
            selected_member_count <= ctx.int_val(max_union_alternatives));
    }
    return constraints;
}

} // namespace structor::z3
