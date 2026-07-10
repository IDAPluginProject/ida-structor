#pragma once

#include <z3++.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace structor::z3 {

/// Solver variables needed to assign one candidate to a union group.
struct UnionConstraintVariables {
    ::z3::expr selected;
    ::z3::expr is_union_member;
    ::z3::expr union_group;
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

/// Constrain the sentinel/group domain. Non-members use -1; selected members
/// use a concrete group in [0, max_union_groups). Unselected candidates are
/// canonicalized as non-members to eliminate irrelevant model symmetry.
[[nodiscard]] inline ::z3::expr union_group_domain_constraint(
    const UnionConstraintVariables& variables,
    std::uint32_t max_union_groups)
{
    const ::z3::expr non_member_domain =
        !variables.is_union_member && variables.union_group == -1;
    const ::z3::expr member_domain =
        variables.selected && variables.is_union_member &&
        variables.union_group >= 0 &&
        variables.union_group < static_cast<int>(max_union_groups);
    return non_member_domain || member_domain;
}

/// Limit the number of selected members assigned to one concrete union group.
/// Unselected candidates and non-members do not consume the alternatives cap.
///
/// Complexity: O(C) time and O(C) Z3 AST nodes for C candidate variables.
[[nodiscard]] inline ::z3::expr union_alternative_limit_constraint(
    ::z3::context& ctx,
    std::span<const UnionConstraintVariables> variables,
    std::uint32_t union_group,
    std::uint32_t max_union_alternatives)
{
    ::z3::expr selected_member_count = ctx.int_val(0);
    for (const auto& variable : variables) {
        const ::z3::expr in_group =
            variable.selected && variable.is_union_member &&
            variable.union_group == static_cast<int>(union_group);
        selected_member_count = selected_member_count +
            ::z3::ite(in_group, ctx.int_val(1), ctx.int_val(0));
    }
    return selected_member_count <= ctx.int_val(max_union_alternatives);
}

/// Build one hard per-group cardinality constraint. Group capacity and member
/// capacity are deliberately independent dimensions.
///
/// Complexity: O(G*C) time/AST nodes and O(G) returned expressions, where
/// G=min(candidate_count,max_fields).
[[nodiscard]] inline std::vector<::z3::expr> union_alternative_limit_constraints(
    ::z3::context& ctx,
    std::span<const UnionConstraintVariables> variables,
    std::uint32_t max_union_groups,
    std::uint32_t max_union_alternatives)
{
    std::vector<::z3::expr> constraints;
    constraints.reserve(max_union_groups);
    for (std::uint32_t group = 0; group < max_union_groups; ++group) {
        constraints.push_back(union_alternative_limit_constraint(
            ctx, variables, group, max_union_alternatives));
    }
    return constraints;
}

} // namespace structor::z3
