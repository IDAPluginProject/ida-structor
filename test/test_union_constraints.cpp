/**
 * @file test_union_constraints.cpp
 * @brief Tests the production union-domain/cardinality constraint helpers.
 */

#include <z3++.h>

#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "structor/z3/union_constraints.hpp"

namespace {

using structor::z3::UnionConstraintVariables;
using structor::z3::union_alternative_limit_constraints;
using structor::z3::union_group_capacity;
using structor::z3::union_group_domain_constraint;
using structor::z3::UnionAlternativeDescriptor;
using structor::z3::checked_layout_relation_count;
using structor::z3::largest_mandatory_union_cluster;

std::vector<UnionConstraintVariables> make_variables(
    z3::context& ctx,
    std::size_t count)
{
    std::vector<UnionConstraintVariables> variables;
    variables.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::string prefix = "u" + std::to_string(i) + "_";
        variables.push_back({
            ctx.bool_const((prefix + "selected").c_str()),
            ctx.bool_const((prefix + "member").c_str()),
            ctx.int_const((prefix + "group").c_str()),
        });
    }
    return variables;
}

void add_production_constraints(
    z3::solver& solver,
    z3::context& ctx,
    const std::vector<UnionConstraintVariables>& variables,
    std::uint32_t max_groups,
    std::uint32_t max_alternatives)
{
    for (const auto& variable : variables) {
        solver.add(union_group_domain_constraint(variable, max_groups));
    }
    for (const auto& constraint : union_alternative_limit_constraints(
             ctx, variables, max_groups, max_alternatives)) {
        solver.add(constraint);
    }
}

void constrain_member(
    z3::solver& solver,
    const UnionConstraintVariables& variable,
    bool selected,
    int group)
{
    solver.add(selected ? variable.selected : !variable.selected);
    solver.add(variable.is_union_member);
    solver.add(variable.union_group == group);
}

void constrain_non_member(
    z3::solver& solver,
    const UnionConstraintVariables& variable,
    bool selected)
{
    solver.add(selected ? variable.selected : !variable.selected);
    solver.add(!variable.is_union_member);
    solver.add(variable.union_group == -1);
}

void test_group_capacity_uses_candidates_and_fields() {
    assert(union_group_capacity(12, 5) == 5);
    assert(union_group_capacity(3, 50) == 3);
    assert(union_group_capacity(0, 50) == 0);
}

void test_relation_budget_counts_union_ast_and_checks_overflow() {
    assert(checked_layout_relation_count(4, 4, false) == 6);
    assert(checked_layout_relation_count(4, 4, true) == 10);
    assert(checked_layout_relation_count(1000, 4096, true) == 500500);
    assert(!checked_layout_relation_count(
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint32_t>::max(),
        true));
}

void test_alternative_limit_is_per_group() {
    z3::context ctx;
    z3::solver solver(ctx);
    auto variables = make_variables(ctx, 6);
    add_production_constraints(solver, ctx, variables, 6, 2);

    for (std::size_t i = 0; i < variables.size(); ++i) {
        constrain_member(solver, variables[i], true, static_cast<int>(i / 2));
    }
    assert(solver.check() == z3::sat);
}

void test_third_selected_alternative_is_rejected() {
    z3::context ctx;
    z3::solver solver(ctx);
    auto variables = make_variables(ctx, 3);
    add_production_constraints(solver, ctx, variables, 3, 2);

    for (const auto& variable : variables) {
        constrain_member(solver, variable, true, 0);
    }
    assert(solver.check() == z3::unsat);
}

void test_unselected_alternative_does_not_consume_limit() {
    z3::context ctx;
    z3::solver solver(ctx);
    auto variables = make_variables(ctx, 3);
    add_production_constraints(solver, ctx, variables, 3, 2);

    constrain_member(solver, variables[0], true, 0);
    constrain_member(solver, variables[1], true, 0);
    constrain_non_member(solver, variables[2], false);
    assert(solver.check() == z3::sat);
}

void test_unselected_candidate_cannot_carry_phantom_union_state() {
    z3::context ctx;
    z3::solver solver(ctx);
    auto variables = make_variables(ctx, 1);
    add_production_constraints(solver, ctx, variables, 1, 1);

    constrain_member(solver, variables[0], false, 0);
    assert(solver.check() == z3::unsat);
}

void test_zero_alternatives_disables_selected_union_members() {
    z3::context ctx;
    z3::solver solver(ctx);
    auto variables = make_variables(ctx, 1);
    add_production_constraints(solver, ctx, variables, 1, 0);

    constrain_member(solver, variables[0], true, 0);
    assert(solver.check() == z3::unsat);
}

void test_group_domain_is_independent_of_alternative_limit() {
    z3::context ctx;
    auto variables = make_variables(ctx, 1);

    z3::solver valid(ctx);
    add_production_constraints(valid, ctx, variables, 6, 2);
    constrain_member(valid, variables[0], true, 5);
    assert(valid.check() == z3::sat);

    z3::solver out_of_range(ctx);
    add_production_constraints(out_of_range, ctx, variables, 6, 2);
    constrain_member(out_of_range, variables[0], true, 6);
    assert(out_of_range.check() == z3::unsat);
}

void test_mandatory_cluster_count_is_exact_range_and_ignores_optional() {
    const std::vector<UnionAlternativeDescriptor> descriptors = {
        {8, 4, true}, {8, 4, true}, {8, 4, true},
        {8, 8, true}, {16, 4, true}, {8, 4, false},
    };
    const auto largest = largest_mandatory_union_cluster(descriptors);
    assert(largest.count == 3);
    assert(largest.offset == 8);
    assert(largest.size == 4);
}

} // namespace

int main() {
    test_group_capacity_uses_candidates_and_fields();
    test_relation_budget_counts_union_ast_and_checks_overflow();
    test_alternative_limit_is_per_group();
    test_third_selected_alternative_is_rejected();
    test_unselected_alternative_does_not_consume_limit();
    test_unselected_candidate_cannot_carry_phantom_union_state();
    test_zero_alternatives_disables_selected_union_members();
    test_group_domain_is_independent_of_alternative_limit();
    test_mandatory_cluster_count_is_exact_range_and_ignores_optional();
    std::cout << "[PASS] production union constraint helper tests\n";
    return 0;
}
