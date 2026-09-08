#include "structor/z3/type_variable_identity.hpp"

#include <z3++.h>

#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>

using namespace structor::z3;

namespace {

struct ConstantKeyHash {
    std::size_t operator()(const TypeVariableKey&) const noexcept { return 0; }
};

template<class Interner>
void check_exact_local_memory_and_name_keys() {
    Interner variables;
    const auto low_function = variables.local(0x100001234ULL, 2, 0);
    const auto high_function = variables.local(0x200001234ULL, 2, 0);
    const auto next_ssa = variables.local(0x100001234ULL, 2, 1);
    const auto next_local = variables.local(0x100001234ULL, 3, 0);
    assert(low_function != high_function);
    assert(low_function != next_ssa);
    assert(low_function != next_local);
    assert(variables.local(0x100001234ULL, 2, 0) == low_function);
    assert(variables.local(0x100001234ULL, 2, 1) == next_ssa);

    // The old XOR-only registry collapses these distinct byte ranges on
    // identity-hashing standard libraries, including the tested libc++ build.
    const auto first_range = variables.memory(0x1000, 0, 4);
    const auto second_range = variables.memory(0x1008, 4, 4);
    const auto different_width = variables.memory(0x1000, 0, 8);
    const auto negative_offset = variables.memory(0x1000, -4, 4);
    assert(first_range != second_range);
    assert(first_range != different_width);
    assert(first_range != negative_offset);
    assert(variables.memory(0x1000, 0, 4) == first_range);
    assert(variables.memory(0x1008, 4, 4) == second_range);

    std::string borrowed_name = "shared_temporary";
    const auto first_temp = variables.named(0x100001234ULL, borrowed_name);
    borrowed_name.assign("changed_after_interning");
    const auto second_temp = variables.named(0x200001234ULL, "shared_temporary");
    const auto renamed_temp = variables.named(0x100001234ULL, borrowed_name);
    assert(first_temp != second_temp);
    assert(first_temp != renamed_temp);
    assert(variables.named(0x100001234ULL, "shared_temporary") == first_temp);

    std::unordered_set<TypeVariableIdentity, TypeVariableIdentityHash> distinct{
        low_function, high_function, next_ssa, next_local,
        first_range, second_range, different_width, negative_offset,
        first_temp, second_temp, renamed_temp};
    assert(distinct.size() == 11);
    assert(variables.size() == distinct.size());
}

template<class Interner>
void check_expression_identity_and_lifetime() {
    Interner variables;
    int first_node = 0;
    int second_node = 0;
    bool rejected_unscoped = false;
    try {
        (void)variables.expression(&first_node);
    } catch (const std::logic_error&) {
        rejected_unscoped = true;
    }
    assert(rejected_unscoped);

    variables.begin_expression_pass();
    const auto local = variables.local(0x100001234ULL, 0);
    const auto first = variables.expression(&first_node);
    const auto second = variables.expression(&second_node);
    assert(first != second);
    assert(first == variables.expression(&first_node));
    assert(variables.size() == 3);

    // Identical storage can represent a replaced or mutated ctree on the next
    // pass. Returned identities remain distinguishable after the keys expire.
    variables.begin_expression_pass();
    assert(variables.size() == 1);
    const auto recycled_address = variables.expression(&first_node);
    assert(recycled_address != first);
    assert(local == variables.local(0x100001234ULL, 0));
    assert(variables.size() == 2);

    bool rejected_null = false;
    try {
        (void)variables.expression(nullptr);
    } catch (const std::invalid_argument&) {
        rejected_null = true;
    }
    assert(rejected_null);
    variables.end_expression_pass();
    assert(variables.size() == 1);
    bool rejected_finished = false;
    try {
        (void)variables.expression(&first_node);
    } catch (const std::logic_error&) {
        rejected_finished = true;
    }
    assert(rejected_finished);
}

void check_distinct_scopes_and_solver_symbols() {
    TypeVariableInterner first_extractor;
    TypeVariableInterner second_extractor;
    const auto first = first_extractor.named(0x100001234ULL, "same_name");
    const auto same_label_other_function =
        first_extractor.named(0x200001234ULL, "same_name");
    const auto same_local_id_other_extractor =
        second_extractor.named(0x100001234ULL, "same_name");
    assert(first.ordinal == same_local_id_other_extractor.ordinal);
    assert(first.module_token == same_local_id_other_extractor.module_token);
    assert(first != same_local_id_other_extractor);

    ::z3::context ctx;
    const auto left = ctx.int_const(type_variable_solver_symbol(first).c_str());
    const auto right = ctx.int_const(
        type_variable_solver_symbol(same_label_other_function).c_str());
    const auto foreign = ctx.int_const(
        type_variable_solver_symbol(same_local_id_other_extractor).c_str());
    assert(!::z3::eq(left, right));
    assert(!::z3::eq(left, foreign));
    assert(::z3::eq(left, ctx.int_const(type_variable_solver_symbol(first).c_str())));

    ::z3::solver solver(ctx);
    solver.add(left == 1);
    solver.add(right == 2);
    solver.add(foreign == 3);
    assert(solver.check() == ::z3::sat);
    solver.add(left == 2);
    assert(solver.check() == ::z3::unsat);
}

void check_invalid_solver_identity_is_rejected() {
    for (const TypeVariableIdentity invalid :
         std::array{TypeVariableIdentity{}, TypeVariableIdentity{0, 1, 1},
                    TypeVariableIdentity{1, 0, 1}, TypeVariableIdentity{1, 1, 0}}) {
        bool rejected = false;
        try {
            (void)type_variable_solver_symbol(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
}

void check_fresh_factory_identities_do_not_alias() {
    const auto first = fresh_type_variable_identity();
    const auto second = fresh_type_variable_identity();
    assert(first.valid() && second.valid());
    assert(first != second);
    const auto copied = first;
    assert(copied == first);
    assert(type_variable_solver_symbol(copied) == type_variable_solver_symbol(first));
    assert(type_variable_solver_symbol(second) != type_variable_solver_symbol(first));
}

} // namespace

int main() {
    check_exact_local_memory_and_name_keys<TypeVariableInterner>();
    check_exact_local_memory_and_name_keys<ExactTypeVariableInterner<ConstantKeyHash>>();
    check_expression_identity_and_lifetime<TypeVariableInterner>();
    check_expression_identity_and_lifetime<ExactTypeVariableInterner<ConstantKeyHash>>();
    check_distinct_scopes_and_solver_symbols();
    check_invalid_solver_identity_is_rejected();
    check_fresh_factory_identities_do_not_alias();
    std::cout << "[PASS] exact type-variable identity and solver-symbol tests\n";
    return 0;
}
