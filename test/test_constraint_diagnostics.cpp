#include "mock_ida.hpp"
#include "structor/z3/constraint_tracker.hpp"

#include <cassert>
#include <iostream>

using namespace structor::z3;

int main() {
    qvector<ConstraintProvenance> diagnostics;
    {
        ::z3::context context;
        ::z3::solver solver(context);
        ConstraintTracker tracker(context);
        const auto variable = context.int_const("diagnostic_lifetime_value");
        auto positive = ConstraintProvenance::for_access(
            0x100001234ULL, 0x100001238ULL, 7, "positive observation");
        auto negative = ConstraintProvenance::make(
            0x200001234ULL, "negative preference", true, 13,
            ConstraintProvenance::Kind::TypeMatch);
        tracker.add_hard(solver, variable > 0, positive);
        tracker.add_soft(solver, variable < 0, negative, 13);
        const auto assumptions = tracker.get_all_literals();
        assert(solver.check(assumptions) == ::z3::unsat);
        const auto original = tracker.analyze_unsat_core(solver.unsat_core());
        assert(original.size() == 2);
        diagnostics = copy_constraint_diagnostics(original);
        assert(diagnostics.size() == original.size());
        for (std::size_t i = 0; i < original.size(); ++i) {
            const auto& source = original[i];
            const auto& detached = diagnostics[i];
            assert(source.tracking_literal.has_value());
            assert(!detached.tracking_literal.has_value());
            assert(source.func_ea == detached.func_ea);
            assert(source.insn_ea == detached.insn_ea);
            assert(source.access_idx == detached.access_idx);
            assert(source.description == detached.description);
            assert(source.is_soft == detached.is_soft);
            assert(source.weight == detached.weight);
            assert(source.kind == detached.kind);
        }
    }

    // Exercise the caller's actual ownership operations after context teardown.
    auto copied = diagnostics;
    auto moved = std::move(copied);
    diagnostics.clear();
    assert(moved.size() == 2);
    for (const auto& diagnostic : moved) {
        assert(!diagnostic.tracking_literal.has_value());
        assert(!diagnostic.description.empty());
    }
    moved.clear();

    {
        ::z3::context context;
        ::z3::solver solver(context);
        ConstraintTracker tracker(context);
        const auto variable = context.int_const("optimizer_diagnostic_value");
        tracker.add_hard(solver, variable > 0,
            ConstraintProvenance::for_access(0x100001234ULL, 0x100001238ULL, 0,
                                              "optimizer positive observation"));
        tracker.add_hard(solver, variable < 0,
            ConstraintProvenance::for_access(0x200001234ULL, 0x200001238ULL, 1,
                                              "optimizer negative observation"));
        ::z3::optimize optimizer(context);
        optimizer.add(solver.assertions());
        const auto hard_literals = tracker.add_hard_literals_to_optimizer(optimizer);
        assert(hard_literals.size() == 2);
        optimizer.minimize(variable);
        assert(optimizer.check() == ::z3::unsat);
        const auto attributed = tracker.analyze_unsat_core(optimizer.unsat_core());
        assert(attributed.size() == 2);
        diagnostics = copy_constraint_diagnostics(attributed);
        for (const auto& source : attributed) {
            assert(source.tracking_literal.has_value());
            assert(!source.is_soft);
            assert(source.access_idx == 0 || source.access_idx == 1);
        }
    }
    assert(diagnostics.size() == 2);
    assert(diagnostics[0].access_idx != diagnostics[1].access_idx);
    for (const auto& diagnostic : diagnostics) {
        assert(!diagnostic.tracking_literal.has_value());
        assert(!diagnostic.description.empty());
    }
    diagnostics.clear();
    std::cout << "[PASS] production UNSAT diagnostics outlive the Z3 context\n";
    return 0;
}
