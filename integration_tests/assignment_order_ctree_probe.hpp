#pragma once

// Constructed SDK ctree witnesses for argument orders that optimized native
// fixtures may split into statements. The production collector traverses them.
#include "ctree_probe_builder.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <set>

namespace structor::testing {

struct AssignmentOrderObservation {
    std::string name;
    bool expects_base_access = false;
    bool matches_expected = false;
    bool original_body_restored = false;
    AccessPattern pattern;
    std::string error;
};

inline std::vector<AssignmentOrderObservation> probe_assignment_order_ctree(cfunc_t* cfunc) {
    using namespace ctree_probe_detail;
    if (!cfunc || !cfunc->mba || !cfunc->get_lvars() ||
        cfunc->body.op != cit_block || cfunc->argidx.size() != 4) {
        throw std::invalid_argument("assignment probe requires a four-argument carrier");
    }
    std::array<int, 4> arguments{};
    for (size_t i = 0; i < arguments.size(); ++i) {
        arguments[i] = cfunc->argidx[i];
        if (arguments[i] < 0 || static_cast<size_t>(arguments[i]) >= cfunc->get_lvars()->size()) {
            throw std::invalid_argument("assignment carrier has an invalid local");
        }
        const tinfo_t& type = cfunc->get_lvars()->at(arguments[i]).type();
        if (!type.is_integral() || type.get_size() != (i == 3 ? 4 : get_ptr_size())) {
            throw std::invalid_argument("assignment carrier has an unexpected local type");
        }
        for (size_t j = 0; j < i; ++j) {
            if (arguments[j] == arguments[i]) {
                throw std::invalid_argument("assignment carrier locals must be distinct");
            }
        }
    }
    Builder builder(*cfunc, arguments);
    tinfo_t word, byte, boolean;
    word.create_simple_type(BTF_UINT32);
    byte.create_simple_type(BTF_UINT8);
    boolean.create_simple_type(BTF_BOOL);
    tinfo_t word_pointer, byte_pointer;
    word_pointer.create_ptr(word);
    byte_pointer.create_ptr(byte);
    const auto reference_index = [&] {
        return builder.unary(cot_ref, builder.variable(3), word_pointer);
    };
    const auto load = [&](uint64 offset) {
        auto address = builder.binary(cot_add,
            builder.unary(cot_cast, builder.variable(0), byte_pointer),
            builder.number(offset), byte_pointer);
        auto delta = builder.binary(cot_mul, builder.variable(3), builder.number(4), word);
        address = builder.binary(cot_add, std::move(address), std::move(delta), byte_pointer);
        auto value = builder.unary(cot_ptr,
            builder.unary(cot_cast, std::move(address), word_pointer), word);
        value->ptrsize = 4;
        return value;
    };
    const auto call = [&](std::vector<Expression> operands) {
        auto args = std::make_unique<carglist_t>();
        for (auto& operand : operands) {
            carg_t& argument = args->push_back();
            argument.swap(*operand);
            argument.formal_type = argument.type;
        }
        return Expression(call_helper(word, args.release(), "assignment_order_effect"));
    };
    const auto nested_call = [&](bool references_index) {
        std::vector<Expression> operands;
        operands.push_back(references_index ? reference_index() : builder.number(0));
        return call(std::move(operands));
    };
    const auto guarded_body = [&](int variant) {
        auto body = builder.block();
        if (variant == 1) builder.append(*body, builder.expression(nested_call(true)));
        auto guarded = builder.block();
        std::vector<Expression> operands;
        if (variant == 0) {
            operands.push_back(reference_index());
            operands.push_back(load(64));
        } else {
            operands.push_back(load(64));
            if (variant == 1) {
                operands.push_back(reference_index());
            } else if (variant == 2 || variant == 4) {
                operands.push_back(nested_call(variant == 2));
            } else {
                auto temporary = builder.variable(2);
                const tinfo_t temporary_type = temporary->type;
                operands.push_back(builder.binary(cot_asg, std::move(temporary),
                    builder.unary(cot_cast, reference_index(), temporary_type), temporary_type));
                auto target = builder.unary(cot_ptr,
                    builder.unary(cot_cast, builder.variable(2), word_pointer), word);
                target->ptrsize = 4;
                operands.push_back(builder.binary(cot_asg, std::move(target), builder.number(7), word));
            }
        }
        builder.append(*guarded, builder.expression(call(std::move(operands))));
        if (variant == 0 || variant == 1) {
            // The outer call may change an escaped index. Its following load
            // must not retain the comparison's old version.
            builder.append(*guarded, builder.expression(load(128)));
        }
        auto branch = std::make_unique<cinsn_t>();
        branch->op = cit_if;
        branch->ea = cfunc->entry_ea;
        branch->cif = new cif_t;
        branch->cif->expr.swap(*builder.binary(cot_ult, builder.variable(3), builder.number(4), boolean));
        branch->cif->ithen = guarded.release();
        builder.append(*body, std::move(branch));
        builder.append(*body, builder.return_zero());
        return body;
    };
    const std::array<const char*, 5> names{
        "direct_address_before_load", "previously_escaped_load_before_call",
        "later_nested_mutator_invalidates_load", "split_sibling_escape_and_write",
        "unescaped_index_survives_unrelated_nested_call"};
    std::vector<AssignmentOrderObservation> observations;
    const cblock_t* original_body = cfunc->body.cblock;
    for (size_t i = 0; i < names.size(); ++i) {
        AssignmentOrderObservation observation;
        observation.name = names[i];
        observation.expects_base_access = i == 0 || i == 1 || i == 4;
        try {
            auto replacement = guarded_body(static_cast<int>(i));
            {
                ScopedBodySwap restore(*cfunc, *replacement);
                SynthOptions options;
                options.min_accesses = 1;
                options.vtable_detection = false;
                AccessCollector collector(options);
                observation.pattern = collector.collect(cfunc, arguments[0]);
            }
            if (!observation.expects_base_access) {
                observation.matches_expected = observation.pattern.accesses.empty();
            } else {
                std::set<sval_t> offsets;
                bool widths_match = true;
                for (const auto& access : observation.pattern.accesses) {
                    offsets.insert(access.offset);
                    widths_match &= access.size == 4;
                }
                observation.matches_expected = widths_match &&
                    offsets == std::set<sval_t>{64, 68, 72, 76};
            }
        } catch (const std::exception& exception) {
            observation.error = exception.what();
        } catch (...) {
            observation.error = "unknown exception while collecting assignment probe";
        }
        observation.original_body_restored =
            cfunc->body.op == cit_block && cfunc->body.cblock == original_body;
        observations.push_back(std::move(observation));
    }
    return observations;
}

} // namespace structor::testing
