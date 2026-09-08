#include "structor/z3/instruction_semantics.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>

using namespace structor::z3;

namespace {

void require_independent_types(Z3Context& context, TypeLatticeEncoder& encoder,
                               const TypeVariable& integer,
                               const TypeVariable& floating) {
    TypeConstraintSet constraints(context);
    constraints.add(TypeConstraint::make_is_base(integer, BaseType::Int32));
    constraints.add(TypeConstraint::make_is_base(floating, BaseType::Float32));
    assert(constraints.variables().size() == 2);
    ::z3::solver solver(context.ctx());
    solver.add(constraints.to_z3_hard(encoder));
    assert(solver.check() == ::z3::sat);
    assert(!::z3::eq(constraints.get_z3_var(integer, encoder),
                    constraints.get_z3_var(floating, encoder)));
    solver.add(constraints.get_z3_var(integer, encoder) ==
               constraints.get_z3_var(floating, encoder));
    assert(solver.check() == ::z3::unsat);
}

void check_public_factory_and_diagnostic_compatibility() {
    Z3Context context;
    TypeLatticeEncoder encoder(context);
    // Existing public signatures and diagnostic IDs remain available. A new
    // factory invocation is a new variable, even when every label is reused.
    const auto integer = TypeVariable::for_temp(7, 0x100001234ULL, "shared");
    const auto floating = TypeVariable::for_temp(7, 0x100001234ULL, "shared");
    assert(integer.id == 7 && floating.id == 7);
    assert(integer != floating);
    require_independent_types(context, encoder, integer, floating);

    auto renamed_copy = integer;
    renamed_copy.name = "changed diagnostic label";
    renamed_copy.id = 999;
    assert(renamed_copy == integer);
    assert(TypeVariableHash{}(renamed_copy) == TypeVariableHash{}(integer));
    TypeConstraintSet constraints(context);
    assert(::z3::eq(constraints.get_z3_var(integer, encoder),
                    constraints.get_z3_var(renamed_copy, encoder)));
    constraints.add(TypeConstraint::make_is_base(integer, BaseType::Int32));
    constraints.add(TypeConstraint::make_is_base(renamed_copy, BaseType::Float32));
    assert(constraints.variables().size() == 1);
    ::z3::solver solver(context.ctx());
    solver.add(constraints.to_z3_hard(encoder));
    assert(solver.check() == ::z3::unsat);

    TypeVariable default_first, default_second;
    default_first.id = default_second.id = 7;
    assert(default_first != default_second);
    require_independent_types(context, encoder, default_first, default_second);
}

void check_production_extractor_keys() {
    Z3Context context;
    InstructionSemanticsExtractor extractor(context);
    auto& encoder = extractor.type_encoder();
    cfunc_t first_function, second_function;
    first_function.entry_ea = 0x100001234ULL;
    second_function.entry_ea = 0x200001234ULL;
    const auto first = extractor.get_var_type(&first_function, 2, 0);
    const auto second = extractor.get_var_type(&second_function, 2, 0);
    const auto version = extractor.get_var_type(&first_function, 2, 1);
    assert(first.func_ea == first_function.entry_ea);
    assert(second.func_ea == second_function.entry_ea);
    assert(version.ssa_version == 1);
    assert(first == extractor.get_var_type(&first_function, 2, 0));
    assert(version == extractor.get_var_type(&first_function, 2, 1));
    require_independent_types(context, encoder, first, second);
    require_independent_types(context, encoder, first, version);

    // These keys collide under the previous hash-only cache on libc++. Exact
    // key equality is also stress-tested with forced collisions in the
    // production interner's separate standalone test.
    const auto memory_first = extractor.get_mem_type(0x1000, 0, 4);
    const auto memory_second = extractor.get_mem_type(0x1008, 4, 4);
    const auto memory_width = extractor.get_mem_type(0x1000, 0, 8);
    assert(memory_first == extractor.get_mem_type(0x1000, 0, 4));
    assert(memory_width.mem_size == 8);
    require_independent_types(context, encoder, memory_first, memory_second);
    require_independent_types(context, encoder, memory_first, memory_width);

    std::string label = "shared";
    const auto temporary_first = extractor.get_temp_type(first_function.entry_ea, label.c_str());
    label = "mutated storage";
    const auto temporary_second = extractor.get_temp_type(second_function.entry_ea, "shared");
    assert(temporary_first == extractor.get_temp_type(first_function.entry_ea, "shared"));
    require_independent_types(context, encoder, temporary_first, temporary_second);

    InstructionSemanticsExtractor separate_extractor(context);
    const auto separate = separate_extractor.get_var_type(&first_function, 2, 0);
    assert(first.id == separate.id);
    require_independent_types(context, encoder, first, separate);
}

void check_shared_encoder_sorts_and_context_lifetime() {
    Z3Context context;
    {
        TypeLatticeEncoder first(context), second(context);
        assert(::z3::eq(first.base_type_sort(), second.base_type_sort()));
        assert(first.base_type_sort().sort_kind() == Z3_DATATYPE_SORT);
        assert(::z3::eq(first.type_sort(), second.type_sort()));
        assert(first.type_sort().is_int());
        TypeConstraintSet constraints(context);
        const auto variable = TypeVariable::for_temp(0, BADADDR, "same");
        assert(::z3::eq(constraints.get_z3_var(variable, first),
                        constraints.get_z3_var(variable, second)));
        ::z3::solver solver(context.ctx());
        solver.add(first.type_eq(constraints.get_z3_var(variable, first),
                                 second.encode(InferredType::make_base(BaseType::Int32))));
        assert(solver.check() == ::z3::sat);
    }
    {
        // The enum remains available after every earlier encoder has died.
        TypeLatticeEncoder recreated(context);
        assert(recreated.base_type_sort().sort_kind() == Z3_DATATYPE_SORT);
    }

    Z3Context destination;
    { TypeLatticeEncoder existing(destination); }
    destination = std::move(context);
    { TypeLatticeEncoder after_assignment(destination); }
    Z3Context moved(std::move(destination));
    { TypeLatticeEncoder after_move(moved); }
}

void check_full_production_extraction_and_node_identity() {
    Z3Context context;
    InstructionSemanticsExtractor extractor(context);
    auto& encoder = extractor.type_encoder();
    cfunc_t function;
    function.entry_ea = 0x100001234ULL;
    cblock_t block;
    block.resize(2);
    function.body.op = cit_block;
    function.body.cblock = &block;

    cexpr_t locals[2], casts[2], constants[2];
    for (int index = 0; index < 2; ++index) {
        block[index].op = cit_expr;
        auto& assignment = block[index].cexpr;
        assignment.op = cot_asg;
        assignment.x = &locals[index];
        assignment.y = &casts[index];
        locals[index].op = cot_var;
        locals[index].v.idx = index;
        casts[index].op = cot_cast;
        casts[index].ea = BADADDR;
        casts[index].x = &constants[index];
        casts[index].type.create_simple_type(index == 0 ? BTF_INT32 : BTF_FLOAT);
        constants[index].op = cot_num;
    }

    auto extracted = extractor.extract(&function);
    assert(extracted.hard_count() == 4);
    assert(extracted.soft_count() == 2);
    assert(extractor.stats().expressions_analyzed == 8);
    assert(extractor.stats().constraints_extracted == 6);
    assert(extractor.stats().hard_constraints == 4);
    assert(extractor.stats().soft_constraints == 2);
    ::z3::solver solver(context.ctx());
    solver.add(extracted.to_z3_hard(encoder));
    assert(solver.check() == ::z3::sat);

    qvector<TypeVariable> assignment_rhs, cast_targets;
    for (const auto& constraint : extracted.constraints()) {
        if (constraint.description == "assignment type equality") {
            assert(constraint.var2);
            assignment_rhs.push_back(*constraint.var2);
        } else if (constraint.description == "cast target type") {
            cast_targets.push_back(constraint.var1);
        }
    }
    assert(cast_targets.size() == 2 && assignment_rhs.size() == 2);
    assert(cast_targets[0].name == cast_targets[1].name);
    assert(cast_targets[0] != cast_targets[1]);
    assert(assignment_rhs[0] == cast_targets[0]);
    assert(assignment_rhs[1] == cast_targets[1]);

    const auto previous_local = extractor.get_var_type(&function, 0);
    auto repeated = extractor.extract(&function);
    assert(previous_local == extractor.get_var_type(&function, 0));
    for (const auto& constraint : repeated.constraints()) {
        if (constraint.description != "cast target type") continue;
        for (const auto& old_target : cast_targets) assert(constraint.var1 != old_target);
    }
}

void check_recycled_node_storage_between_expression_passes() {
    Z3Context context;
    InstructionSemanticsExtractor extractor(context);
    auto& encoder = extractor.type_encoder();
    cfunc_t function;
    function.entry_ea = 0x100001234ULL;
    cexpr_t cast, constant;
    cast.op = cot_cast;
    cast.x = &constant;
    cast.ea = BADADDR;
    cast.type.create_simple_type(BTF_INT32);
    const auto first = extractor.extract_expr(&cast, &function);
    cast.type.create_simple_type(BTF_FLOAT);
    const auto second = extractor.extract_expr(&cast, &function);
    assert(first.size() == 1 && second.size() == 1);
    assert(first[0].var1 != second[0].var1);
    require_independent_types(context, encoder, first[0].var1, second[0].var1);
    assert(extractor.extract_expr(&cast, nullptr).empty());
    assert(extractor.extract_expr(nullptr, &function).empty());
    assert(extractor.extract(nullptr).total_count() == 0);
}

void check_storage_width_validation_before_narrowing() {
    Z3Context context;
    InstructionSemanticsExtractor extractor(context);
    cfunc_t function;
    function.entry_ea = 0x100001234ULL;

    tinfo_t integer, void_type, empty_type, zero_width, oversized;
    integer.create_simple_type(BTF_INT32);
    void_type.create_simple_type(BTF_VOID);
    zero_width.create_array(integer, 0);
    tinfo_t wide_element;
    wide_element.create_simple_type(BTF_INT64);
    oversized.create_array(wide_element,
        static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) / 8 + 2);
    assert(oversized.get_size() > std::numeric_limits<uint32_t>::max());

    struct Case { tinfo_t type; std::optional<uint32_t> width; };
    for (const auto& test : std::vector<Case>{
             {integer, 4}, {void_type, std::nullopt}, {empty_type, std::nullopt},
             {zero_width, std::nullopt}, {oversized, std::nullopt}}) {
        for (const auto opcode : {cot_cast, cot_idx, cot_ptr}) {
            cexpr_t expression, operand, index;
            expression.op = opcode;
            expression.x = &operand;
            if (opcode == cot_idx) expression.y = &index;
            if (opcode == cot_cast) {
                expression.type = integer;
                operand.type = test.type;
            } else {
                expression.type = test.type;
            }
            const auto constraints = extractor.extract_expr(&expression, &function);
            std::size_t sizes = 0;
            for (const auto& constraint : constraints) {
                if (constraint.kind != TypeConstraint::Kind::HasSize) continue;
                ++sizes;
                assert(test.width.has_value());
                assert(constraint.size == test.width);
            }
            assert(sizes == (test.width.has_value() ? 1u : 0u));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string selected = argc == 2 ? argv[1] : "";
    bool ran = false;
    const auto run = [&](const char* name, auto check) {
        if (!selected.empty() && selected != name) return;
        check();
        ran = true;
        std::cout << "[PASS] " << name << '\n';
    };
    run("factory", check_public_factory_and_diagnostic_compatibility);
    run("keys", check_production_extractor_keys);
    run("sorts", check_shared_encoder_sorts_and_context_lifetime);
    run("full_extraction", check_full_production_extraction_and_node_identity);
    run("recycled_node", check_recycled_node_storage_between_expression_passes);
    run("storage_widths", check_storage_width_validation_before_narrowing);
    assert(ran);
    std::cout << "Production instruction semantics identity checks passed\n";
}
