#pragma once

#if !defined(STRUCTOR_LIVE_TEST_HOOKS)
#error "Constructed ctree probes require STRUCTOR_LIVE_TEST_HOOKS"
#endif

#include "structor/access_collector.hpp"
#include <array>
#include <memory>

namespace structor::testing::ctree_probe_detail {

using Expression = std::unique_ptr<cexpr_t>;
using Statement = std::unique_ptr<cinsn_t>;

// No lvars, types, saved user data, maturity, or cfunc caches are changed.
// Swap restores the original body on both success and exception unwinding.
class ScopedBodySwap {
public:
    ScopedBodySwap(cfunc_t& owner, cinsn_t& replacement)
        : owner_(owner), replacement_(replacement) {
        owner_.body.swap(replacement_);
    }
    ~ScopedBodySwap() noexcept { owner_.body.swap(replacement_); }
    ScopedBodySwap(const ScopedBodySwap&) = delete;
    ScopedBodySwap& operator=(const ScopedBodySwap&) = delete;

private:
    cfunc_t& owner_;
    cinsn_t& replacement_;
};

class Builder {
public:
    Builder(cfunc_t& owner, const std::array<int, 4>& arguments)
        : owner_(owner), arguments_(arguments) {}

    Expression variable(size_t argument) const {
        auto expression = std::make_unique<cexpr_t>(cot_var, nullptr);
        expression->v.mba = owner_.mba;
        expression->v.idx = arguments_.at(argument);
        expression->type = owner_.get_lvars()->at(expression->v.idx).type();
        expression->ea = owner_.entry_ea;
        return expression;
    }

    Expression number(uint64 value, type_sign_t sign = type_unsigned) const {
        auto expression = std::make_unique<cexpr_t>();
        expression->put_number(&owner_, value, 4, sign);
        expression->ea = owner_.entry_ea;
        return expression;
    }

    Expression binary(ctype_t opcode, Expression lhs, Expression rhs,
                      const tinfo_t& type) const {
        auto expression = std::make_unique<cexpr_t>(opcode, nullptr);
        expression->x = lhs.release();
        expression->y = rhs.release();
        expression->type = type;
        expression->ea = owner_.entry_ea;
        return expression;
    }

    Expression unary(ctype_t opcode, Expression operand, const tinfo_t& type) const {
        auto expression = std::make_unique<cexpr_t>(opcode, nullptr);
        expression->x = operand.release();
        expression->type = type;
        expression->ea = owner_.entry_ea;
        return expression;
    }

    Statement block() const {
        auto statement = std::make_unique<cinsn_t>();
        statement->op = cit_block;
        statement->cblock = new cblock_t;
        statement->ea = owner_.entry_ea;
        return statement;
    }

    void append(cinsn_t& block, Statement child) const {
        block.new_insn(owner_.entry_ea).swap(*child);
    }

    Statement expression(Expression value) const {
        auto statement = std::make_unique<cinsn_t>();
        statement->op = cit_expr;
        statement->cexpr = value.release();
        statement->ea = owner_.entry_ea;
        return statement;
    }

    Statement return_zero() const {
        auto statement = std::make_unique<cinsn_t>();
        statement->op = cit_return;
        statement->creturn = new creturn_t;
        statement->creturn->expr.put_number(&owner_, 0, get_ptr_size(), type_unsigned);
        statement->ea = owner_.entry_ea;
        return statement;
    }

private:
    cfunc_t& owner_;
    const std::array<int, 4>& arguments_;
};

} // namespace structor::testing::ctree_probe_detail
