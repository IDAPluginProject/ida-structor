/// @file access_collector.cpp
/// @brief Access pattern collection implementation

#include <structor/access_collector.hpp>
#include <structor/config.hpp>

namespace structor {

namespace {

bool is_aggregate_member_container(const cexpr_t* expr) {
    if (!expr || (expr->op != cot_memref && expr->op != cot_memptr) || !expr->x) {
        return false;
    }

    tinfo_t base_type = expr->x->type;
    if (base_type.is_ptr()) {
        base_type = base_type.get_pointed_object();
    }

    udt_type_data_t udt;
    if (!base_type.get_udt_details(&udt)) {
        return false;
    }

    const uint64 member_offset = static_cast<uint64>(expr->m) * 8;
    for (const auto& member : udt) {
        if (member.offset != member_offset || member.type.empty()) {
            continue;
        }

        return member.type.is_array() || member.type.is_struct() || member.type.is_union();
    }

    return false;
}

bool is_assignment_op(ctype_t op) {
    switch (op) {
        case cot_asg:
        case cot_asgbor:
        case cot_asgxor:
        case cot_asgband:
        case cot_asgadd:
        case cot_asgsub:
        case cot_asgmul:
        case cot_asgsshr:
        case cot_asgushr:
        case cot_asgshl:
        case cot_asgsdiv:
        case cot_asgudiv:
        case cot_asgsmod:
        case cot_asgumod:
            return true;
        default:
            return false;
    }
}

bool is_pointee_access(const utils::PtrArithInfo& arith) {
    return arith.through_pointer_alias || arith.base_indirection > 1;
}

bool checked_add_sval(sval_t lhs, sval_t rhs, sval_t& out) {
    if ((rhs > 0 && lhs > std::numeric_limits<sval_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<sval_t>::min() - rhs)) {
        return false;
    }
    out = lhs + rhs;
    return true;
}

bool checked_mul_sval(sval_t lhs, sval_t rhs, sval_t& out) {
    if (lhs == 0 || rhs == 0) {
        out = 0;
        return true;
    }
    if ((lhs == -1 && rhs == std::numeric_limits<sval_t>::min()) ||
        (rhs == -1 && lhs == std::numeric_limits<sval_t>::min())) {
        return false;
    }
    if (lhs > 0) {
        if ((rhs > 0 && lhs > std::numeric_limits<sval_t>::max() / rhs) ||
            (rhs < 0 && rhs < std::numeric_limits<sval_t>::min() / lhs)) {
            return false;
        }
    } else {
        if ((rhs > 0 && lhs < std::numeric_limits<sval_t>::min() / rhs) ||
            (rhs < 0 && lhs < std::numeric_limits<sval_t>::max() / rhs)) {
            return false;
        }
    }
    out = lhs * rhs;
    return true;
}

struct BaseConversionInfo {
    bool has_base = false;
    bool lossy = false;
};

BaseConversionInfo inspect_base_conversions(const cexpr_t* expr, int target_var,
        const std::unordered_set<int>& address_aliases, int depth = 0) {
    if (!expr || depth > 64) return {false, true};
    if (expr->op == cot_var) {
        return {expr->v.idx == target_var || address_aliases.contains(expr->v.idx), false};
    }
    switch (expr->op) {
        case cot_num: return {};
        case cot_cast: {
            auto inner = inspect_base_conversions(expr->x, target_var, address_aliases, depth + 1);
            if (inner.has_base) {
                const auto preserves_address = [](const tinfo_t& type) {
                    return (type.is_ptr() || type.is_integral()) && type.get_size() == get_ptr_size();
                };
                inner.lossy |= !preserves_address(expr->type) || !preserves_address(expr->x->type);
            }
            return inner;
        }
        case cot_ref:
            if (expr->x && (expr->x->op == cot_ptr || expr->x->op == cot_idx ||
                           expr->x->op == cot_memptr || expr->x->op == cot_memref)) {
                return inspect_base_conversions(expr->x->x, target_var, address_aliases, depth + 1);
            }
            return {};
        case cot_ptr: case cot_memptr: case cot_memref: {
            const auto inner = inspect_base_conversions(expr->x, target_var, address_aliases, depth + 1);
            return {false, inner.lossy}; // A loaded value is not the object's base address.
        }
        case cot_add: case cot_sub: case cot_mul: case cot_idx: {
            const auto lhs = inspect_base_conversions(expr->x, target_var, address_aliases, depth + 1);
            const auto rhs = inspect_base_conversions(expr->y, target_var, address_aliases, depth + 1);
            return {expr->op != cot_idx && (lhs.has_base || rhs.has_base), lhs.lossy || rhs.lossy};
        }
        default: return {};
    }
}

// Evaluate bounded integer expressions in their declared bit width. Discarding
// casts changes addresses for truncation, sign extension, and wraparound.
std::optional<sval_t> normalize_integer_value(std::uint64_t raw, const tinfo_t& type) {
    if (!type.is_integral()) return std::nullopt;
    if (type.is_bool()) return raw != 0 ? 1 : 0;
    const auto size = type.get_size();
    if (size == 0 || size == BADSIZE || size > sizeof(sval_t)) return std::nullopt;
    const unsigned bits = static_cast<unsigned>(size * 8);
    const std::uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    raw &= mask;
    const std::uint64_t sign_bit = UINT64_C(1) << (bits - 1);
    if (type.is_signed() && (raw & sign_bit) != 0) {
        raw |= ~mask;
    } else if (!type.is_unsigned() && !type.is_signed() && (raw & sign_bit) != 0) {
        // An unspecified integer sign cannot establish a signed byte delta.
        return std::nullopt;
    } else if (raw > static_cast<std::uint64_t>(std::numeric_limits<sval_t>::max())) {
        return std::nullopt;
    }
    return static_cast<sval_t>(raw);
}

bool find_symbolic_index(const cexpr_t* expr, int target_var, int& index_var) {
    unsigned references = 0;
    unsigned updates = 0;
    std::function<bool(const cexpr_t*, int)> walk = [&](const cexpr_t* item, int depth) {
        if (!item || depth > 64) return false;
        if (item->op == cot_var) {
            if (item->v.idx == target_var) return true;
            if (!item->type.is_integral() || (index_var >= 0 && index_var != item->v.idx)) return false;
            index_var = item->v.idx;
            ++references;
            return true;
        }
        switch (item->op) {
            case cot_num: return true;
            case cot_preinc: case cot_predec: case cot_postinc: case cot_postdec:
                if (!item->x || item->x->op != cot_var || item->x->v.idx == target_var) return false;
                ++updates;
                return walk(item->x, depth + 1);
            case cot_cast: case cot_neg: case cot_bnot:
                return walk(item->x, depth + 1);
            case cot_add: case cot_sub: case cot_mul:
            case cot_shl: case cot_sshr: case cot_ushr:
            case cot_band: case cot_bor: case cot_xor:
                return walk(item->x, depth + 1) && walk(item->y, depth + 1);
            default: return false;
        }
    };
    // A single increment/decrement can be evaluated using its old/new value.
    // Multiple references around a mutation need explicit sequencing analysis.
    return walk(expr, 0) && (updates == 0 || (updates == 1 && references == 1));
}

std::optional<sval_t> evaluate_integer_unary(const cexpr_t* expr, sval_t lhs) {
    const auto raw = static_cast<std::uint64_t>(lhs);
    switch (expr->op) {
        case cot_cast: return normalize_integer_value(raw, expr->type);
        case cot_neg: return normalize_integer_value(UINT64_C(0) - raw, expr->type);
        case cot_bnot: return normalize_integer_value(~raw, expr->type);
        case cot_postinc: case cot_postdec: return normalize_integer_value(raw, expr->type);
        case cot_preinc: return normalize_integer_value(raw + 1, expr->type);
        case cot_predec: return normalize_integer_value(raw - 1, expr->type);
        default: return std::nullopt;
    }
}

std::optional<sval_t> evaluate_integer_binary(const cexpr_t* expr, sval_t lhs, sval_t rhs) {
    const auto unsigned_lhs = static_cast<std::uint64_t>(lhs);
    const auto unsigned_rhs = static_cast<std::uint64_t>(rhs);
    std::uint64_t value;
    switch (expr->op) {
        case cot_add: value = unsigned_lhs + unsigned_rhs; break;
        case cot_sub: value = unsigned_lhs - unsigned_rhs; break;
        case cot_mul: value = unsigned_lhs * unsigned_rhs; break;
        case cot_band: value = unsigned_lhs & unsigned_rhs; break;
        case cot_bor: value = unsigned_lhs | unsigned_rhs; break;
        case cot_xor: value = unsigned_lhs ^ unsigned_rhs; break;
        case cot_shl: case cot_sshr: case cot_ushr: {
            const auto lhs_size = expr->x->type.get_size();
            if (lhs_size == 0 || lhs_size == BADSIZE || lhs_size > sizeof(sval_t) ||
                rhs < 0 || static_cast<std::uint64_t>(rhs) >= lhs_size * 8) {
                return std::nullopt;
            }
            if (expr->op == cot_shl) {
                value = unsigned_lhs << unsigned_rhs;
            } else if (expr->op == cot_sshr) {
                value = static_cast<std::uint64_t>(lhs >> unsigned_rhs);
            } else {
                const auto mask = lhs_size == 8 ? UINT64_MAX : (UINT64_C(1) << (lhs_size * 8)) - 1;
                value = (unsigned_lhs & mask) >> unsigned_rhs;
            }
            break;
        }
        default: return std::nullopt;
    }
    return normalize_integer_value(value, expr->type);
}

std::optional<sval_t> evaluate_index_expression(
        const cexpr_t* expr, int index_var, sval_t index_value, int depth = 0) {
    if (!expr || depth > 64 || !expr->type.is_integral()) return std::nullopt;
    if (expr->op == cot_num) {
        return normalize_integer_value(expr->numval(), expr->type);
    }
    if (expr->op == cot_var) {
        return expr->v.idx == index_var
            ? normalize_integer_value(static_cast<std::uint64_t>(index_value), expr->type)
            : std::nullopt;
    }
    const auto lhs = evaluate_index_expression(expr->x, index_var, index_value, depth + 1);
    if (!lhs.has_value()) return std::nullopt;
    if (expr->op == cot_cast || expr->op == cot_neg || expr->op == cot_bnot ||
        expr->op == cot_preinc || expr->op == cot_predec ||
        expr->op == cot_postinc || expr->op == cot_postdec) {
        return evaluate_integer_unary(expr, *lhs);
    }
    const auto rhs = evaluate_index_expression(expr->y, index_var, index_value, depth + 1);
    return rhs.has_value() ? evaluate_integer_binary(expr, *lhs, *rhs) : std::nullopt;
}

struct SymbolicPointerValue {
    bool has_base = false;
    sval_t offset = 0;
};

std::optional<SymbolicPointerValue> evaluate_symbolic_pointer(
        const cexpr_t* expr, int target_var, int index_var, sval_t index_value, int depth = 0) {
    if (!expr || depth > 64) return std::nullopt;
    if (expr->op == cot_var && expr->v.idx == target_var) {
        return SymbolicPointerValue{true, 0};
    }
    if (expr->op == cot_var || expr->op == cot_num) {
        const auto scalar = evaluate_index_expression(expr, index_var, index_value, depth);
        return scalar.has_value()
            ? std::optional<SymbolicPointerValue>({false, *scalar}) : std::nullopt;
    }
    const auto lhs = evaluate_symbolic_pointer(expr->x, target_var, index_var, index_value, depth + 1);
    if (!lhs.has_value()) return std::nullopt;
    auto address_width = [](const tinfo_t& type) {
        return (type.is_integral() || type.is_ptr()) && type.get_size() == get_ptr_size();
    };
    if (expr->op == cot_cast || expr->op == cot_neg || expr->op == cot_bnot ||
        expr->op == cot_preinc || expr->op == cot_predec ||
        expr->op == cot_postinc || expr->op == cot_postdec) {
        if (!lhs->has_base) {
            const auto scalar = evaluate_integer_unary(expr, lhs->offset);
            return scalar.has_value()
                ? std::optional<SymbolicPointerValue>({false, *scalar}) : std::nullopt;
        }
        if (expr->op != cot_cast || !address_width(expr->type) || !address_width(expr->x->type)) {
            return std::nullopt;
        }
        return lhs;
    }
    const auto rhs = evaluate_symbolic_pointer(expr->y, target_var, index_var, index_value, depth + 1);
    if (!rhs.has_value()) return std::nullopt;
    if (!lhs->has_base && !rhs->has_base) {
        const auto scalar = evaluate_integer_binary(expr, lhs->offset, rhs->offset);
        return scalar.has_value()
            ? std::optional<SymbolicPointerValue>({false, *scalar}) : std::nullopt;
    }
    if ((expr->op != cot_add && expr->op != cot_sub) || !address_width(expr->type)) {
        return std::nullopt;
    }
    // A structure-relative address contains exactly one positive base term.
    if (lhs->has_base == rhs->has_base || (expr->op == cot_sub && rhs->has_base)) {
        return std::nullopt;
    }
    const auto& base = lhs->has_base ? *lhs : *rhs;
    const auto& scalar = lhs->has_base ? *rhs : *lhs;
    const auto* base_expr = lhs->has_base ? expr->x : expr->y;
    sval_t scale = 1;
    if (base_expr->type.is_ptr()) {
        const auto size = base_expr->type.get_pointed_object().get_size();
        if (size == 0 || size == BADSIZE || size > static_cast<std::size_t>(
                std::numeric_limits<sval_t>::max())) return std::nullopt;
        scale = static_cast<sval_t>(size);
    }
    sval_t delta = 0;
    if (!checked_mul_sval(scalar.offset, scale, delta)) return std::nullopt;
    if (expr->op == cot_sub && !checked_mul_sval(delta, -1, delta)) return std::nullopt;
    sval_t offset = 0;
    if (!checked_add_sval(base.offset, delta, offset)) return std::nullopt;
    return SymbolicPointerValue{true, offset};
}

std::optional<std::uint32_t> regular_offset_stride(const qvector<sval_t>& offsets) {
    if (offsets.size() < 2) return std::nullopt;
    const auto first_delta = checked_sval_sub(offsets[1], offsets[0]);
    if (!first_delta.has_value() || *first_delta == 0) return std::nullopt;
    for (std::size_t i = 2; i < offsets.size(); ++i) {
        if (checked_sval_sub(offsets[i], offsets[i - 1]) != first_delta) return std::nullopt;
    }
    const auto magnitude = *first_delta > 0 ? first_delta : checked_sval_mul(*first_delta, -1);
    if (!magnitude.has_value() || *magnitude > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(*magnitude);
}

} // namespace

// ============================================================================
// AccessPatternVisitor Implementation
// ============================================================================

AccessPatternVisitor::AccessPatternVisitor(cfunc_t* cfunc, int target_var_idx)
    : ctree_visitor_t(CV_PARENTS)
    , cfunc_(cfunc)
    , target_var_idx_(target_var_idx)
    , has_unstructured_control_flow_(cfunc && cfunc->body.contains_insn(cit_goto)) {}

int AccessPatternVisitor::visit_insn(cinsn_t* insn) {
    if (!insn) return 0;
    // The SDK's default ctree walk visits branch/loop bodies before their
    // conditions, and a for-loop step before its body. Bounds and local value
    // versions require execution order, so enumerate these children explicitly
    // while retaining the SDK-maintained parent stack.
    auto visit_children = [&](std::initializer_list<citem_t*> children) {
        for (citem_t* child : children) {
            if (child) {
                const int code = apply_to(child, insn);
                if (code != 0) return code;
            }
        }
        prune_now();
        return 0;
    };
    switch (insn->op) {
        case cit_if:
            return visit_children({&insn->cif->expr, insn->cif->ithen, insn->cif->ielse});
        case cit_while:
            return visit_children({&insn->cwhile->expr, insn->cwhile->body});
        case cit_for:
            return visit_children({&insn->cfor->init, &insn->cfor->expr,
                                   insn->cfor->body, &insn->cfor->step});
        default:
            return 0;
    }
}

int AccessPatternVisitor::visit_expr(cexpr_t* expr) {
    if (!expr) return 0;

    switch (expr->op) {
        case cot_ptr:
            // Dereference: *(ptr + offset) or *ptr
            if (involves_target_var(expr->x)) {
                process_dereference(expr, expr->x);
            }
            break;

        case cot_memptr:
            // Member pointer access: ptr->member
            if (involves_target_var(expr->x)) {
                process_memptr_access(expr);
            }
            break;

        case cot_idx:
            // Array indexing: ptr[idx]
            if (involves_target_var(expr->x)) {
                process_array_access(expr);
            }
            break;

        case cot_asg:
        case cot_asgbor:
        case cot_asgxor:
        case cot_asgband:
        case cot_asgadd:
        case cot_asgsub:
        case cot_asgmul:
        case cot_asgsshr:
        case cot_asgushr:
        case cot_asgshl:
        case cot_asgsdiv:
        case cot_asgudiv:
        case cot_asgsmod:
        case cot_asgumod:
            process_assignment(expr);
            break;

        case cot_band: {
            const cexpr_t* mask_expr = nullptr;
            const cexpr_t* value_expr = nullptr;
            if (expr->x && expr->x->op == cot_num) {
                mask_expr = expr->x;
                value_expr = expr->y;
            } else if (expr->y && expr->y->op == cot_num) {
                mask_expr = expr->y;
                value_expr = expr->x;
            }

            if (mask_expr && value_expr) {
                const cexpr_t* base_expr = value_expr;
                int shift = 0;

                if (base_expr->op == cot_sshr || base_expr->op == cot_ushr) {
                    if (base_expr->y && base_expr->y->op == cot_num) {
                        shift = static_cast<int>(base_expr->y->numval());
                        base_expr = base_expr->x;
                    }
                }

                while (base_expr && base_expr->op == cot_cast) {
                    base_expr = base_expr->x;
                }

                sval_t offset = 0;
                uint32_t size = 0;
                std::optional<std::uint8_t> base_indirection;
                BitfieldInfo info;
                bool resolved = extract_access(base_expr, offset, size, &base_indirection);
                if (!resolved && base_expr && base_expr->op == cot_var) {
                    auto it = local_aliases_.find(base_expr->v.idx);
                    if (it != local_aliases_.end() &&
                        !address_aliases_.contains(base_expr->v.idx)) {
                        offset = it->second.offset;
                        size = it->second.size;
                        base_indirection = it->second.base_indirection;
                        resolved = true;
                    }
                }

                if (resolved &&
                    compute_bitfield(static_cast<std::uint64_t>(mask_expr->numval()),
                                     shift, info.bit_offset, info.bit_size)) {
                    if (static_cast<unsigned>(info.bit_offset + info.bit_size) <= size * 8) {
                        record_bitfield_access(expr, offset, size, info, base_indirection);
                    }
                }
            }
            break;
        }

        case cot_eq:
        case cot_ne:
            process_constant_comparison(expr);
            process_index_bound(expr);
            break;

        case cot_ult:
        case cot_ule:
        case cot_slt:
        case cot_sle:
        case cot_ugt:
        case cot_uge:
        case cot_sgt:
        case cot_sge:
            process_index_bound(expr);
            break;

        case cot_preinc:
        case cot_predec:
        case cot_postinc:
        case cot_postdec:
            if (expr->x && expr->x->op == cot_var) {
                invalidate_local_var_state(expr->x->v.idx, true);
            }
            break;

        case cot_ref:
            if (expr->x && expr->x->op == cot_var) {
                escaped_local_vars_.insert(expr->x->v.idx);
                invalidate_local_var_state(expr->x->v.idx, true);
            }
            break;

        case cot_call:
            for (int var_idx : escaped_local_vars_) {
                invalidate_local_var_state(var_idx, true);
            }
            process_call_argument_uses(expr);
            // Check for indirect calls through our variable
            process_call_through_ptr(expr);
            break;

        default:
            break;
    }

    return 0;
}

void AccessPatternVisitor::process_assignment(cexpr_t* expr) {
    if (!expr || !is_assignment_op(expr->op) || !expr->x || !expr->y) {
        return;
    }

    cexpr_t* lhs = expr->x;
    if (!lhs || lhs->op != cot_var) {
        return;
    }

    if (expr->op != cot_asg) {
        invalidate_local_var_state(lhs->v.idx, true);
        pending_constants_.erase(lhs->v.idx);
        return;
    }

    cexpr_t* rhs = expr->y;
    if (inspect_base_conversions(rhs, target_var_idx_, address_aliases_).lossy) {
        invalidate_local_var_state(lhs->v.idx, true);
        return;
    }
    while (rhs && rhs->op == cot_cast) {
        rhs = rhs->x;
    }

    if (!rhs) {
        invalidate_local_var_state(lhs->v.idx, true);
        pending_constants_.erase(lhs->v.idx);
        return;
    }

    FieldAccess alias;
    bool resolved = false;
    bool address_only = false;

    sval_t offset = 0;
    uint32_t size = 0;
    std::optional<std::uint8_t> base_indirection;
    if (extract_access(rhs, offset, size, &base_indirection)) {
        alias.insn_ea = expr->ea;
        alias.source_func_ea = cfunc_->entry_ea;
        alias.offset = offset;
        alias.size = size;
        alias.access_type = AccessType::Read;
        alias.semantic_type = infer_semantic_from_usage(rhs, parent_expr());
        alias.context_expr = utils::expr_to_string(rhs, cfunc_);
        alias.inferred_type = rhs->type;
        alias.base_indirection = base_indirection;
        resolved = true;
    } else if (rhs->op == cot_var) {
        auto it = local_aliases_.find(rhs->v.idx);
        if (it != local_aliases_.end()) {
            alias = it->second;
            address_only = address_aliases_.contains(rhs->v.idx);
            resolved = true;
        } else if (rhs->v.idx == target_var_idx_) {
            alias.insn_ea = expr->ea;
            alias.source_func_ea = cfunc_->entry_ea;
            alias.offset = 0;
            alias.size = get_ptr_size();
            alias.access_type = AccessType::Read;
            alias.semantic_type = infer_semantic_from_usage(rhs, parent_expr());
            alias.context_expr = utils::expr_to_string(rhs, cfunc_);
            alias.inferred_type = rhs->type;
            address_only = true;
            resolved = true;
        }
    }

    invalidate_local_var_state(lhs->v.idx, false);

    if (resolved) {
        auto pending_it = pending_constants_.find(lhs->v.idx);
        if (pending_it != pending_constants_.end()) {
            if (!address_only) {
                msg("Structor:   Applying %zu pending constants to local v%d\n",
                    pending_it->second.size(), lhs->v.idx);
                for (auto value : pending_it->second) {
                    alias.add_observed_constant(value);
                }
            }
            pending_constants_.erase(pending_it);
        }
        local_aliases_[lhs->v.idx] = std::move(alias);
        if (address_only) {
            address_aliases_.insert(lhs->v.idx);
        }
        return;
    }

    invalidate_local_var_state(lhs->v.idx, true);
    pending_constants_.erase(lhs->v.idx);
}

void AccessPatternVisitor::invalidate_local_var_state(int var_idx,
                                                      bool clear_pending_constants) {
    local_aliases_.erase(var_idx);
    address_aliases_.erase(var_idx);
    ++local_var_versions_[var_idx];
    if (clear_pending_constants) {
        pending_constants_.erase(var_idx);
    }
}

utils::PtrArithInfo AccessPatternVisitor::resolve_ptr_arith(const cexpr_t* expr) const {
    if (inspect_base_conversions(expr, target_var_idx_, address_aliases_).lossy) {
        return {};
    }
    utils::PtrArithInfo info = utils::extract_ptr_arith(expr);
    if (!info.valid) {
        return info;
    }

    if (info.var_idx == target_var_idx_) {
        return info;
    }

    auto it = local_aliases_.find(info.var_idx);
    if (it == local_aliases_.end()) {
        return info;
    }

    const FieldAccess& alias = it->second;
    const bool is_loaded_value = !address_aliases_.contains(info.var_idx);
    info.var_idx = target_var_idx_;
    const auto rebased = checked_sval_add(info.offset, alias.offset);
    if (!rebased.has_value()) {
        info.valid = false;
        return info;
    }
    info.offset = *rebased;
    if (alias.base_indirection.has_value()) {
        info.base_indirection = static_cast<std::uint8_t>(
            std::min<int>(0xFF, info.base_indirection + *alias.base_indirection));
        info.through_pointer_alias = true;
    }
    info.through_pointer_alias = info.through_pointer_alias || is_loaded_value;
    return info;
}

void AccessPatternVisitor::process_constant_comparison(cexpr_t* expr) {
    if (!expr || !expr->x || !expr->y) {
        return;
    }

    const cexpr_t* value_expr = nullptr;
    std::uint64_t constant = 0;

    if (expr->x->op == cot_num) {
        constant = static_cast<std::uint64_t>(expr->x->numval());
        value_expr = expr->y;
    } else if (expr->y->op == cot_num) {
        constant = static_cast<std::uint64_t>(expr->y->numval());
        value_expr = expr->x;
    } else {
        return;
    }

    while (value_expr && value_expr->op == cot_cast) {
        value_expr = value_expr->x;
    }
    if (!value_expr) {
        return;
    }

    sval_t offset = 0;
    uint32_t size = 0;
    std::optional<std::uint8_t> base_indirection;
    bool resolved = extract_access(value_expr, offset, size, &base_indirection);

    FieldAccess access;
    if (!resolved && value_expr->op == cot_var) {
        auto it = local_aliases_.find(value_expr->v.idx);
        if (it == local_aliases_.end()) {
            msg("Structor:   Queueing constant 0x%llX for unresolved local v%d\n",
                static_cast<unsigned long long>(constant), value_expr->v.idx);
            pending_constants_[value_expr->v.idx].push_back(constant);
            return;
        }
        if (address_aliases_.contains(value_expr->v.idx)) {
            return;
        }
        access = it->second;
        resolved = true;
    }

    if (!resolved) {
        return;
    }

    if (access.size == 0) {
        access.insn_ea = expr->ea;
        access.source_func_ea = cfunc_->entry_ea;
        access.offset = offset;
        access.size = size;
        access.access_type = AccessType::Read;
        access.semantic_type = infer_semantic_from_usage(value_expr, parent_expr());
        access.context_expr = utils::expr_to_string(value_expr, cfunc_);
        access.inferred_type = value_expr->type;
        access.base_indirection = base_indirection;
    }

    msg("Structor:   Observed comparison constant 0x%llX at offset 0x%llX size=%u\n",
        static_cast<unsigned long long>(constant),
        static_cast<unsigned long long>(access.offset), access.size);
    access.add_observed_constant(constant);
    accesses_.push_back(std::move(access));
}

void AccessPatternVisitor::process_index_bound(cexpr_t* expr) {
    if (!expr || !expr->x || !expr->y) {
        return;
    }

    const cexpr_t* variable = expr->x;
    const cexpr_t* number = expr->y;
    bool reversed = false;
    if (variable->op == cot_num) {
        std::swap(variable, number);
        reversed = true;
    }
    if (number->op != cot_num) {
        return;
    }
    if ((expr->op == cot_eq || expr->op == cot_ne) && number->numval() == 0 &&
        variable->op == cot_band && variable->x && variable->y) {
        const cexpr_t* value = variable->x;
        const cexpr_t* mask = variable->y;
        if (value->op == cot_num) std::swap(value, mask);
        if (mask->op != cot_num) return;
        while (value && value->op == cot_cast) {
            if (!value->x || !value->type.is_integral() || !value->x->type.is_integral() ||
                value->type.get_size() != value->x->type.get_size()) return;
            value = value->x;
        }
        if (!value || value->op != cot_var || !value->type.is_integral()) return;
        const auto size = value->type.get_size();
        if (size == 0 || size == BADSIZE || size > sizeof(sval_t) ||
            variable->type.get_size() != size) return;
        const auto sign_bit = UINT64_C(1) << (size * 8 - 1);
        const auto width_mask = size == 8 ? UINT64_MAX : (UINT64_C(1) << (size * 8)) - 1;
        if ((mask->numval() & width_mask) != sign_bit) return;
        const auto signed_max = static_cast<sval_t>(sign_bit - 1);
        IndexRange clear{0, signed_max, true, true};
        IndexRange set;
        if (value->type.is_signed()) {
            const auto signed_min = size == 8 ? std::numeric_limits<sval_t>::min()
                                              : -static_cast<sval_t>(sign_bit);
            set = {signed_min, -1, true, true};
        } else if (value->type.is_unsigned() && size < 8) {
            set = {static_cast<sval_t>(sign_bit), static_cast<sval_t>(width_mask), true, true};
        }
        const bool zero_when_true = expr->op == cot_eq;
        const int var_idx = value->v.idx;
        index_comparisons_[expr] = {var_idx, local_var_versions_[var_idx],
            zero_when_true ? clear : set, zero_when_true ? set : clear};
        return;
    }
    // A narrowing cast constrains only the truncated value, not the index.
    // Require every cast to preserve the variable's storage width.
    const tinfo_t compared_type = variable->type;
    while (variable && variable->op == cot_cast) {
        if (!variable->x || !variable->type.is_integral() || !variable->x->type.is_integral() ||
            variable->type.is_bool() != variable->x->type.is_bool() ||
            variable->type.get_size() == BADSIZE || variable->type.get_size() == 0 ||
            variable->type.get_size() != variable->x->type.get_size()) {
            return;
        }
        variable = variable->x;
    }
    if (!variable || variable->op != cot_var || !variable->type.is_integral()) {
        return;
    }

    const bool unsigned_comparison = expr->op == cot_ult || expr->op == cot_ule ||
        expr->op == cot_ugt || expr->op == cot_uge;
    const bool signed_comparison = expr->op == cot_slt || expr->op == cot_sle ||
        expr->op == cot_sgt || expr->op == cot_sge;
    // A signed comparison of unsigned storage may admit the high half of its
    // bit patterns as negative numbers. A single numeric interval cannot map
    // that discontinuity back to the unsigned index without losing values.
    if (signed_comparison && variable->type.is_unsigned()) return;
    const auto variable_size = compared_type.get_size();
    const auto number_size = number->type.get_size();
    if (variable_size == 0 || variable_size == BADSIZE || variable_size > sizeof(sval_t) ||
        number_size == 0 || number_size == BADSIZE || number_size > sizeof(sval_t)) return;
    const auto operand_size = std::max(variable_size, number_size);
    const auto width_mask = operand_size == 8 ? UINT64_MAX : (UINT64_C(1) << (operand_size * 8)) - 1;
    auto raw_constant = static_cast<std::uint64_t>(number->numval()) & width_mask;
    if (signed_comparison && (raw_constant & (UINT64_C(1) << (operand_size * 8 - 1))) != 0) {
        raw_constant |= ~width_mask;
    }
    // Preserve wider constants (e.g. 0x100000004); never reduce their width
    // merely because the variable is narrower.
    if (unsigned_comparison && raw_constant > static_cast<std::uint64_t>(std::numeric_limits<sval_t>::max())) return;
    sval_t constant = static_cast<sval_t>(raw_constant);
    if (expr->op == cot_eq || expr->op == cot_ne) {
        const auto candidate = normalize_integer_value(raw_constant, variable->type);
        const auto compared = candidate.has_value()
            ? normalize_integer_value(static_cast<std::uint64_t>(*candidate), compared_type)
            : std::nullopt;
        if (!compared.has_value() ||
            (static_cast<std::uint64_t>(*compared) & width_mask) != (raw_constant & width_mask)) return;
        constant = *candidate;
    }
    enum class Relation { Less, LessEqual, Greater, GreaterEqual, Equal, NotEqual };
    Relation relation;
    switch (expr->op) {
        case cot_ult: case cot_slt: relation = Relation::Less; break;
        case cot_ule: case cot_sle: relation = Relation::LessEqual; break;
        case cot_ugt: case cot_sgt: relation = Relation::Greater; break;
        case cot_uge: case cot_sge: relation = Relation::GreaterEqual; break;
        case cot_eq: relation = Relation::Equal; break;
        case cot_ne: relation = Relation::NotEqual; break;
        default: return;
    }
    if (reversed) {
        switch (relation) {
            case Relation::Less: relation = Relation::Greater; break;
            case Relation::LessEqual: relation = Relation::GreaterEqual; break;
            case Relation::Greater: relation = Relation::Less; break;
            case Relation::GreaterEqual: relation = Relation::LessEqual; break;
            default: break;
        }
    }

    auto range_for = [&](bool truth) {
        IndexRange range;
        if (unsigned_comparison) {
            range.first = 0;
            range.first_known = true;
        }
        Relation effective = relation;
        if (!truth) {
            switch (effective) {
                case Relation::Less: effective = Relation::GreaterEqual; break;
                case Relation::LessEqual: effective = Relation::Greater; break;
                case Relation::Greater: effective = Relation::LessEqual; break;
                case Relation::GreaterEqual: effective = Relation::Less; break;
                case Relation::Equal: effective = Relation::NotEqual; break;
                case Relation::NotEqual: effective = Relation::Equal; break;
            }
        }
        switch (effective) {
            case Relation::Less:
                if (constant == std::numeric_limits<sval_t>::min()) {
                    range = {1, 0, true, true};
                } else {
                    range.last = constant - 1;
                    range.last_known = true;
                }
                break;
            case Relation::LessEqual:
                range.last = constant;
                range.last_known = true;
                break;
            case Relation::Greater:
                if (constant == std::numeric_limits<sval_t>::max()) {
                    // For unsigned comparisons this may still admit large
                    // values. Keep an unbounded range instead of inventing one.
                    if (!unsigned_comparison) {
                        range = {1, 0, true, true};
                    }
                } else {
                    range.first = constant + 1;
                    range.first_known = true;
                }
                break;
            case Relation::GreaterEqual:
                range.first = constant;
                range.first_known = true;
                break;
            case Relation::Equal: range = {constant, constant, true, true}; break;
            case Relation::NotEqual: break; // an interval cannot represent the hole
        }
        if (unsigned_comparison && variable->type.is_signed()) {
            const auto size = variable->type.get_size();
            if (size == 0 || size == BADSIZE || size > sizeof(sval_t)) return IndexRange{};
            const auto signed_max = size == sizeof(sval_t)
                ? std::numeric_limits<sval_t>::max()
                : static_cast<sval_t>((UINT64_C(1) << (size * 8 - 1)) - 1);
            // An unsigned upper bound below the sign bit proves the original
            // signed index is nonnegative. An unsigned lower bound alone does
            // not: it can also admit every negative signed bit pattern.
            if (!range.last_known || range.last < 0 || range.last > signed_max) {
                return IndexRange{};
            }
        }
        return range;
    };

    const int var_idx = variable->v.idx;
    index_comparisons_[expr] = {
        var_idx, local_var_versions_[var_idx], range_for(true), range_for(false)};
}

AccessPatternVisitor::IndexRange AccessPatternVisitor::condition_index_range(
        const cexpr_t* condition, int var_idx, bool truth, int depth) const {
    if (!condition || depth > 64) {
        return {};
    }
    if (condition->op == cot_lnot) {
        return condition_index_range(condition->x, var_idx, !truth, depth + 1);
    }
    if (condition->op == cot_land || condition->op == cot_lor) {
        const auto lhs = condition_index_range(condition->x, var_idx, truth, depth + 1);
        const auto rhs = condition_index_range(condition->y, var_idx, truth, depth + 1);
        const bool intersection = (condition->op == cot_land) == truth;
        return intersection ? lhs.intersect(rhs) : lhs.unite(rhs);
    }
    const auto it = index_comparisons_.find(condition);
    if (it == index_comparisons_.end() || it->second.var_idx != var_idx) {
        return {};
    }
    const auto version = local_var_versions_.find(var_idx);
    if (version == local_var_versions_.end() || version->second != it->second.version) {
        return {};
    }
    return truth ? it->second.when_true : it->second.when_false;
}

bool AccessPatternVisitor::loop_may_change_index(const cinsn_t* loop, int var_idx) const {
    auto it = loop_index_effects_.find(loop);
    if (it == loop_index_effects_.end()) {
        struct EffectVisitor final : ctree_visitor_t {
            LoopIndexEffects effects;
            EffectVisitor() : ctree_visitor_t(CV_FAST) {}
            int idaapi visit_expr(cexpr_t* expr) override {
                if (expr->op == cot_call) effects.has_call = true;
                const bool may_write = is_assignment_op(expr->op) ||
                    expr->op == cot_preinc || expr->op == cot_predec ||
                    expr->op == cot_postinc || expr->op == cot_postdec ||
                    expr->op == cot_ref;
                if (may_write && expr->x && expr->x->op == cot_var) {
                    effects.written_or_referenced_vars.insert(expr->x->v.idx);
                }
                return 0;
            }
        } visitor;
        visitor.apply_to(const_cast<cinsn_t*>(loop), nullptr);
        it = loop_index_effects_.emplace(loop, std::move(visitor.effects)).first;
    }
    return it->second.written_or_referenced_vars.contains(var_idx) ||
        (it->second.has_call && escaped_local_vars_.contains(var_idx));
}

std::optional<AccessPatternVisitor::IndexRange> AccessPatternVisitor::bounded_index_range(
        const cexpr_t* access_expr, int var_idx) const {
    // Lexical ancestry alone cannot prove dominance when a goto may enter a
    // guarded region. Such functions need a CFG-based proof before expansion.
    if (has_unstructured_control_flow_) {
        return std::nullopt;
    }
    IndexRange result;
    auto constrain = [&](const cexpr_t* condition, bool truth) {
        const auto range = condition_index_range(condition, var_idx, truth);
        result = result.intersect(range);
    };

    const citem_t* child = access_expr;
    for (std::size_t i = parents.size(); i > 0; --i) {
        const citem_t* parent = parents[i - 1];
        // apply_to(..., nullptr) retains a null sentinel for the tree root.
        if (!parent) continue;
        if (parent->is_expr()) {
            const auto* expression = static_cast<const cexpr_t*>(parent);
            if (expression->op == cot_land && child == expression->y) {
                constrain(expression->x, true);
            } else if (expression->op == cot_lor && child == expression->y) {
                constrain(expression->x, false);
            } else if (expression->op == cot_tern) {
                if (child == expression->y) constrain(expression->x, true);
                if (child == expression->z) constrain(expression->x, false);
            }
        } else {
            const auto* instruction = static_cast<const cinsn_t*>(parent);
            if (instruction->op == cit_if && instruction->cif) {
                if (child == instruction->cif->ithen) constrain(&instruction->cif->expr, true);
                if (child == instruction->cif->ielse) constrain(&instruction->cif->expr, false);
            } else if (instruction->op == cit_while && instruction->cwhile &&
                       child == instruction->cwhile->body) {
                constrain(&instruction->cwhile->expr, true);
            } else if (instruction->op == cit_for && instruction->cfor &&
                       child == instruction->cfor->body) {
                constrain(&instruction->cfor->expr, true);
            }
            // A do-loop condition does not guard the first body execution.
            // A guard outside a loop must also hold after every backedge. A
            // single lexical traversal cannot establish that if the loop can
            // modify the index after this access, so retain only inner guards.
            if ((instruction->op == cit_for || instruction->op == cit_while ||
                 instruction->op == cit_do) && loop_may_change_index(instruction, var_idx)) {
                break;
            }
        }
        child = parent;
    }
    const auto span = checked_sval_sub(result.last, result.first);
    if (!result.first_known || !result.last_known || !span.has_value() ||
        *span < 0 || *span >= 32) {
        return std::nullopt;
    }
    return result;
}

void AccessPatternVisitor::process_dereference(cexpr_t* expr, const cexpr_t* ptr_expr) {
    auto arith = resolve_ptr_arith(expr);

    if (!arith.valid && ptr_expr) {
        int index_var = -1;
        if (!find_symbolic_index(ptr_expr, target_var_idx_, index_var) || index_var < 0) {
            return;
        }
        const auto range = bounded_index_range(expr, index_var);
        if (!range.has_value()) return;
        qvector<sval_t> offsets;
        const auto count = static_cast<std::uint32_t>(range->last - range->first) + 1;
        for (std::uint32_t item = 0; item < count; ++item) {
            const sval_t idx = range->first + item;
            const auto value = evaluate_symbolic_pointer(ptr_expr, target_var_idx_, index_var, idx);
            if (!value.has_value() || !value->has_base) return;
            offsets.push_back(value->offset);
        }
        const auto stride = regular_offset_stride(offsets);
        for (sval_t offset : offsets) {
            FieldAccess access;
            access.insn_ea = expr->ea;
            access.offset = offset;
            access.size = !expr->type.empty()
                ? utils::get_type_size(expr->type, get_ptr_size()) : get_ptr_size();
            const cexpr_t* rhs = nullptr;
            access.access_type = determine_access_type(expr, &rhs);
            if (access.access_type == AccessType::Write) {
                extract_and_add_rhs_constant(access, rhs);
                access.is_zero_init = is_zero_initialization(expr);
            }
            access.semantic_type = infer_semantic_from_usage(expr, parent_expr());
            access.context_expr = utils::expr_to_string(expr, cfunc_);
            access.inferred_type = expr->type;
            access.source_func_ea = cfunc_->entry_ea;
            access.array_stride_hint = stride;
            access.is_call_argument = is_call_argument_use(expr);
            accesses_.push_back(std::move(access));
        }
        return;
    }

    if (!arith.valid || arith.var_idx != target_var_idx_) {
        return;
    }

    FieldAccess access;
    access.insn_ea = expr->ea;
    access.offset = arith.offset;

    // Determine size from expression type
    if (!expr->type.empty()) {
        access.size = utils::get_type_size(expr->type, get_ptr_size());
    } else {
        access.size = get_ptr_size();
    }

    const cexpr_t* rhs = nullptr;
    access.access_type = determine_access_type(expr, &rhs);
    if (access.access_type == AccessType::Write) {
        extract_and_add_rhs_constant(access, rhs);
    }

    // Check if this is a zero-initialization write
    if (access.access_type == AccessType::Write) {
        access.is_zero_init = is_zero_initialization(expr);
    }

    // Infer semantic type from context
    const cexpr_t* parent = parent_expr();
    access.semantic_type = infer_semantic_from_usage(expr, parent);
    access.is_call_argument = is_call_argument_use(expr);

    if (arith.base_indirection > 0) {
        access.base_indirection = arith.base_indirection;
    }

    // Check for vtable access pattern: *(*var + offset)
    // This is a double dereference where the outer load resolves to a
    // function pointer slot through a vtable pointer field on the object.
    const cexpr_t* normalized_ptr = ptr_expr;
    while (normalized_ptr && normalized_ptr->op == cot_cast) {
        normalized_ptr = normalized_ptr->x;
    }

    const cexpr_t* slot_base = normalized_ptr;
    if (slot_base && slot_base->op == cot_add) {
        if (slot_base->x && slot_base->x->op == cot_num) {
            slot_base = slot_base->y;
        } else if (slot_base->y && slot_base->y->op == cot_num) {
            slot_base = slot_base->x;
        }
    }
    while (slot_base && slot_base->op == cot_cast) {
        slot_base = slot_base->x;
    }

    if (slot_base && slot_base->op == cot_ptr) {
        auto inner_arith = resolve_ptr_arith(slot_base->x);
        const bool function_slot_like =
            expr->type.is_funcptr() ||
            access.semantic_type == SemanticType::FunctionPointer;
        if (inner_arith.valid && inner_arith.var_idx == target_var_idx_ &&
            arith.offset >= inner_arith.offset && function_slot_like) {
            const sval_t slot_offset = arith.offset - inner_arith.offset;
            if (!is_valid_vtable_slot_offset(slot_offset)) {
                return;
            }
            // Normalize nested vtable slot dereferences back to the parent
            // vtable pointer field so we don't mistake the object for the
            // vtable layout itself.
            access.offset = inner_arith.offset;
            access.semantic_type = SemanticType::VTablePointer;
            access.is_vtable_access = true;
            access.vtable_slot = slot_offset / get_ptr_size();
            (void)access.set_vtable_nested_access(
                inner_arith.offset, slot_offset, expr->type);
            access.base_indirection.reset();
        }
    }

    if (!access.is_vtable_access && is_pointee_access(arith)) {
        return;
    }

    access.context_expr = utils::expr_to_string(expr, cfunc_);
    access.inferred_type = expr->type;
    access.source_func_ea = cfunc_->entry_ea;

    accesses_.push_back(std::move(access));
}

void AccessPatternVisitor::process_memptr_access(cexpr_t* expr) {
    const cexpr_t* parent = parent_expr();
    if ((expr->type.is_array() || expr->type.is_struct() || is_aggregate_member_container(expr)) &&
        parent != nullptr) {
        switch (parent->op) {
            case cot_idx:
            case cot_call:
            case cot_ref:
            case cot_memref:
            case cot_memptr:
                return;
            default:
                break;
        }
    }

    auto arith = resolve_ptr_arith(expr->x);
    if (!arith.valid || arith.var_idx != target_var_idx_) {
        return;
    }
    if (is_pointee_access(arith)) {
        return;
    }

    FieldAccess access;
    access.insn_ea = expr->ea;
    if (!checked_add_sval(
            arith.offset, static_cast<sval_t>(expr->m), access.offset)) {
        return;
    }

    if (!expr->type.empty()) {
        access.size = utils::get_type_size(expr->type, get_ptr_size());
    } else {
        access.size = get_ptr_size();
    }

    const cexpr_t* rhs = nullptr;
    access.access_type = determine_access_type(expr, &rhs);
    if (access.access_type == AccessType::Write) {
        extract_and_add_rhs_constant(access, rhs);
    }
    access.semantic_type = infer_semantic_from_usage(expr, parent);
    access.is_call_argument = is_call_argument_use(expr);
    if (arith.base_indirection > 0) {
        access.base_indirection = arith.base_indirection;
    }
    access.context_expr = utils::expr_to_string(expr, cfunc_);
    access.inferred_type = expr->type;
    access.source_func_ea = cfunc_->entry_ea;

    accesses_.push_back(std::move(access));
}

void AccessPatternVisitor::process_array_access(cexpr_t* expr) {
    auto arith = resolve_ptr_arith(expr->x);
    if (!arith.valid || arith.var_idx != target_var_idx_) {
        return;
    }

    // Calculate offset
    sval_t offset = arith.offset;
    std::optional<std::uint32_t> stride_hint;

    tinfo_t elem_type = expr->type;
    if (!elem_type.empty()) {
        size_t elem_size = elem_type.get_size();
        if (elem_size != BADSIZE && elem_size > 0 &&
            elem_size <= std::numeric_limits<std::uint32_t>::max()) {
            stride_hint = static_cast<std::uint32_t>(elem_size);
        }
    }

    if (!stride_hint.has_value()) {
        tinfo_t ptr_elem = expr->x->type.get_pointed_object();
        if (!ptr_elem.empty()) {
            size_t elem_size = ptr_elem.get_size();
            if (elem_size != BADSIZE && elem_size > 0 &&
                elem_size <= std::numeric_limits<std::uint32_t>::max()) {
                stride_hint = static_cast<std::uint32_t>(elem_size);
            }
        }
    }

    const bool function_slot_like = expr->type.is_funcptr();
    const sval_t base_offset = arith.offset;

    if (expr->y->op == cot_num) {
        const sval_t index_value = static_cast<sval_t>(expr->y->numval());
        sval_t index_delta = index_value;
        if (stride_hint.has_value()) {
            if (!checked_mul_sval(
                    index_value, static_cast<sval_t>(*stride_hint),
                    index_delta)) {
                return;
            }
        }
        if (!checked_add_sval(offset, index_delta, offset)) {
            return;
        }
    }

    if (expr->y->op != cot_num && stride_hint.has_value()) {
        int index_var = -1;
        if (!find_symbolic_index(expr->y, target_var_idx_, index_var) || index_var < 0) return;
        const auto range = bounded_index_range(expr, index_var);
        if (!range.has_value() || (is_pointee_access(arith) && !function_slot_like)) return;
        qvector<sval_t> offsets;
        const auto count = static_cast<std::uint32_t>(range->last - range->first) + 1;
        for (std::uint32_t item = 0; item < count; ++item) {
            const sval_t idx = range->first + item;
            const auto index = evaluate_index_expression(expr->y, index_var, idx);
            sval_t delta = 0;
            sval_t bounded_offset = 0;
            if (!index.has_value() ||
                !checked_mul_sval(*index, static_cast<sval_t>(*stride_hint), delta) ||
                !checked_add_sval(offset, delta, bounded_offset)) return;
            offsets.push_back(bounded_offset);
        }
        const auto observed_stride = regular_offset_stride(offsets);
        for (sval_t bounded_offset : offsets) {
            FieldAccess bounded;
            bounded.insn_ea = expr->ea;
            bounded.offset = bounded_offset;
            bounded.size = !expr->type.empty()
                ? utils::get_type_size(expr->type, get_ptr_size()) : get_ptr_size();
            const cexpr_t* rhs = nullptr;
            bounded.access_type = determine_access_type(expr, &rhs);
            if (bounded.access_type == AccessType::Write) {
                extract_and_add_rhs_constant(bounded, rhs);
                bounded.is_zero_init = is_zero_initialization(expr);
            }
            bounded.semantic_type = infer_semantic_from_usage(expr, parent_expr());
            bounded.is_call_argument = is_call_argument_use(expr);
            if (function_slot_like && arith.base_indirection > 0 && *stride_hint == get_ptr_size()) {
                const auto slot_offset = checked_sval_sub(bounded_offset, base_offset);
                if (!slot_offset.has_value() || !is_valid_vtable_slot_offset(*slot_offset)) continue;
                bounded.offset = base_offset;
                bounded.semantic_type = SemanticType::VTablePointer;
                bounded.is_vtable_access = true;
                bounded.vtable_slot = *slot_offset / get_ptr_size();
                (void)bounded.set_vtable_nested_access(base_offset, *slot_offset, expr->type);
            }
            if (arith.base_indirection > 0 && !bounded.is_vtable_access) {
                bounded.base_indirection = arith.base_indirection;
            }
            bounded.context_expr = utils::expr_to_string(expr, cfunc_);
            bounded.inferred_type = expr->type;
            bounded.source_func_ea = cfunc_->entry_ea;
            bounded.array_stride_hint = observed_stride;
            accesses_.push_back(std::move(bounded));
        }
        return;
    }

    if (expr->y->op != cot_num) {
        return;
    }

    FieldAccess access;
    access.insn_ea = expr->ea;
    access.offset = offset;

    if (!expr->type.empty()) {
        access.size = utils::get_type_size(expr->type, get_ptr_size());
    } else {
        access.size = get_ptr_size();
    }

    const cexpr_t* rhs = nullptr;
    access.access_type = determine_access_type(expr, &rhs);
    if (access.access_type == AccessType::Write) {
        extract_and_add_rhs_constant(access, rhs);
    }
    const cexpr_t* parent = parent_expr();
    access.semantic_type = infer_semantic_from_usage(expr, parent);
    access.is_call_argument = is_call_argument_use(expr);
    if (function_slot_like && arith.base_indirection > 0 &&
        stride_hint.has_value() && *stride_hint == get_ptr_size() &&
        expr->y->op == cot_num) {
        const sval_t slot_offset = offset - base_offset;
        access.offset = base_offset;
        access.semantic_type = SemanticType::VTablePointer;
        access.is_vtable_access = true;
        access.vtable_slot = slot_offset / static_cast<sval_t>(*stride_hint);
        (void)access.set_vtable_nested_access(base_offset, slot_offset, expr->type);
    }
    if (!access.is_vtable_access && is_pointee_access(arith)) {
        return;
    }
    if (arith.base_indirection > 0) {
        if (!access.is_vtable_access) {
            access.base_indirection = arith.base_indirection;
        }
    }
    access.context_expr = utils::expr_to_string(expr, cfunc_);
    access.inferred_type = expr->type;
    access.source_func_ea = cfunc_->entry_ea;
    access.array_stride_hint = stride_hint;

    accesses_.push_back(std::move(access));
}

void AccessPatternVisitor::process_call_argument_uses(cexpr_t* call_expr) {
    if (!call_expr || call_expr->op != cot_call || !call_expr->a) {
        return;
    }

    for (const auto& arg : *call_expr->a) {
        const cexpr_t* arg_expr = &arg;
        while (arg_expr && arg_expr->op == cot_cast) {
            arg_expr = arg_expr->x;
        }
        if (!arg_expr) {
            continue;
        }

        if (arg_expr->type.is_array() || arg_expr->type.is_struct() || is_aggregate_member_container(arg_expr)) {
            continue;
        }

        FieldAccess access;
        bool resolved = false;

        sval_t offset = 0;
        uint32_t size = 0;
        std::optional<std::uint8_t> base_indirection;
        if (extract_access(arg_expr, offset, size, &base_indirection)) {
            access.insn_ea = call_expr->ea;
            access.source_func_ea = cfunc_->entry_ea;
            access.offset = offset;
            access.size = size;
            access.access_type = determine_access_type(arg_expr);
            access.semantic_type = infer_semantic_from_usage(arg_expr, call_expr);
            access.context_expr = utils::expr_to_string(arg_expr, cfunc_);
            access.inferred_type = arg_expr->type;
            if (base_indirection.has_value()) {
                access.base_indirection = base_indirection;
            }
            resolved = true;
        } else if (arg_expr->op == cot_var) {
            auto it = local_aliases_.find(arg_expr->v.idx);
            if (it != local_aliases_.end() &&
                !address_aliases_.contains(arg_expr->v.idx)) {
                access = it->second;
                access.insn_ea = call_expr->ea;
                access.source_func_ea = cfunc_->entry_ea;
                access.context_expr = utils::expr_to_string(arg_expr, cfunc_);
                resolved = true;
            }
        }

        if (!resolved) {
            continue;
        }

        access.is_call_argument = true;
        if (access.access_type == AccessType::Unknown) {
            access.access_type = AccessType::Read;
        }
        accesses_.push_back(std::move(access));
    }
}

void AccessPatternVisitor::process_call_through_ptr(cexpr_t* call_expr) {
    if (!call_expr || call_expr->op != cot_call) return;

    cexpr_t* callee = call_expr->x;
    if (!callee) return;

    while (callee->op == cot_cast) {
        callee = callee->x;
    }

    tinfo_t funcptr_type = build_funcptr_type(call_expr);

    auto add_fp_access = [&](sval_t offset, SemanticType sem, bool is_vtable, sval_t slot_offset) {
        if (is_vtable && !is_valid_vtable_slot_offset(slot_offset)) {
            return;
        }
        FieldAccess access;
        access.insn_ea = call_expr->ea;
        access.source_func_ea = cfunc_->entry_ea;
        access.offset = offset;
        access.size = get_ptr_size();
        access.access_type = AccessType::Call;
        access.semantic_type = sem;
        access.context_expr = utils::expr_to_string(call_expr, cfunc_);
        if (!funcptr_type.empty()) {
            access.inferred_type = funcptr_type;
        }

        if (is_vtable) {
            access.is_vtable_access = true;
            access.vtable_slot = slot_offset / get_ptr_size();
            (void)access.set_vtable_nested_access(offset, slot_offset, funcptr_type);
        }

        accesses_.push_back(std::move(access));
    };

    // Pattern 1: Direct call through dereferenced var: (*var)(args)
    // Pattern 2: VTable call: (*(*(type**)var + slot))(args)
    if (callee->op == cot_ptr) {
        const cexpr_t* ptr = callee->x;
        while (ptr && ptr->op == cot_cast) {
            ptr = ptr->x;
        }
        if (!ptr) {
            return;
        }

        // Check for double dereference (vtable pattern)
        if (ptr->op == cot_add || ptr->op == cot_ptr) {
            const cexpr_t* base_ptr = ptr;
            sval_t slot_offset = 0;

            if (ptr->op == cot_add) {
                if (ptr->y->op == cot_num) {
                    slot_offset = ptr->y->numval();
                    base_ptr = ptr->x;
                } else if (ptr->x->op == cot_num) {
                    slot_offset = ptr->x->numval();
                    base_ptr = ptr->y;
                }
            }

            while (base_ptr && base_ptr->op == cot_cast) {
                base_ptr = base_ptr->x;
            }

            if (base_ptr && base_ptr->op == cot_ptr) {
                auto arith = utils::extract_ptr_arith(base_ptr->x);
                if (arith.valid && arith.var_idx == target_var_idx_) {
                    add_fp_access(arith.offset, SemanticType::VTablePointer, true, slot_offset);
                    return;
                }
            }
        }

        // Simple dereference call: (*var)(args)
        auto arith = utils::extract_ptr_arith(ptr);
        if (arith.valid && arith.var_idx == target_var_idx_) {
            add_fp_access(arith.offset, SemanticType::FunctionPointer, false, 0);
            return;
        }
    }

    // Member function pointer call: obj->fp(args)
    if (callee->op == cot_memptr || callee->op == cot_memref) {
        auto arith = utils::extract_ptr_arith(callee->x);
        if (arith.valid && arith.var_idx == target_var_idx_) {
            const auto member_offset = checked_sval_from_u64(callee->m);
            const auto total = member_offset.has_value()
                ? checked_sval_add(arith.offset, *member_offset)
                : std::nullopt;
            if (total.has_value()) {
                add_fp_access(*total, SemanticType::FunctionPointer, false, 0);
            }
            return;
        }
    }

    // Indexed function pointer call: fp_array[idx](args)
    if (callee->op == cot_idx) {
        if (!callee->y || callee->y->op != cot_num) {
            // The normal array visitor handles proven finite index ranges.
            return;
        }
        auto arith = utils::extract_ptr_arith(callee->x);
        if (arith.valid && arith.var_idx == target_var_idx_) {
            sval_t offset = arith.offset;
            if (callee->y && callee->y->op == cot_num) {
                const auto index = checked_sval_from_u64(callee->y->numval());
                const auto scaled = index.has_value()
                    ? checked_sval_mul(*index, static_cast<sval_t>(get_ptr_size()))
                    : std::nullopt;
                const auto total = scaled.has_value()
                    ? checked_sval_add(offset, *scaled)
                    : std::nullopt;
                if (!total.has_value()) {
                    return;
                }
                offset = *total;
            }
            add_fp_access(offset, SemanticType::FunctionPointer, false, 0);
            return;
        }
    }
}

void AccessPatternVisitor::record_bitfield_access(const cexpr_t* expr, sval_t offset,
                                                    uint32_t size, const BitfieldInfo& info,
                                                    const std::optional<std::uint8_t>& base_indirection) {
    FieldAccess access;
    access.insn_ea = expr->ea;
    access.source_func_ea = cfunc_->entry_ea;
    access.offset = offset;
    access.size = size;
    access.access_type = AccessType::Read;
    access.semantic_type = SemanticType::UnsignedInteger;
    access.context_expr = utils::expr_to_string(expr, cfunc_);
    access.inferred_type = expr->type;
    access.is_call_argument = is_call_argument_use(expr);
    if (base_indirection.has_value()) {
        access.base_indirection = base_indirection;
    }
    access.add_bitfield(info);

    accesses_.push_back(std::move(access));
}


bool AccessPatternVisitor::extract_access(const cexpr_t* expr, sval_t& offset, uint32_t& size,
                                               std::optional<std::uint8_t>* base_indirection) const {
    if (base_indirection) {
        base_indirection->reset();
    }

    if (!expr) return false;

    if (expr->op == cot_ptr) {
        auto arith = resolve_ptr_arith(expr);
        if (!arith.valid || arith.var_idx != target_var_idx_) return false;
        if (is_pointee_access(arith)) return false;
        offset = arith.offset;
        size = expr->type.empty() ? get_ptr_size() : utils::get_type_size(expr->type, get_ptr_size());
        if (base_indirection && arith.base_indirection > 0) {
            *base_indirection = arith.base_indirection;
        }
        return true;
    }

    if (expr->op == cot_memptr || expr->op == cot_memref) {
        auto arith = resolve_ptr_arith(expr->x);
        if (!arith.valid || arith.var_idx != target_var_idx_) return false;
        if (is_pointee_access(arith)) return false;
        const auto member_offset = checked_sval_from_u64(expr->m);
        const auto total = member_offset.has_value()
            ? checked_sval_add(arith.offset, *member_offset)
            : std::nullopt;
        if (!total.has_value()) return false;
        offset = *total;
        size = expr->type.empty() ? get_ptr_size() : utils::get_type_size(expr->type, get_ptr_size());
        if (base_indirection && arith.base_indirection > 0) {
            *base_indirection = arith.base_indirection;
        }
        return true;
    }

    return false;
}

bool AccessPatternVisitor::compute_bitfield(std::uint64_t mask, int shift,
                                               std::uint16_t& bit_offset,
                                               std::uint16_t& bit_size) const {
    if (mask == 0 || shift < 0 || shift > 63) return false;

    int lsb = 0;
    while (lsb < 64 && ((mask >> lsb) & 1ULL) == 0) {
        ++lsb;
    }

    int msb = 63;
    while (msb >= 0 && ((mask >> msb) & 1ULL) == 0) {
        --msb;
    }

    if (lsb > msb) return false;

    int width = msb - lsb + 1;
    std::uint64_t contig = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
    if ((mask >> lsb) != contig) return false;

    bit_offset = static_cast<std::uint16_t>(lsb + shift);
    bit_size = static_cast<std::uint16_t>(width);
    return bit_size > 0;
}

tinfo_t AccessPatternVisitor::build_funcptr_type(const cexpr_t* call_expr) const {
    tinfo_t result;
    if (!call_expr || call_expr->op != cot_call) return result;

    if (call_expr->x && !call_expr->x->type.empty()) {
        tinfo_t callee_type = call_expr->x->type;
        if (callee_type.is_funcptr()) {
            return callee_type;
        }
        if (callee_type.is_func() && result.create_ptr(callee_type)) {
            return result;
        }
    }

    func_type_data_t ftd;
    if (!call_expr->type.empty()) {
        ftd.rettype = call_expr->type;
    } else {
        ftd.rettype.create_simple_type(BTF_VOID);
    }
    ftd.set_cc(CM_CC_UNKNOWN);

    if (call_expr->a) {
        for (const auto& arg : *call_expr->a) {
            tinfo_t arg_type = arg.type;
            if (arg_type.empty()) {
                tinfo_t void_type;
                void_type.create_simple_type(BTF_VOID);
                arg_type.create_ptr(void_type);
            }
            funcarg_t farg;
            farg.type = arg_type;
            ftd.push_back(farg);
        }
    }

    tinfo_t func_type;
    if (func_type.create_func(ftd)) {
        result.create_ptr(func_type);
    }

    return result;
}

bool AccessPatternVisitor::involves_target_var(const cexpr_t* expr) const {
    if (!expr) return false;

    auto arith = resolve_ptr_arith(expr);
    if (arith.valid && arith.var_idx == target_var_idx_) {
        return true;
    }

    if (expr->op == cot_var) {
        if (expr->v.idx == target_var_idx_) {
            return true;
        }
        return local_aliases_.find(expr->v.idx) != local_aliases_.end();
    }

    // Recurse through common operations
    switch (expr->op) {
        case cot_cast:
        case cot_ref:
        case cot_ptr:
        case cot_memref:
        case cot_memptr:
            return involves_target_var(expr->x);

        case cot_add:
        case cot_sub:
            return involves_target_var(expr->x) || involves_target_var(expr->y);

        case cot_idx:
            return involves_target_var(expr->x);

        default:
            return false;
    }
}

bool AccessPatternVisitor::is_call_argument_use(const cexpr_t* expr) const {
    const cexpr_t* current = expr;

    for (size_t i = 0; i < parents.size(); ++i) {
        const citem_t* parent_item = parents[parents.size() - 1 - i];
        if (!parent_item || !parent_item->is_expr()) {
            continue;
        }

        const cexpr_t* parent = static_cast<const cexpr_t*>(parent_item);
        if (parent->op == cot_call) {
            return parent->x != current;
        }

        current = parent;
    }

    return false;
}

SemanticType AccessPatternVisitor::infer_semantic_from_usage(const cexpr_t* expr, const cexpr_t* parent) {
    if (!expr) return SemanticType::Unknown;

    // Check the type first
    if (!expr->type.empty()) {
        if (expr->type.is_ptr()) {
            return SemanticType::Pointer;
        }
        if (expr->type.is_funcptr()) {
            return SemanticType::FunctionPointer;
        }
        if (expr->type.is_floating()) {
            return expr->type.get_size() == 4 ? SemanticType::Float : SemanticType::Double;
        }
    }

    // Check parent context
    if (parent) {
        switch (parent->op) {
            case cot_call:
                // Value is used as function pointer
                if (parent->x == expr) {
                    return SemanticType::FunctionPointer;
                }
                break;

            case cot_ptr:
                // Value is being dereferenced - it's a pointer
                return SemanticType::Pointer;

            case cot_fadd:
            case cot_fsub:
            case cot_fmul:
            case cot_fdiv:
                return SemanticType::Double;

            case cot_ult:
            case cot_ule:
            case cot_ugt:
            case cot_uge:
                return SemanticType::UnsignedInteger;

            default:
                break;
        }
    }

    // Default based on size
    std::uint32_t size = utils::get_type_size(expr->type, get_ptr_size());
    if (size == get_ptr_size()) {
        // Could be pointer or integer - check if it's ever dereferenced
        return SemanticType::Unknown;
    }

    return SemanticType::Integer;
}

AccessType AccessPatternVisitor::determine_access_type(const cexpr_t* expr, const cexpr_t** out_rhs) {
    if (out_rhs) *out_rhs = nullptr;
    // Walk up parents to determine if this is a read or write
    const cexpr_t* current = expr;

    for (size_t i = 0; i < parents.size(); ++i) {
        const citem_t* parent_item = parents[parents.size() - 1 - i];
        if (!parent_item || !parent_item->is_expr()) {
            continue;
        }

        const cexpr_t* parent = static_cast<const cexpr_t*>(parent_item);

        switch (parent->op) {
            case cot_asg:
            case cot_asgbor:
            case cot_asgxor:
            case cot_asgband:
            case cot_asgadd:
            case cot_asgsub:
            case cot_asgmul:
            case cot_asgsshr:
            case cot_asgushr:
            case cot_asgshl:
            case cot_asgsdiv:
            case cot_asgudiv:
            case cot_asgsmod:
            case cot_asgumod:
                // If current is the left side, it's a write
                if (parent->x == current) {
                    // For compound assignments, it's both read and write
                    if (parent->op != cot_asg) {
                        return AccessType::ReadWrite;
                    }
                    if (out_rhs) *out_rhs = parent->y;
                    return AccessType::Write;
                }
                return AccessType::Read;

            case cot_preinc:
            case cot_predec:
            case cot_postinc:
            case cot_postdec:
                return AccessType::ReadWrite;

            case cot_ref:
                return AccessType::AddressTaken;

            default:
                current = parent;
                break;
        }
    }

    return AccessType::Read;
}

bool AccessPatternVisitor::is_zero_initialization(const cexpr_t* expr) const {
    // Walk up parents to find an assignment
    const cexpr_t* current = expr;

    for (size_t i = 0; i < parents.size(); ++i) {
        const citem_t* parent_item = parents[parents.size() - 1 - i];
        if (!parent_item || !parent_item->is_expr()) {
            continue;
        }

        const cexpr_t* parent = static_cast<const cexpr_t*>(parent_item);

        if (parent->op == cot_asg && parent->x == current) {
            // This is a write - check if the value is zero
            const cexpr_t* rhs = parent->y;
            if (!rhs) return false;

            // Check for numeric constant 0
            if (rhs->op == cot_num && rhs->numval() == 0) {
                return true;
            }

            // Check for cast of 0: (type)0
            if (rhs->op == cot_cast && rhs->x && 
                rhs->x->op == cot_num && rhs->x->numval() == 0) {
                return true;
            }

            return false;
        }

        current = parent;
    }

    return false;
}

void AccessPatternVisitor::extract_and_add_rhs_constant(FieldAccess& access, const cexpr_t* rhs) const {
    if (!rhs) return;
    while (rhs && (rhs->op == cot_cast || rhs->op == cot_ref)) {
        rhs = rhs->x;
    }
    if (!rhs) return;
    if (rhs->op == cot_num) {
        access.add_observed_constant(rhs->numval());
    } else if (rhs->op == cot_obj) {
        access.add_observed_constant(rhs->obj_ea);
    }
}

// ============================================================================
// AccessCollector Implementation
// ============================================================================

AccessPattern AccessCollector::collect(ea_t func_ea, int var_idx) {
    AccessPattern pattern;
    pattern.func_ea = func_ea;
    pattern.var_idx = var_idx;

    cfuncptr_t cfunc = utils::get_cfunc(func_ea);
    if (!cfunc) {
        return pattern;
    }

    return collect(cfunc, var_idx);
}

AccessPattern AccessCollector::collect(cfunc_t* cfunc, int var_idx) {
    AccessPattern pattern;
    if (!cfunc) return pattern;

    func_t* func = cfunc->entry_ea != BADADDR ? get_func(cfunc->entry_ea) : nullptr;
    pattern.func_ea = func ? func->start_ea : BADADDR;
    pattern.var_idx = var_idx;

    // Get variable info
    lvars_t& lvars = *cfunc->get_lvars();
    if (var_idx >= 0 && static_cast<size_t>(var_idx) < lvars.size()) {
        lvar_t& var = lvars[var_idx];
        pattern.var_name = var.name;
        pattern.original_type = var.type();
    }

    // Collect accesses
    AccessPatternVisitor visitor(cfunc, var_idx);
    visitor.apply_to(&cfunc->body, nullptr);

    pattern.accesses = std::move(visitor.mutable_accesses());

    // Post-process
    analyze_accesses(pattern);
    deduplicate_accesses(pattern);
    analyze_accesses(pattern);

    if (options_.debug_mode) {
        qstring func_name;
        get_func_name(&func_name, pattern.func_ea);
        msg("Structor: collected %zu accesses for %s var_idx=%d\n",
            pattern.accesses.size(), func_name.c_str(), var_idx);
        for (const auto& access : pattern.accesses) {
            msg("Structor:   access off=0x%llX size=%u kind=%s zero_init=%s sem=%s type=%s base_indir=%u call_arg=%s ctx=%s\n",
                static_cast<unsigned long long>(access.offset),
                access.size,
                access_type_str(access.access_type),
                access.is_zero_init ? "true" : "false",
                semantic_type_str(access.semantic_type),
                access.inferred_type.dstr(),
                access.base_indirection.value_or(0),
                access.is_call_argument ? "true" : "false",
                access.context_expr.c_str());
        }
    }

    if (options_.vtable_detection) {
        detect_vtable_pattern(pattern);
    }

    return pattern;
}

AccessPattern AccessCollector::collect(ea_t func_ea, const char* var_name) {
    AccessPattern pattern;
    pattern.func_ea = func_ea;

    cfuncptr_t cfunc = utils::get_cfunc(func_ea);
    if (!cfunc) {
        return pattern;
    }

    lvar_t* var = utils::find_lvar_by_name(cfunc, var_name);
    if (!var) {
        return pattern;
    }

    // Find index
    lvars_t& lvars = *cfunc->get_lvars();
    for (size_t i = 0; i < lvars.size(); ++i) {
        if (&lvars[i] == var) {
            return collect(cfunc, static_cast<int>(i));
        }
    }

    return pattern;
}

void AccessCollector::analyze_accesses(AccessPattern& pattern) {
    if (pattern.accesses.empty()) return;

    qvector<FieldAccess> bounded;
    bounded.reserve(pattern.accesses.size());
    for (auto& access : pattern.accesses) {
        if (checked_interval_end(access.offset, access.size).has_value()) {
            bounded.push_back(std::move(access));
        }
    }
    pattern.accesses = std::move(bounded);
    if (pattern.accesses.empty()) {
        pattern.min_offset = 0;
        pattern.max_offset = 0;
        return;
    }

    pattern.sort_by_offset();

    // Recalculate min/max
    pattern.min_offset = pattern.accesses.front().offset;
    pattern.max_offset = *checked_interval_end(
        pattern.accesses.front().offset, pattern.accesses.front().size);

    for (const auto& access : pattern.accesses) {
        pattern.min_offset = std::min(pattern.min_offset, access.offset);
        const auto access_end = checked_interval_end(access.offset, access.size);
        if (access_end.has_value()) {
            pattern.max_offset = std::max(pattern.max_offset, *access_end);
        }
    }
}

void AccessCollector::deduplicate_accesses(AccessPattern& pattern) {
    if (pattern.accesses.size() <= 1) return;

    // Apply predicate filter first (adopted from Suture)
    if (options_.access_filter) {
        qvector<FieldAccess> filtered;
        filtered.reserve(pattern.accesses.size());
        for (auto& access : pattern.accesses) {
            if (options_.access_filter(access)) {
                filtered.push_back(std::move(access));
            }
        }
        pattern.accesses = std::move(filtered);

        utils::debug_log("After predicate filtering: %zu accesses remain", pattern.accesses.size());
    }

    if (pattern.accesses.size() <= 1) return;

    qvector<FieldAccess> unique;
    unique.reserve(pattern.accesses.size());

    for (auto& access : pattern.accesses) {
        bool found = false;
        for (auto& existing : unique) {
            if (existing.offset == access.offset &&
                existing.size == access.size &&
                field_access_evidence_compatible(existing, access)) {
                merge_field_access_evidence(existing, access);
                found = true;
                break;
            }
        }

        if (!found) {
            unique.push_back(std::move(access));
        }
    }

    pattern.accesses = std::move(unique);
    std::sort(pattern.accesses.begin(), pattern.accesses.end(),
              canonical_field_access_less);

    // Drop coarse aggregate accesses when we already observed finer-grained
    // accesses inside the same region. These usually come from decompiler-
    // synthesized array/struct expressions and can overwhelm real field
    // recovery by forcing overly large direct candidates.
    qvector<FieldAccess> filtered;
    filtered.reserve(pattern.accesses.size());

    for (const auto& access : pattern.accesses) {
        bool redundant_aggregate = false;
        if (!access.inferred_type.empty() &&
            (access.inferred_type.is_array() || access.inferred_type.is_struct())) {
            int contained_smaller = 0;
            const auto access_end = checked_interval_end(access.offset, access.size);
            if (!access_end.has_value()) {
                continue;
            }

            for (const auto& other : pattern.accesses) {
                if (&other == &access) {
                    continue;
                }
                const auto other_end = checked_interval_end(other.offset, other.size);
                if (other_end.has_value() && other.offset >= access.offset &&
                    *other_end <= *access_end &&
                    (other.size < access.size || other.offset != access.offset)) {
                    ++contained_smaller;
                }
            }

            redundant_aggregate = contained_smaller >= 2;
        }

        if (!redundant_aggregate) {
            filtered.push_back(access);
        }
    }

    pattern.accesses = std::move(filtered);
    utils::debug_log("After deduplication: %zu unique accesses", pattern.accesses.size());
}

void AccessCollector::detect_vtable_pattern(AccessPattern& pattern) {
    // Look for vtable access patterns
    for (const auto& access : pattern.accesses) {
        if (access.is_vtable_access) {
            pattern.has_vtable = true;
            pattern.vtable_offset = access.offset;
            break;
        }
    }

    // Also check for pointer at offset 0 that's always dereferenced and called through
    if (!pattern.has_vtable) {
        int deref_calls_at_zero = 0;
        for (const auto& access : pattern.accesses) {
            if (access.offset == 0 &&
                access.semantic_type == SemanticType::VTablePointer) {
                ++deref_calls_at_zero;
            }
        }
        if (deref_calls_at_zero >= 1) {
            pattern.has_vtable = true;
            pattern.vtable_offset = 0;
        }
    }
}

} // namespace structor
