#pragma once

#include "synth_types.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <memory>

namespace structor {

/// Reports pseudocode expressions that would render as recovered field
/// accesses after the corresponding type is applied. Hex-Rays ctree is not a
/// durable rewrite store; persistent expression rendering is driven by lvar
/// and global type application. This helper may add a verified summary comment
/// but never reports a planned expression as applied.
class PseudocodeRewriter {
public:
    explicit PseudocodeRewriter(const SynthOptions& opts = Config::instance().options())
        : options_(opts) {}

    /// Rewrite pseudocode after structure synthesis
    [[nodiscard]] RewriteResult rewrite(
        cfunc_t* cfunc,
        int var_idx,
        const SynthStruct& synth_struct);

    /// Refresh the decompiler view
    void refresh_view(vdui_t* vdui);

    /// Highlight transformed expressions (visual feedback)
    void highlight_transforms(vdui_t* vdui, const RewriteResult& result);

private:
    /// Microcode modifier for rewriting expressions
    class ExprRewriter : public ctree_visitor_t {
    public:
        ExprRewriter(cfunc_t* cfunc, int var_idx, const SynthStruct& synth_struct)
            : ctree_visitor_t(CV_PARENTS | CV_POST)
            , cfunc_(cfunc)
            , var_idx_(var_idx)
            , synth_struct_(synth_struct) {}

        int idaapi visit_expr(cexpr_t* expr) override;

        [[nodiscard]] const qvector<RewriteTransform>& transforms() const noexcept {
            return transforms_;
        }

    private:
        bool rewrite_ptr_deref(cexpr_t* expr);
        bool rewrite_memptr_access(cexpr_t* expr);
        bool rewrite_array_access(cexpr_t* expr);
        bool rewrite_vtable_call(cexpr_t* expr);

        [[nodiscard]] const SynthField* find_field_at(sval_t offset) const;
        [[nodiscard]] const VTableSlot* find_vtable_slot(int slot_idx) const;

        cfunc_t* cfunc_;
        int var_idx_;
        const SynthStruct& synth_struct_;
        qvector<RewriteTransform> transforms_;
    };

    void apply_user_cmts(cfunc_t* cfunc, const SynthStruct& synth_struct, int var_idx);

    SynthOptions options_;
};

// ============================================================================
// ExprRewriter Implementation
// ============================================================================

inline int PseudocodeRewriter::ExprRewriter::visit_expr(cexpr_t* expr) {
    if (!expr) return 0;

    bool rewritten = false;

    switch (expr->op) {
        case cot_ptr:
            rewritten = rewrite_ptr_deref(expr);
            break;

        case cot_memptr:
            rewritten = rewrite_memptr_access(expr);
            break;

        case cot_idx:
            rewritten = rewrite_array_access(expr);
            break;

        case cot_call:
            rewritten = rewrite_vtable_call(expr);
            break;

        default:
            break;
    }

    (void)rewritten;
    return 0;
}

inline bool PseudocodeRewriter::ExprRewriter::rewrite_ptr_deref(cexpr_t* expr) {
    // Pattern: *(var + offset) -> var->field_XX
    auto arith = utils::extract_ptr_arith(expr->x);
    if (!arith.valid || arith.var_idx != var_idx_) {
        return false;
    }

    const SynthField* field = find_field_at(arith.offset);
    if (!field || field->is_padding) {
        return false;
    }

    RewriteTransform transform;
    transform.insn_ea = expr->ea;
    transform.original_expr = utils::expr_to_string(expr, cfunc_);

    // Record a diagnostic plan. No ctree mutation is performed here.
    transform.rewritten_expr.sprnt("a%d->%s", var_idx_, field->name.c_str());
    transform.success = false;
    transform.failure_reason =
        "Expression plan only; apply the recovered type for durable rendering";

    transforms_.push_back(std::move(transform));
    return true;
}

inline bool PseudocodeRewriter::ExprRewriter::rewrite_memptr_access(cexpr_t* expr) {
    auto arith = utils::extract_ptr_arith(expr->x);
    if (!arith.valid || arith.var_idx != var_idx_) {
        return false;
    }

    const auto member_offset = checked_sval_from_u64(expr->m);
    const auto total_offset = member_offset.has_value()
        ? checked_sval_add(arith.offset, *member_offset)
        : std::nullopt;
    if (!total_offset.has_value()) {
        return false;
    }
    const SynthField* field = find_field_at(*total_offset);
    if (!field || field->is_padding) {
        return false;
    }

    RewriteTransform transform;
    transform.insn_ea = expr->ea;
    transform.original_expr = utils::expr_to_string(expr, cfunc_);
    transform.rewritten_expr.sprnt("a%d->%s", var_idx_, field->name.c_str());
    transform.success = false;
    transform.failure_reason =
        "Expression plan only; apply the recovered type for durable rendering";

    transforms_.push_back(std::move(transform));
    return true;
}

inline bool PseudocodeRewriter::ExprRewriter::rewrite_array_access(cexpr_t* expr) {
    auto arith = utils::extract_ptr_arith(expr->x);
    if (!arith.valid || arith.var_idx != var_idx_) {
        return false;
    }

    sval_t offset = arith.offset;
    if (expr->y->op == cot_num) {
        tinfo_t elem_type = expr->x->type.get_pointed_object();
        if (!elem_type.empty()) {
            const size_t element_size = elem_type.get_size();
            const auto index = checked_sval_from_u64(expr->y->numval());
            if (element_size == BADSIZE ||
                element_size > static_cast<size_t>(
                    std::numeric_limits<sval_t>::max()) ||
                !index.has_value()) {
                return false;
            }
            const auto scaled = checked_sval_mul(
                *index, static_cast<sval_t>(element_size));
            const auto total = scaled.has_value()
                ? checked_sval_add(offset, *scaled)
                : std::nullopt;
            if (!total.has_value()) {
                return false;
            }
            offset = *total;
        }
    }

    const SynthField* field = find_field_at(offset);
    if (!field || field->is_padding) {
        return false;
    }

    RewriteTransform transform;
    transform.insn_ea = expr->ea;
    transform.original_expr = utils::expr_to_string(expr, cfunc_);
    transform.rewritten_expr.sprnt("a%d->%s", var_idx_, field->name.c_str());
    transform.success = false;
    transform.failure_reason =
        "Expression plan only; apply the recovered type for durable rendering";

    transforms_.push_back(std::move(transform));
    return true;
}

inline bool PseudocodeRewriter::ExprRewriter::rewrite_vtable_call(cexpr_t* expr) {
    if (!synth_struct_.has_vtable()) return false;

    // Pattern: (*(*(var + vtbl_off) + slot_off))(args)
    // -> var->vtbl->slot_N(args)

    cexpr_t* callee = expr->x;
    if (!callee || callee->op != cot_ptr) return false;

    const cexpr_t* slot_expr = callee->x;
    sval_t slot_offset = 0;
    const cexpr_t* vtbl_deref = slot_expr;

    if (slot_expr->op == cot_add) {
        if (slot_expr->y && slot_expr->y->op == cot_num) {
            const auto converted =
                checked_sval_from_u64(slot_expr->y->numval());
            if (!converted.has_value()) return false;
            slot_offset = *converted;
        }
        vtbl_deref = slot_expr->x;
    }

    if (!vtbl_deref || vtbl_deref->op != cot_ptr) return false;

    auto arith = utils::extract_ptr_arith(vtbl_deref->x);
    if (!arith.valid || arith.var_idx != var_idx_) return false;

    if (slot_offset < 0 ||
        slot_offset % static_cast<sval_t>(get_ptr_size()) != 0 ||
        slot_offset / static_cast<sval_t>(get_ptr_size()) >
            std::numeric_limits<int>::max()) {
        return false;
    }
    int slot_idx = static_cast<int>(
        slot_offset / static_cast<sval_t>(get_ptr_size()));
    const VTableSlot* slot = find_vtable_slot(slot_idx);
    if (!slot) return false;

    RewriteTransform transform;
    transform.insn_ea = expr->ea;
    transform.original_expr = utils::expr_to_string(expr, cfunc_);
    transform.rewritten_expr.sprnt("a%d->vtbl->%s(...)", var_idx_, slot->name.c_str());
    transform.success = false;
    transform.failure_reason =
        "Expression plan only; apply the recovered type for durable rendering";

    transforms_.push_back(std::move(transform));
    return true;
}

inline const SynthField* PseudocodeRewriter::ExprRewriter::find_field_at(sval_t offset) const {
    for (const auto& field : synth_struct_.fields) {
        if (field.offset == offset) {
            return &field;
        }
        // Also check if offset falls within field range
        const auto field_end = checked_interval_end(field.offset, field.size);
        if (field_end.has_value() && offset >= field.offset &&
            offset < *field_end) {
            return &field;
        }
    }
    return nullptr;
}

inline const VTableSlot* PseudocodeRewriter::ExprRewriter::find_vtable_slot(int slot_idx) const {
    if (!synth_struct_.has_vtable()) return nullptr;

    for (const auto& slot : synth_struct_.vtable->slots) {
        if (static_cast<int>(slot.index) == slot_idx) {
            return &slot;
        }
    }
    return nullptr;
}

// ============================================================================
// PseudocodeRewriter Implementation
// ============================================================================

inline RewriteResult PseudocodeRewriter::rewrite(
    cfunc_t* cfunc,
    int var_idx,
    const SynthStruct& synth_struct)
{
    RewriteResult result;

    if (!cfunc) {
        return result;
    }

    // Visit all expressions and collect rewrites
    ExprRewriter rewriter(cfunc, var_idx, synth_struct);
    rewriter.apply_to(&cfunc->body, nullptr);

    result.transforms = rewriter.transforms();

    for (const auto& t : result.transforms) {
        if (t.success) {
            ++result.success_count;
        } else {
            ++result.failure_count;
        }
    }

    // Apply user comments for field names at specific locations
    apply_user_cmts(cfunc, synth_struct, var_idx);

    // Planned expressions alone do not justify claiming a ctree refresh.
    result.refresh_required = false;

    return result;
}

inline void PseudocodeRewriter::refresh_view(vdui_t* vdui) {
    if (!vdui) return;

    // Refresh the pseudocode view
    vdui->refresh_view(true);
}

inline void PseudocodeRewriter::highlight_transforms(vdui_t* vdui, const RewriteResult& result) {
    if (!vdui || !options_.highlight_changes) return;

    // Highlighting transformed expressions
    // This uses IDA's color API to temporarily highlight changed lines

    for (const auto& transform : result.transforms) {
        if (!transform.success) continue;

        // Find the line containing this EA
        const strvec_t& pseudocode = vdui->cfunc->get_pseudocode();
        for (size_t i = 0; i < pseudocode.size(); ++i) {
            const simpleline_t& line = pseudocode[i];

            // Check if this line contains an item at the transform's EA
            ctree_item_t item;
            if (vdui->cfunc->get_line_item(line.line.c_str(), 0, true, nullptr, &item, nullptr)) {
                if (item.is_citem() && item.e && item.e->ea == transform.insn_ea) {
                    // Line highlighting via the pseudocode view's built-in mechanism
                    // would require mutable access - for now we just log the transform
                    // The actual highlighting would need to be done through vdui->refresh_view()
                    // after setting up appropriate callback handlers
                }
            }
        }
    }

    // Request view refresh to apply any visual changes
    vdui->refresh_view(true);
}

inline void PseudocodeRewriter::apply_user_cmts(
    cfunc_t* cfunc,
    const SynthStruct& synth_struct,
    int var_idx)
{
    if (!cfunc || !options_.generate_comments) return;

    // Work with our own copy - don't touch cfunc->user_cmts directly.
    // restore_user_cmts returns a newly allocated map from IDB.
    const auto free_comments = +[](user_cmts_t* comments) noexcept {
        if (comments != nullptr) {
            try { user_cmts_free(comments); } catch (...) {}
        }
    };
    std::unique_ptr<user_cmts_t, decltype(free_comments)> cmts(
        restore_user_cmts(cfunc->entry_ea), free_comments);
    if (!cmts) {
        cmts.reset(user_cmts_new());
        if (!cmts) {
            return;
        }
    }

    treeloc_t loc;
    loc.ea = cfunc->entry_ea;
    loc.itp = ITP_BLOCK1;

    qstring cmt;
    cmt.sprnt("Structor: Synthesized %s for variable idx %d\n", synth_struct.name.c_str(), var_idx);
    cmt.cat_sprnt("  Size: %u bytes, %zu fields", synth_struct.size, synth_struct.field_count());

    if (synth_struct.has_vtable()) {
        cmt.cat_sprnt(", %zu vtable slots", synth_struct.vtable->slot_count());
    }

    // Never replace a user-owned comment. Structor only occupies an unused
    // location; repeated runs preserve the first generated annotation too.
    if (cmts->find(loc) != cmts->end()) {
        return;
    }
    cmts->operator[](loc) = citem_cmt_t(cmt.c_str());

    save_user_cmts(cfunc->entry_ea, cmts.get());
}

} // namespace structor
