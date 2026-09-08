#pragma once

#include "synth_types.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <unordered_set>

namespace structor {

struct BitfieldInfo;

/// Visitor that collects all access patterns for a specific variable
class AccessPatternVisitor : public ctree_visitor_t {
public:
    AccessPatternVisitor(cfunc_t* cfunc, int target_var_idx);

    int idaapi visit_expr(cexpr_t* expr) override;
    int idaapi visit_insn(cinsn_t* insn) override;

    [[nodiscard]] const qvector<FieldAccess>& accesses() const noexcept {
        return accesses_;
    }

    [[nodiscard]] qvector<FieldAccess>& mutable_accesses() noexcept {
        return accesses_;
    }

private:
    struct IndexRange {
        sval_t first = std::numeric_limits<sval_t>::min();
        sval_t last = std::numeric_limits<sval_t>::max();
        bool first_known = false;
        bool last_known = false;

        [[nodiscard]] IndexRange intersect(const IndexRange& other) const {
            return {std::max(first, other.first), std::min(last, other.last),
                    first_known || other.first_known, last_known || other.last_known};
        }
        [[nodiscard]] IndexRange unite(const IndexRange& other) const {
            return {std::min(first, other.first), std::max(last, other.last),
                    first_known && other.first_known, last_known && other.last_known};
        }
    };

    struct LoopIndexEffects {
        std::unordered_set<int> written_or_referenced_vars;
        bool has_call = false;
    };

    struct IndexComparison {
        int var_idx = -1;
        std::size_t version = 0;
        IndexRange when_true;
        IndexRange when_false;
    };

    void process_dereference(cexpr_t* expr, const cexpr_t* ptr_expr);
    void process_memptr_access(cexpr_t* expr);
    void process_call_argument_uses(cexpr_t* call_expr);
    void process_call_through_ptr(cexpr_t* call_expr);
    void process_array_access(cexpr_t* expr);
    void process_assignment(cexpr_t* expr);
    void process_constant_comparison(cexpr_t* expr);
    void process_index_bound(cexpr_t* expr);
    [[nodiscard]] IndexRange condition_index_range(
        const cexpr_t* condition, int var_idx, bool truth, int depth = 0) const;
    [[nodiscard]] std::optional<IndexRange> bounded_index_range(
        const cexpr_t* access_expr, int var_idx) const;
    [[nodiscard]] bool loop_may_change_index(const cinsn_t* loop, int var_idx) const;
    void invalidate_local_var_state(int var_idx, bool clear_pending_constants);

    void record_bitfield_access(const cexpr_t* expr, sval_t offset, uint32_t size,
                                const BitfieldInfo& info,
                                const std::optional<std::uint8_t>& base_indirection);
    [[nodiscard]] bool extract_access(const cexpr_t* expr, sval_t& offset, uint32_t& size,
                                      std::optional<std::uint8_t>* base_indirection) const;
    [[nodiscard]] utils::PtrArithInfo resolve_ptr_arith(const cexpr_t* expr) const;
    void extract_and_add_rhs_constant(FieldAccess& access, const cexpr_t* rhs) const;
    [[nodiscard]] bool compute_bitfield(std::uint64_t mask, int shift,
                                        std::uint16_t& bit_offset,
                                        std::uint16_t& bit_size) const;
    [[nodiscard]] tinfo_t build_funcptr_type(const cexpr_t* call_expr) const;

    [[nodiscard]] bool involves_target_var(const cexpr_t* expr) const;
    [[nodiscard]] bool is_call_argument_use(const cexpr_t* expr) const;
    [[nodiscard]] SemanticType infer_semantic_from_usage(const cexpr_t* expr, const cexpr_t* parent);
    [[nodiscard]] AccessType determine_access_type(const cexpr_t* expr, const cexpr_t** out_rhs = nullptr);
    [[nodiscard]] bool is_zero_initialization(const cexpr_t* expr) const;

    cfunc_t* cfunc_;
    int target_var_idx_;
    bool has_unstructured_control_flow_;
    qvector<FieldAccess> accesses_;
    std::unordered_map<int, FieldAccess> local_aliases_;
    // Distinguish copies of the base address from values loaded from a field.
    // Address copies support later dereferences, but are not field reads.
    std::unordered_set<int> address_aliases_;
    std::unordered_map<int, qvector<std::uint64_t>> pending_constants_;
    // A comparison is usable only on the control-flow edge where it holds and
    // while the index still has the value tested by that comparison.
    std::unordered_map<const cexpr_t*, IndexComparison> index_comparisons_;
    std::unordered_map<int, std::size_t> local_var_versions_;
    std::unordered_set<int> escaped_local_vars_;
    mutable std::unordered_map<const cinsn_t*, LoopIndexEffects> loop_index_effects_;
};

/// Collects all access patterns for a variable in a function
class AccessCollector {
public:
    explicit AccessCollector(const SynthOptions& opts = Config::instance().options())
        : options_(opts) {}

    /// Collect all accesses to a variable in a function
    [[nodiscard]] AccessPattern collect(ea_t func_ea, int var_idx);

    /// Collect accesses using existing cfunc
    [[nodiscard]] AccessPattern collect(cfunc_t* cfunc, int var_idx);

    /// Collect accesses for a variable by name
    [[nodiscard]] AccessPattern collect(ea_t func_ea, const char* var_name);

private:
    void analyze_accesses(AccessPattern& pattern);
    void deduplicate_accesses(AccessPattern& pattern);
    void detect_vtable_pattern(AccessPattern& pattern);

    // Own the options snapshot. Public construction with SynthOptions{} must
    // not retain a reference to a temporary, and collection must remain
    // isolated from later global Config mutations.
    SynthOptions options_;
};

} // namespace structor
