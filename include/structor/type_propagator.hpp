#pragma once

#include "synth_types.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace structor {

class StructurePersistence;

/// Propagates synthesized types to related functions and variables
class TypePropagator {
public:
    using CfuncCache = std::unordered_map<ea_t, cfuncptr_t>;

    explicit TypePropagator(
        const SynthOptions& opts = Config::instance().options(),
        CfuncCache* shared_cfunc_cache = nullptr,
        StructurePersistence* persistence = nullptr)
        : options_(opts)
        , shared_cfunc_cache_(shared_cfunc_cache)
        , persistence_(persistence) {}

    /// Propagate type to related functions
    [[nodiscard]] PropagationResult propagate(
        ea_t origin_func,
        int origin_var_idx,
        const tinfo_t& new_type,
        PropagationDirection direction = PropagationDirection::Both);

    /// Propagate type within a single function
    [[nodiscard]] PropagationResult propagate_local(
        cfunc_t* cfunc,
        int var_idx,
        const tinfo_t& new_type);

    /// Apply type to a variable
    [[nodiscard]] bool apply_type(cfunc_t* cfunc, int var_idx, const tinfo_t& type);

    /// Apply the supplied type verbatim. This shares the same persistent and
    /// in-memory rollback plus function-signature synchronization as
    /// apply_type(), but does not convert a recovered aggregate view into a
    /// pointer. It is used by the general type-fixing path.
    [[nodiscard]] bool apply_exact_type(
        cfunc_t* cfunc, int var_idx, const tinfo_t& type);

    /// True only when the most recent apply operation could not restore its
    /// saved lvar/prototype/auxiliary state. Callers owning a named-type
    /// transaction must retain referenced types instead of deleting them.
    [[nodiscard]] bool last_application_rollback_failed() const noexcept {
        return last_application_rollback_failed_;
    }

private:
    struct VisitKey {
        ea_t func_ea = BADADDR;
        lvar_locator_t locator;

        [[nodiscard]] bool operator<(const VisitKey& other) const {
            return func_ea < other.func_ea ||
                (func_ea == other.func_ea && locator < other.locator);
        }
    };

    [[nodiscard]] bool apply_type_impl(
        cfunc_t* cfunc,
        int var_idx,
        const tinfo_t& type,
        bool normalize_structure_view);

    struct PropagationWork {
        ea_t        func_ea;
        int         var_idx;
        int         depth;
        PropagationDirection direction;
    };

    struct CalleeArgInfo {
        ea_t callee_ea = BADADDR;
        int param_idx = -1;
        bool by_ref = false;
        tinfo_t passed_type;
        sval_t member_offset = -1;
    };

    struct CallerArgInfo {
        ea_t caller_ea = BADADDR;
        int var_idx = -1;
        ea_t global_ea = BADADDR;
        bool by_ref = false;
        sval_t member_offset = -1;
    };

    void propagate_forward(
        ea_t func_ea,
        int var_idx,
        const tinfo_t& type,
        int depth,
        PropagationResult& result);

    void propagate_backward(
        ea_t func_ea,
        int var_idx,
        const tinfo_t& type,
        int depth,
        PropagationResult& result);

    void propagate_return_to_callers(
        ea_t func_ea,
        int return_var_idx,
        const tinfo_t& type,
        int depth,
        PropagationResult& result);

    void propagate_callee_args(
        const qvector<CalleeArgInfo>& callees,
        const tinfo_t& type,
        int depth,
        PropagationResult& result);

    void propagate_global_forward(
        cfunc_t* cfunc,
        ea_t global_ea,
        const tinfo_t& type,
        int depth,
        PropagationResult& result);

    void find_callees_with_arg(
        cfunc_t* cfunc,
        int var_idx,
        qvector<CalleeArgInfo>& callees);

    void find_callees_with_global(
        cfunc_t* cfunc,
        ea_t global_ea,
        qvector<CalleeArgInfo>& callees);

    [[nodiscard]] bool find_callers_with_param(
        ea_t func_ea,
        int param_idx,
        qvector<CallerArgInfo>& callers);

    void find_aliased_vars(
        cfunc_t* cfunc,
        int var_idx,
        qvector<int>& aliases);

    void find_assigned_from(
        cfunc_t* cfunc,
        int var_idx,
        qvector<std::pair<ea_t, int>>& sources);

    void find_return_sources(
        cfunc_t* cfunc,
        qvector<std::pair<int, sval_t>>& sources);

    [[nodiscard]] bool find_callers_with_return(
        ea_t func_ea,
        qvector<std::pair<ea_t, int>>& callers);

    [[nodiscard]] bool has_material_return_consumer(ea_t func_ea);

    [[nodiscard]] bool synchronize_function_signature(
        cfunc_t* cfunc,
        int var_idx,
        const tinfo_t& lvar_type);

    [[nodiscard]] bool is_parameter(cfunc_t* cfunc, int var_idx);
    [[nodiscard]] int get_param_index(cfunc_t* cfunc, int var_idx);
    [[nodiscard]] cfuncptr_t get_cfunc(ea_t func_ea);
    [[nodiscard]] std::optional<int> resolve_after_application(
        ea_t func_ea,
        const lvar_locator_t& locator);

    SynthOptions options_;
    std::set<VisitKey> visited_;
    std::unordered_set<ea_t> visited_globals_;
    CfuncCache* shared_cfunc_cache_ = nullptr;
    StructurePersistence* persistence_ = nullptr;
    bool last_application_rollback_failed_ = false;
    CfuncCache local_cfunc_cache_;

    [[nodiscard]] static VisitKey make_visit_key(
        ea_t func_ea, const lvar_locator_t& locator) {
        return VisitKey{func_ea, locator};
    }
};

} // namespace structor
