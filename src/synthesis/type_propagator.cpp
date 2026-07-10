/// @file type_propagator.cpp
/// @brief Type propagation implementation

#include <structor/type_propagator.hpp>
#include <structor/access_collector.hpp>
#include <structor/layout_synthesizer.hpp>
#include <structor/naming.hpp>
#include <structor/structure_persistence.hpp>
#include <structor/global_tinfo_transaction.hpp>

namespace structor {

namespace {

struct CalleeGroup {
    ea_t callee_ea = BADADDR;
    int param_idx = -1;
    qvector<int> info_indices;
    std::optional<lvar_locator_t> locator;
};

[[nodiscard]] std::optional<int> get_lvar_index_for_param(
    cfunc_t* cfunc,
    int param_ordinal)
{
    if (cfunc == nullptr || param_ordinal < 0) {
        return std::nullopt;
    }
    lvars_t* lvars = cfunc->get_lvars();
    if (lvars == nullptr) {
        return std::nullopt;
    }
    int observed_ordinal = 0;
    for (size_t lvar_index = 0; lvar_index < lvars->size(); ++lvar_index) {
        if (!lvars->at(lvar_index).is_arg_var()) {
            continue;
        }
        if (observed_ordinal == param_ordinal) {
            return static_cast<int>(lvar_index);
        }
        ++observed_ordinal;
    }
    return std::nullopt;
}

[[nodiscard]] bool checked_accumulate(sval_t& value, sval_t delta) noexcept {
    const auto combined = checked_sval_add(value, delta);
    if (!combined.has_value()) {
        return false;
    }
    value = *combined;
    return true;
}

[[nodiscard]] bool checked_accumulate_u64(
    sval_t& value, std::uint64_t raw_delta) noexcept {
    const auto delta = checked_sval_from_u64(raw_delta);
    return delta.has_value() && checked_accumulate(value, *delta);
}

[[nodiscard]] bool checked_subtract_u64(
    sval_t& value, std::uint64_t raw_delta) noexcept {
    const auto delta = checked_sval_from_u64(raw_delta);
    if (!delta.has_value()) {
        return false;
    }
    const auto combined = checked_sval_sub(value, *delta);
    if (!combined.has_value()) {
        return false;
    }
    value = *combined;
    return true;
}

[[nodiscard]] std::optional<sval_t> checked_scaled_ctree_constant(
    std::uint64_t raw_value,
    std::size_t scale) noexcept {
    const auto value = checked_sval_from_u64(raw_value);
    if (!value.has_value() || scale == BADSIZE ||
        scale > static_cast<std::size_t>(
            std::numeric_limits<sval_t>::max())) {
        return std::nullopt;
    }
    return checked_sval_mul(*value, static_cast<sval_t>(scale));
}

bool restore_function_type_snapshot(
    ea_t entry_ea,
    bool had_stored_type,
    const tinfo_t& stored_before) noexcept
{
    try {
        if (!had_stored_type) {
            del_tinfo(entry_ea);
            tinfo_t residual;
            const bool restored = !get_tinfo(&residual, entry_ea);
            (void)mark_cfunc_dirty(entry_ea, false);
            return restored;
        }

        if (!set_tinfo(entry_ea, &stored_before)) {
            return false;
        }
        tinfo_t restored;
        const bool restored_ok = get_tinfo(&restored, entry_ea) &&
                                 restored.equals_to(stored_before);
        (void)mark_cfunc_dirty(entry_ea, false);
        return restored_ok;
    } catch (...) {
        return false;
    }
}

bool exact_saved_lvar_equal(
    const lvar_saved_info_t& lhs,
    const lvar_saved_info_t& rhs)
{
    const bool types_equal =
        (lhs.type.empty() && rhs.type.empty()) ||
        (!lhs.type.empty() && !rhs.type.empty() && lhs.type.equals_to(rhs.type));
    return lhs.ll == rhs.ll &&
           lhs.name == rhs.name &&
           types_equal &&
           lhs.cmt == rhs.cmt &&
           lhs.size == rhs.size &&
           lhs.flags == rhs.flags;
}

bool exact_lvar_settings_equal(
    const lvar_uservec_t& lhs,
    const lvar_uservec_t& rhs)
{
    if (lhs.stkoff_delta != rhs.stkoff_delta ||
        lhs.ulv_flags != rhs.ulv_flags ||
        lhs.lvvec.size() != rhs.lvvec.size() ||
        lhs.lmaps != rhs.lmaps) {
        return false;
    }

    // Hex-Rays may canonicalize vector order on persistence. Compare by the
    // SDK-defined stable locator while retaining every persisted attribute.
    for (const auto& expected : lhs.lvvec) {
        std::size_t matches = 0;
        for (const auto& observed : rhs.lvvec) {
            if (expected.ll == observed.ll &&
                exact_saved_lvar_equal(expected, observed)) {
                ++matches;
            }
        }
        if (matches != 1) {
            return false;
        }
    }
    return true;
}

bool restore_lvar_settings_snapshot(
    cfunc_t* cfunc,
    bool had_prior_settings,
    const lvar_uservec_t& settings) noexcept
{
    if (!cfunc) {
        return false;
    }
    try {
        save_user_lvar_settings(cfunc->entry_ea, settings);
        (void)mark_cfunc_dirty(cfunc->entry_ea, false);
        lvar_uservec_t observed;
        const bool has_observed_settings =
            restore_user_lvar_settings(&observed, cfunc->entry_ea);
        return had_prior_settings
            ? has_observed_settings &&
                exact_lvar_settings_equal(settings, observed)
            : !has_observed_settings && observed.empty();
    } catch (...) {
        return false;
    }
}

/// Snapshot the mutable, host-owned lvar cache in addition to the persisted
/// lvar settings. A failed signature synchronization can otherwise leave the
/// current cfunc_t referring to a named type that the enclosing persistence
/// transaction subsequently deletes.
struct LvarMemorySnapshot {
    tinfo_t type;
    int width = 0;
    uint64 divisor = 0;
    bool typed = false;
    bool user_typed = false;
    bool no_pointer = false;
    bool partially_typed = false;
    bool floating = false;
    bool unknown_width = false;

    explicit LvarMemorySnapshot(const lvar_t& var)
        : type(var.type())
        , width(var.width)
        , divisor(var.divisor)
        , typed(var.typed())
        , user_typed(var.has_user_type())
        , no_pointer(var.is_noptr_var())
        , partially_typed(var.is_partialy_typed())
        , floating(var.is_floating_var())
        , unknown_width(var.is_unknown_width()) {}

    [[nodiscard]] bool restore(lvar_t& var) const noexcept {
        try {
            var.tif = type;
            var.width = width;
            var.divisor = divisor;

            user_typed ? var.set_user_type() : var.clr_user_type();
            typed ? var.set_typed() : var.set_non_typed();
            no_pointer ? var.set_noptr_var() : var.clr_noptr_var();
            partially_typed ? var.set_partialy_typed() : var.clr_partialy_typed();
            floating ? var.set_floating_var() : var.clr_floating_var();
            unknown_width ? var.set_unknown_width() : var.clr_unknown_width();
            return var.type().equals_to(type) &&
                   var.width == width &&
                   var.divisor == divisor &&
                   var.typed() == typed &&
                   var.has_user_type() == user_typed &&
                   var.is_noptr_var() == no_pointer &&
                   var.is_partialy_typed() == partially_typed &&
                   var.is_floating_var() == floating &&
                   var.is_unknown_width() == unknown_width;
        } catch (...) {
            return false;
        }
    }
};

std::size_t effective_member_size(const udm_t& member) {
    if (!member.type.empty()) {
        const std::size_t type_size = member.type.get_size();
        if (type_size != BADSIZE && type_size > 0) {
            return type_size;
        }
    }

    return static_cast<std::size_t>(member.size / 8);
}

tinfo_t make_scalar_or_bytes_type(std::size_t size) {
    tinfo_t type;
    switch (size) {
        case 1:
            type.create_simple_type(BT_INT8 | BTMT_USIGNED);
            break;
        case 2:
            type.create_simple_type(BT_INT16 | BTMT_USIGNED);
            break;
        case 4:
            type.create_simple_type(BT_INT32 | BTMT_USIGNED);
            break;
        case 8:
            type.create_simple_type(BT_INT64 | BTMT_USIGNED);
            break;
        default: {
            tinfo_t byte_type;
            byte_type.create_simple_type(BT_INT8 | BTMT_USIGNED);
            type.create_array(byte_type, size);
            break;
        }
    }

    return type;
}

qstring rebase_window_member_name(const qstring& name, sval_t offset) {
    return rebase_textual_generated_name(name, offset);
}

bool build_shifted_tail_udt(const tinfo_t& parent_type,
                            sval_t delta,
                            udt_type_data_t& out_udt) {
    if (delta < 0) {
        return false;
    }

    tinfo_t parent = parent_type;
    if (parent.is_ptr()) {
        parent = parent.get_pointed_object();
    }

    udt_type_data_t udt;
    if (!parent.get_udt_details(&udt) || udt.empty()) {
        return false;
    }

    const std::size_t parent_size = parent.get_size();
    if (parent_size == BADSIZE ||
        parent_size > static_cast<std::size_t>(
            std::numeric_limits<sval_t>::max()) ||
        delta >= static_cast<sval_t>(parent_size)) {
        return false;
    }

    out_udt.is_union = false;
    out_udt.total_size = parent_size - static_cast<std::size_t>(delta);
    out_udt.pack = 1;
    out_udt.sda = 1;

    bool added = false;
    for (const auto& member : udt) {
        if ((member.offset % 8) != 0 ||
            member.offset / 8 > static_cast<std::uint64_t>(
                std::numeric_limits<sval_t>::max())) {
            return false;
        }
        const sval_t member_offset = static_cast<sval_t>(member.offset / 8);
        const std::size_t member_size = effective_member_size(member);
        if (member_size == 0) {
            continue;
        }
        if (member_size > static_cast<std::size_t>(
                std::numeric_limits<sval_t>::max())) {
            return false;
        }

        const auto member_end = checked_sval_add(
            member_offset, static_cast<sval_t>(member_size));
        if (!member_end.has_value()) {
            return false;
        }
        if (*member_end <= delta) {
            continue;
        }

        if (member_offset >= delta) {
            udm_t shifted = member;
            const auto rebased = checked_sval_sub(member_offset, delta);
            if (!rebased.has_value() || *rebased < 0 ||
                static_cast<std::uint64_t>(*rebased) >
                    std::numeric_limits<std::uint64_t>::max() / 8) {
                return false;
            }
            const sval_t rebased_offset = *rebased;
            shifted.offset = static_cast<uint64>(rebased_offset) * 8;
            shifted.name = rebase_window_member_name(member.name, rebased_offset);
            out_udt.push_back(std::move(shifted));
            added = true;
            continue;
        }

        const std::size_t inner_delta = static_cast<std::size_t>(delta - member_offset);
        if (!member.type.empty() && member.type.is_struct()) {
            udt_type_data_t nested_tail;
            if (build_shifted_tail_udt(member.type, static_cast<sval_t>(inner_delta), nested_tail)) {
                for (auto nested : nested_tail) {
                    nested.name = rebase_window_member_name(
                        nested.name, static_cast<sval_t>(nested.offset / 8));
                    out_udt.push_back(std::move(nested));
                }
                added = true;
                continue;
            }
        }

        const auto fragment_span = checked_sval_sub(*member_end, delta);
        if (!fragment_span.has_value() || *fragment_span <= 0 ||
            static_cast<std::uint64_t>(*fragment_span) >
                std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        const std::size_t fragment_size =
            static_cast<std::size_t>(*fragment_span);
        udm_t fragment;
        fragment.name = generate_field_name(0, SemanticType::Unknown, static_cast<std::uint32_t>(fragment_size));
        fragment.offset = 0;
        fragment.type = make_scalar_or_bytes_type(fragment_size);
        fragment.size = fragment_size * 8;
        out_udt.push_back(std::move(fragment));
        added = true;
    }

    if (!added) {
        return false;
    }

    std::sort(out_udt.begin(), out_udt.end(), [](const udm_t& a, const udm_t& b) {
        if (a.offset != b.offset) {
            return a.offset < b.offset;
        }
        return a.name < b.name;
    });

    return true;
}

bool build_shifted_object_type(const tinfo_t& parent_type,
                               const qstring& parent_name,
                               sval_t delta,
                               StructurePersistence* persistence,
                               tinfo_t& out_type) {
    udt_type_data_t tail_udt;
    if (!build_shifted_tail_udt(parent_type, delta, tail_udt)) {
        return false;
    }

    tinfo_t tail_type;
    if (!tail_type.create_udt(tail_udt)) {
        return false;
    }

    std::uint32_t tail_alignment = 0;
    const std::size_t tail_size = tail_type.get_size(&tail_alignment);
    udt_type_data_t observed_tail;
    if (tail_size == BADSIZE || tail_size != tail_udt.total_size ||
        tail_alignment != 1 ||
        !tail_type.get_udt_details(&observed_tail) ||
        observed_tail.size() != tail_udt.size()) {
        return false;
    }
    for (std::size_t i = 0; i < tail_udt.size(); ++i) {
        const auto& expected = tail_udt[i];
        const auto& observed = observed_tail[i];
        if (observed.offset != expected.offset ||
            observed.size != expected.size ||
            observed.name != expected.name ||
            observed.type.empty() != expected.type.empty() ||
            (!observed.type.empty() &&
             !observed.type.equals_to(expected.type))) {
            return false;
        }
    }

    if (!parent_name.empty() && persistence != nullptr &&
        persistence->transaction_active()) {
        qstring tail_name = make_shifted_tail_type_name(parent_name, delta);
        tinfo_t named_type;
        if (persistence->stage_auxiliary_named_type(
                tail_name, tail_type, named_type)) {
            out_type = named_type;
            return true;
        }
    }

    out_type = tail_type;
    return true;
}

bool make_shifted_window_ptr(
    const tinfo_t& type,
    StructurePersistence* persistence,
    tinfo_t& out_ptr_type) {
    qstring type_name;
    type.get_type_name(&type_name);
    if (type_name.empty()) {
        return false;
    }

    const std::optional<sval_t> parsed_delta = extract_shifted_view_delta(type_name);
    if (!parsed_delta.has_value() || *parsed_delta <= 0 ||
        *parsed_delta > std::numeric_limits<std::int32_t>::max()) {
        return false;
    }

    const sval_t delta = *parsed_delta;

    tinfo_t object_type;
    tinfo_t exact_object_type;
    bool exact_is_aggregate = false;
    udt_type_data_t udt;
    if (type.get_udt_details(&udt)) {
        for (const auto& member : udt) {
            if (member.offset == static_cast<uint64>(delta) * 8 && !member.type.empty()) {
                exact_object_type = member.type;
                exact_is_aggregate = member.type.is_struct() ||
                                     member.type.is_union() ||
                                     member.type.is_array();
                break;
            }
        }
    }

    if (exact_is_aggregate) {
        object_type = exact_object_type;
    } else {
        build_shifted_object_type(
            type, type_name, static_cast<sval_t>(delta), persistence,
            object_type);
        if (object_type.empty()) {
            object_type = exact_object_type;
        }
    }

    if (object_type.empty()) {
        object_type.create_simple_type(BT_INT8 | BTMT_USIGNED);
    }

    ptr_type_data_t pi(
        tinfo_t(), static_cast<std::size_t>(get_ptr_size()), type,
        static_cast<int32>(delta));
    pi.obj_type = object_type;
    pi.taptr_bits |= TAPTR_SHIFTED;
    return out_ptr_type.create_ptr(pi);
}

bool resolve_member_type_from_parent(const tinfo_t& parent_type,
                                     sval_t member_offset,
                                     bool by_ref,
                                     tinfo_t& out_type) {
    if (member_offset < 0) {
        return false;
    }

    tinfo_t parent = parent_type;
    if (parent.is_ptr()) {
        parent = parent.get_pointed_object();
    }

    udt_type_data_t udt;
    if (!parent.get_udt_details(&udt)) {
        return false;
    }

    for (const auto& member : udt) {
        if (member.offset == static_cast<uint64>(member_offset) * 8 && !member.type.empty()) {
            out_type = member.type;
            if (by_ref && !out_type.is_ptr()) {
                tinfo_t ptr_type;
                ptr_type.create_ptr(out_type);
                out_type = ptr_type;
            }
            return true;
        }
    }

    return false;
}

bool types_equal(const tinfo_t& a, const tinfo_t& b) {
    if (a.empty() || b.empty()) {
        return a.empty() && b.empty();
    }
    return a.equals_to(b);
}

bool apply_global_object_type(ea_t global_ea, const tinfo_t& type) {
    if (global_ea == BADADDR || type.empty()) {
        return false;
    }

    return apply_global_tinfo(global_ea, type);
}

bool derive_caller_type_from_shifted_view(const tinfo_t& shifted_type,
                                          sval_t member_offset,
                                          tinfo_t& out_type) {
    if (member_offset < 0) {
        return false;
    }

    qstring type_name;
    shifted_type.get_type_name(&type_name);
    const std::optional<sval_t> view_delta = extract_shifted_view_delta(type_name);
    if (!view_delta.has_value() || *view_delta <= 0 || *view_delta != member_offset) {
        return false;
    }

    if (shifted_type.is_ptr()) {
        out_type = shifted_type;
        return true;
    }

    tinfo_t ptr_type;
    if (!ptr_type.create_ptr(shifted_type)) {
        return false;
    }

    out_type = ptr_type;
    return true;
}

bool derive_return_source_type(const tinfo_t& returned_type,
                               sval_t return_delta,
                               tinfo_t& out_type) {
    if (return_delta <= 0) {
        return false;
    }

    return derive_caller_type_from_shifted_view(returned_type, return_delta, out_type);
}

tinfo_t derive_callee_type(const tinfo_t& parent_type,
                           sval_t member_offset,
                           bool by_ref,
                           const tinfo_t& passed_type) {
    tinfo_t callee_type = parent_type;
    if (resolve_member_type_from_parent(parent_type, member_offset, by_ref, callee_type)) {
        return callee_type;
    }
    if (!passed_type.empty()) {
        return passed_type;
    }
    if (by_ref) {
        tinfo_t ptr_type;
        ptr_type.create_ptr(parent_type);
        return ptr_type;
    }
    return callee_type;
}

bool synthesize_callee_param_type(ea_t func_ea,
                                  int var_idx,
                                  const SynthOptions& opts,
                                  StructurePersistence& persistence,
                                  tinfo_t& out_type) {
    AccessCollector collector(opts);
    AccessPattern pattern = collector.collect(func_ea, var_idx);
    if (pattern.accesses.empty() || static_cast<int>(pattern.access_count()) < opts.min_accesses) {
        return false;
    }

    LayoutSynthesizer synthesizer(opts);
    SynthesisResult synth_result = synthesizer.synthesize(pattern, opts);
    if (!synth_result.success()) {
        return false;
    }

    tid_t tid = synth_result.sub_structs.empty()
        ? persistence.create_struct(synth_result.structure)
        : persistence.create_struct_with_substructs(synth_result.structure, synth_result.sub_structs);
    if (tid == BADADDR) {
        return false;
    }

    return out_type.get_type_by_tid(tid);
}

} // namespace

cfuncptr_t TypePropagator::get_cfunc(ea_t func_ea) {
    if (shared_cfunc_cache_ != nullptr) {
        auto shared_it = shared_cfunc_cache_->find(func_ea);
        if (shared_it != shared_cfunc_cache_->end()) {
            return shared_it->second;
        }
    }

    auto local_it = local_cfunc_cache_.find(func_ea);
    if (local_it != local_cfunc_cache_.end()) {
        return local_it->second;
    }

    cfuncptr_t cfunc = utils::get_cfunc(func_ea);
    local_cfunc_cache_.emplace(func_ea, cfunc);
    if (shared_cfunc_cache_ != nullptr) {
        shared_cfunc_cache_->emplace(func_ea, cfunc);
    }
    return cfunc;
}

std::optional<int> TypePropagator::resolve_after_application(
    ea_t func_ea,
    const lvar_locator_t& locator)
{
    local_cfunc_cache_.erase(func_ea);
    if (shared_cfunc_cache_ != nullptr) {
        shared_cfunc_cache_->erase(func_ea);
    }
    cfuncptr_t current = get_cfunc(func_ea);
    if (!current) {
        return std::nullopt;
    }
    lvars_t* lvars = current->get_lvars();
    if (lvars == nullptr || lvars->find(locator) == nullptr) {
        return std::nullopt;
    }
    for (size_t i = 0; i < lvars->size(); ++i) {
        if (static_cast<const lvar_locator_t&>(lvars->at(i)) == locator) {
            return static_cast<int>(i);
        }
    }
    return std::nullopt;
}

PropagationResult TypePropagator::propagate(
    ea_t origin_func,
    int origin_var_idx,
    const tinfo_t& new_type,
    PropagationDirection direction)
{
    PropagationResult result;
    try {
    result.sites.reserve(MAX_FIELDS);
    visited_.clear();
    visited_globals_.clear();

    // Apply to origin first. Downstream mutation is conditional on a durable
    // origin update; otherwise the result would describe a propagation graph
    // whose root does not have the requested type.
    cfuncptr_t cfunc = get_cfunc(origin_func);
    PropagationSite origin_site;
    origin_site.func_ea = origin_func;
    origin_site.var_idx = origin_var_idx;
    origin_site.new_type = new_type;
    origin_site.direction = PropagationDirection::Forward;
    if (!cfunc) {
        origin_site.failure_reason = "Failed to decompile origin function";
        result.add_failure(std::move(origin_site));
        return result;
    }
    lvars_t* origin_lvars = cfunc->get_lvars();
    std::optional<lvar_locator_t> origin_locator;
    if (origin_lvars != nullptr && origin_var_idx >= 0 &&
        static_cast<size_t>(origin_var_idx) < origin_lvars->size()) {
        origin_locator = static_cast<const lvar_locator_t&>(
            origin_lvars->at(static_cast<size_t>(origin_var_idx)));
        origin_site.var_name = origin_lvars->at(origin_var_idx).name;
        origin_site.old_type = origin_lvars->at(origin_var_idx).type();
    }
    if (!origin_locator.has_value()) {
        origin_site.failure_reason = "Origin variable has no stable locator";
        result.add_failure(std::move(origin_site));
        return result;
    }
    visited_.insert(make_visit_key(origin_func, *origin_locator));
    if (!apply_type(cfunc, origin_var_idx, new_type)) {
        origin_site.failure_reason = last_application_rollback_failed_
            ? "Failed to apply origin type and rollback failed"
            : "Failed to apply origin type";
        result.add_failure(std::move(origin_site));
        return result;
    }
    const auto current_origin_idx =
        resolve_after_application(origin_func, *origin_locator);
    origin_site.var_idx = current_origin_idx.value_or(-1);
    result.add_success(std::move(origin_site));
    if (!current_origin_idx.has_value()) {
        result.mark_incomplete(
            "origin variable locator no longer resolves after application");
        return result;
    }

    // Propagate in requested directions
    if (direction == PropagationDirection::Forward || direction == PropagationDirection::Both) {
        if (options_.propagate_to_callees) {
            propagate_forward(
                origin_func, *current_origin_idx, new_type, 0, result);
        }
    }

    if (direction == PropagationDirection::Backward || direction == PropagationDirection::Both) {
        if (options_.propagate_to_callers) {
            propagate_backward(
                origin_func, *current_origin_idx, new_type, 0, result);
        }
    }

    } catch (...) {
        result.mark_incomplete(
            "propagation raised after one or more durable type applications");
    }
    return result;
}

PropagationResult TypePropagator::propagate_local(
    cfunc_t* cfunc,
    int var_idx,
    const tinfo_t& new_type)
{
    PropagationResult result;
    try {
    result.sites.reserve(MAX_FIELDS);

    if (!cfunc) return result;
    const ea_t entry_ea = cfunc->entry_ea;

    // Find all aliases
    qvector<int> aliases;
    find_aliased_vars(cfunc, var_idx, aliases);
    std::optional<lvar_locator_t> source_locator;
    std::vector<std::pair<int, lvar_locator_t>> alias_locators;
    if (lvars_t* lvars = cfunc->get_lvars(); lvars != nullptr) {
        if (var_idx >= 0 && static_cast<size_t>(var_idx) < lvars->size()) {
            source_locator = static_cast<const lvar_locator_t&>(
                lvars->at(static_cast<size_t>(var_idx)));
        }
        alias_locators.reserve(aliases.size());
        for (int alias_idx : aliases) {
            if (alias_idx != var_idx && alias_idx >= 0 &&
                static_cast<size_t>(alias_idx) < lvars->size()) {
                alias_locators.emplace_back(
                    alias_idx,
                    static_cast<const lvar_locator_t&>(
                        lvars->at(static_cast<size_t>(alias_idx))));
            }
        }
    }

    // Apply to the source first; aliases are not mutated if the requested root
    // update fails.
    PropagationSite origin_site;
    origin_site.func_ea = cfunc->entry_ea;
    origin_site.var_idx = var_idx;
    origin_site.new_type = new_type;
    if (lvars_t* lvars = cfunc->get_lvars(); lvars != nullptr &&
        var_idx >= 0 && static_cast<size_t>(var_idx) < lvars->size()) {
        origin_site.var_name = lvars->at(var_idx).name;
        origin_site.old_type = lvars->at(var_idx).type();
    }
    if (!source_locator.has_value()) {
        origin_site.failure_reason = "Source variable has no stable locator";
        result.add_failure(std::move(origin_site));
        return result;
    }
    if (!apply_type(cfunc, var_idx, new_type)) {
        origin_site.failure_reason = last_application_rollback_failed_
            ? "Failed to apply source type and rollback failed"
            : "Failed to apply source type";
        result.add_failure(std::move(origin_site));
        return result;
    }
    const auto current_source_idx =
        resolve_after_application(entry_ea, *source_locator);
    origin_site.var_idx = current_source_idx.value_or(-1);
    result.add_success(std::move(origin_site));

    if (!current_source_idx.has_value()) {
        result.mark_incomplete(
            "source locator no longer resolves after local application");
        return result;
    }

    // Apply to aliases using their pre-mutation locators.
    for (const auto& [original_alias_idx, alias_locator] : alias_locators) {
        cfuncptr_t current_cfunc = get_cfunc(entry_ea);
        const auto alias_idx = resolve_after_application(entry_ea, alias_locator);
        current_cfunc = get_cfunc(entry_ea);
        if (!current_cfunc || !alias_idx.has_value()) {
            result.mark_incomplete("alias locator no longer resolves");
            continue;
        }

        PropagationSite site;
        site.func_ea = entry_ea;
        site.var_idx = *alias_idx;
        site.new_type = new_type;
        if (lvars_t* lvars = current_cfunc->get_lvars(); lvars != nullptr) {
            site.var_name = lvars->at(static_cast<size_t>(*alias_idx)).name;
            site.old_type = lvars->at(static_cast<size_t>(*alias_idx)).type();
        }
        if (!result.can_record_site()) {
            result.mark_incomplete("propagation site limit exceeded");
            break;
        }
        if (apply_type(current_cfunc, *alias_idx, new_type)) {
            const auto current_alias_idx =
                resolve_after_application(entry_ea, alias_locator);
            site.var_idx = current_alias_idx.value_or(-1);
            result.add_success(std::move(site));
            if (!current_alias_idx.has_value()) {
                result.mark_incomplete(
                    "alias locator no longer resolves after application");
            }
        } else {
            site.failure_reason = last_application_rollback_failed_
                ? "Failed to apply alias type and rollback failed"
                : "Failed to apply alias type";
            result.add_failure(std::move(site));
        }
    }

    } catch (...) {
        result.mark_incomplete(
            "local propagation raised after one or more durable type applications");
    }
    return result;
}

bool TypePropagator::apply_type(cfunc_t* cfunc, int var_idx, const tinfo_t& type) {
    return apply_type_impl(cfunc, var_idx, type, true);
}

bool TypePropagator::apply_exact_type(
    cfunc_t* cfunc, int var_idx, const tinfo_t& type) {
    return apply_type_impl(cfunc, var_idx, type, false);
}

bool TypePropagator::apply_type_impl(
    cfunc_t* cfunc,
    int var_idx,
    const tinfo_t& type,
    bool normalize_structure_view) {
    last_application_rollback_failed_ = false;
    if (!cfunc || type.empty()) return false;

    lvar_uservec_t prior_settings;
    bool had_prior_settings = false;
    bool settings_snapshot_captured = false;
    bool lvar_mutation_attempted = false;
    lvar_t* cached_var = nullptr;
    std::optional<LvarMemorySnapshot> memory_before;
    bool cached_lvar_mutated = false;
    StructurePersistence local_persistence(options_);
    std::optional<StructurePersistence::Transaction> local_transaction;

    const auto rollback_lvar = [&]() noexcept {
        // synchronize_function_signature() may already have observed a failed
        // prototype rollback. Preserve that failure state: any local named
        // helper can still be referenced by the surviving prototype/lvar.
        bool restored = !last_application_rollback_failed_;
        if (settings_snapshot_captured && lvar_mutation_attempted) {
            restored = restore_lvar_settings_snapshot(
                cfunc, had_prior_settings, prior_settings) && restored;
        }
        if (cached_lvar_mutated && cached_var != nullptr &&
            memory_before.has_value()) {
            restored = memory_before->restore(*cached_var) && restored;
        }
        if (local_transaction.has_value() && local_transaction->active()) {
            if (restored) {
                restored = local_transaction->rollback();
            } else {
                const bool retained = local_transaction->commit();
                if (!retained) {
                    msg("Structor: CRITICAL: failed to retain a local named "
                        "helper after lvar/prototype rollback failure at 0x%llX\n",
                        static_cast<unsigned long long>(cfunc->entry_ea));
                }
            }
        }
        if (!restored) {
            last_application_rollback_failed_ = true;
            msg("Structor: CRITICAL: failed to restore local-variable state "
                "after type-application failure at 0x%llX\n",
                static_cast<unsigned long long>(cfunc->entry_ea));
        }
        return restored;
    };

    try {
        lvars_t* lvars = cfunc->get_lvars();
        if (!lvars || var_idx < 0 || static_cast<size_t>(var_idx) >= lvars->size()) {
            return false;
        }

        lvar_t& var = lvars->at(var_idx);
        cached_var = &var;
        memory_before.emplace(var);

        tinfo_t applied_type = type;
        if (normalize_structure_view && !applied_type.is_ptr()) {
            StructurePersistence* auxiliary_persistence = persistence_;
            if (auxiliary_persistence == nullptr ||
                !auxiliary_persistence->transaction_active()) {
                local_transaction = local_persistence.begin_transaction();
                if (!local_transaction.has_value()) {
                    return false;
                }
                auxiliary_persistence = &local_persistence;
            }
            const tinfo_t current_type = var.type();
            const bool risky_split_local =
                !var.is_arg_var() &&
                (var.is_split_var() || var.is_overlapped_var() || var.is_mapdst_var());
            const bool keep_aggregate_type =
                !var.is_arg_var() &&
                !current_type.empty() &&
                !current_type.is_ptr() &&
                !current_type.is_funcptr() &&
                (current_type.is_struct() ||
                 current_type.is_union() ||
                 current_type.is_array());

            if (!(keep_aggregate_type || risky_split_local)) {
                if (!make_shifted_window_ptr(
                        type, auxiliary_persistence, applied_type)) {
                    applied_type.create_ptr(type);
                }
                if (auxiliary_persistence->transaction_poisoned()) {
                    (void)rollback_lvar();
                    return false;
                }
            }
        }

        if (!var.is_arg_var() &&
            var.is_used_byref() &&
            applied_type.is_ptr() &&
            (var.is_split_var() || var.is_overlapped_var() || var.type().empty())) {
            (void)rollback_lvar();
            return false;
        }

        lvar_saved_info_t lsi;
        lsi.ll = var;
        lsi.type = applied_type;

        // Snapshot the complete saved-lvar vector. A failed return-signature
        // synchronization must not leave only the local-variable half of the
        // requested type change persisted.
        had_prior_settings = restore_user_lvar_settings(
            &prior_settings, cfunc->entry_ea);
        settings_snapshot_captured = true;
        lvar_mutation_attempted = true;
        if (!modify_user_lvar_info(cfunc->entry_ea, MLI_TYPE, lsi)) {
            (void)rollback_lvar();
            return false;
        }

        // Keep the cached lvar in sync when Hex-Rays accepts the type, but
        // never force an in-memory update that would raise an internal error
        // on already-decompiled special arguments such as C++ 'this'.
        if (var.accepts_type(applied_type, false)) {
            cached_lvar_mutated = var.set_lvar_type(applied_type, true);
        }

        // A non-argument return-register local is not part of the function
        // prototype.  Updating only its saved lvar type leaves call sites with
        // the old scalar return type, even though the returned local now has a
        // recovered pointer type.  Keep the prototype consistent whenever the
        // ctree directly returns this lvar.
        if (synchronize_function_signature(cfunc, var_idx, applied_type)) {
            if (local_transaction.has_value()) {
                if (!local_transaction->commit()) {
                    (void)rollback_lvar();
                    return false;
                }
            }
            return true;
        }

        (void)rollback_lvar();
        return false;
    } catch (const vd_interr_t&) {
        (void)rollback_lvar();
        return false;
    } catch (const vd_failure_t&) {
        (void)rollback_lvar();
        return false;
    } catch (...) {
        (void)rollback_lvar();
        return false;
    }
}

void TypePropagator::propagate_forward(
    ea_t func_ea,
    int var_idx,
    const tinfo_t& type,
    int depth,
    PropagationResult& result)
{
    if (depth >= options_.max_propagation_depth) return;

    cfuncptr_t cfunc = get_cfunc(func_ea);
    if (!cfunc) return;

    // Find all callees where this variable is passed as an argument
    qvector<CalleeArgInfo> callees;
    find_callees_with_arg(cfunc, var_idx, callees);
    // Propagate through return-value assignments: var = callee()
    qvector<std::pair<ea_t, int>> return_sources;
    find_assigned_from(cfunc, var_idx, return_sources);

    // Capture every flow edge before any callee application can recurse back
    // and invalidate this function's decompilation.
    propagate_callee_args(callees, type, depth, result);

    for (const auto& [callee_ea, ret_marker] : return_sources) {
        (void)ret_marker;
        cfuncptr_t callee_cfunc = get_cfunc(callee_ea);
        if (!callee_cfunc) {
            result.mark_incomplete(
                "return-source callee decompilation unavailable");
            continue;
        }

        qvector<std::pair<int, sval_t>> return_vars;
        find_return_sources(callee_cfunc, return_vars);
        std::vector<std::optional<lvar_locator_t>> return_locators;
        return_locators.reserve(return_vars.size());
        lvars_t* initial_return_lvars = callee_cfunc->get_lvars();
        for (const auto& [return_var_idx, _] : return_vars) {
            std::optional<lvar_locator_t> locator;
            if (initial_return_lvars != nullptr && return_var_idx >= 0 &&
                static_cast<size_t>(return_var_idx) <
                    initial_return_lvars->size()) {
                locator = static_cast<const lvar_locator_t&>(
                    initial_return_lvars->at(
                        static_cast<size_t>(return_var_idx)));
            }
            return_locators.push_back(std::move(locator));
        }

        for (size_t return_index = 0;
             return_index < return_vars.size(); ++return_index) {
            const auto& [return_var_idx, return_delta] =
                return_vars[return_index];
            if (!return_locators[return_index].has_value()) {
                result.mark_incomplete(
                    "return-source variable has no stable locator");
                continue;
            }
            const auto resolved_return_idx = resolve_after_application(
                callee_ea, *return_locators[return_index]);
            callee_cfunc = get_cfunc(callee_ea);
            if (!callee_cfunc || !resolved_return_idx.has_value()) {
                result.mark_incomplete(
                    "return-source locator no longer resolves");
                continue;
            }
            tinfo_t return_type = type;
            (void)derive_return_source_type(type, return_delta, return_type);

            PropagationSite site;
            site.func_ea = callee_ea;
            site.var_idx = *resolved_return_idx;
            site.new_type = return_type;
            site.direction = PropagationDirection::Forward;

            lvars_t& callee_lvars = *callee_cfunc->get_lvars();
            std::optional<lvar_locator_t> return_locator;
            if (*resolved_return_idx >= 0 &&
                static_cast<size_t>(*resolved_return_idx) < callee_lvars.size()) {
                return_locator = return_locators[return_index];
                site.var_name = callee_lvars[*resolved_return_idx].name;
                site.old_type = callee_lvars[*resolved_return_idx].type();
            }
            if (!return_locator.has_value()) {
                continue;
            }
            auto key = make_visit_key(callee_ea, *return_locator);
            if (visited_.count(key)) continue;
            visited_.insert(std::move(key));

            if (!result.can_record_site()) {
                result.mark_incomplete("propagation site limit exceeded");
                return;
            }

            if (apply_type(callee_cfunc, *resolved_return_idx, return_type)) {
                const auto current_return_idx = return_locator.has_value()
                    ? resolve_after_application(callee_ea, *return_locator)
                    : std::nullopt;
                site.var_idx = current_return_idx.value_or(-1);
                result.add_success(std::move(site));
                if (!current_return_idx.has_value()) {
                    result.mark_incomplete(
                        "return-source locator no longer resolves after application");
                    continue;
                }
                propagate_forward(
                    callee_ea, *current_return_idx,
                    return_type, depth + 1, result);
            } else {
                site.failure_reason = "Failed to apply type";
                result.add_failure(std::move(site));
            }
        }
    }
}

void TypePropagator::propagate_callee_args(
    const qvector<CalleeArgInfo>& callees,
    const tinfo_t& type,
    int depth,
    PropagationResult& result)
{

    qvector<CalleeGroup> groups;
    for (size_t info_index = 0; info_index < callees.size(); ++info_index) {
        const auto& info = callees[info_index];
        CalleeGroup* group = nullptr;
        for (auto& existing : groups) {
            if (existing.callee_ea == info.callee_ea && existing.param_idx == info.param_idx) {
                group = &existing;
                break;
            }
        }

        if (!group) {
            CalleeGroup created;
            created.callee_ea = info.callee_ea;
            created.param_idx = info.param_idx;
            groups.push_back(std::move(created));
            group = &groups.back();
        }

        group->info_indices.push_back(static_cast<int>(info_index));
    }

    // Capture every target before the first application can dirty/reorder a
    // callee's lvar vector.
    for (auto& group : groups) {
        cfuncptr_t initial_callee = get_cfunc(group.callee_ea);
        const auto initial_lvar_idx = get_lvar_index_for_param(
            initial_callee, group.param_idx);
        lvars_t* initial_lvars =
            initial_callee ? initial_callee->get_lvars() : nullptr;
        if (initial_lvars != nullptr && initial_lvar_idx.has_value()) {
            group.locator = static_cast<const lvar_locator_t&>(
                initial_lvars->at(static_cast<size_t>(*initial_lvar_idx)));
        }
    }

    for (const auto& group : groups) {
        const func_t* callee_func = get_func(group.callee_ea);
        if (callee_func != nullptr &&
            (callee_func->flags & (FUNC_LIB | FUNC_THUNK)) != 0) {
            continue;
        }

        if (!group.locator.has_value()) {
            result.mark_incomplete(
                "callee parameter has no stable locator");
            continue;
        }
        auto key = make_visit_key(group.callee_ea, *group.locator);
        if (visited_.count(key)) continue;
        visited_.insert(std::move(key));

        const auto resolved_param_idx =
            resolve_after_application(group.callee_ea, *group.locator);
        cfuncptr_t callee_cfunc = get_cfunc(group.callee_ea);
        if (!callee_cfunc || !resolved_param_idx.has_value()) {
            result.mark_incomplete(
                "callee parameter locator no longer resolves");
            continue;
        }

        qvector<tinfo_t> candidate_types;
        for (int info_index : group.info_indices) {
            const auto& info = callees[static_cast<size_t>(info_index)];
            tinfo_t candidate = derive_callee_type(type,
                                                   info.member_offset,
                                                   info.by_ref,
                                                   info.passed_type);
            bool duplicate = false;
            for (const auto& existing : candidate_types) {
                if (types_equal(existing, candidate)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                candidate_types.push_back(std::move(candidate));
            }
        }
        tinfo_t callee_type;
        StructurePersistence synthesized_type_persistence(options_);
        std::optional<StructurePersistence::Transaction>
            synthesized_type_transaction;
        if (candidate_types.size() == 1) {
            callee_type = candidate_types.front();
        } else if (candidate_types.size() > 1) {
            synthesized_type_transaction =
                synthesized_type_persistence.begin_transaction();
            if (!synthesized_type_transaction.has_value() ||
                !synthesize_callee_param_type(
                    group.callee_ea, *resolved_param_idx, options_,
                    synthesized_type_persistence, callee_type)) {
                if (synthesized_type_transaction.has_value()) {
                    (void)synthesized_type_transaction->rollback();
                    synthesized_type_transaction.reset();
                }
                callee_type = candidate_types.front();
            }
        } else {
            callee_type = type;
        }

        PropagationSite site;
        site.func_ea = group.callee_ea;
        site.var_idx = *resolved_param_idx;
        site.new_type = callee_type;
        site.direction = PropagationDirection::Forward;

        lvars_t& callee_lvars = *callee_cfunc->get_lvars();
        std::optional<lvar_locator_t> callee_locator;
        if (*resolved_param_idx >= 0 &&
            static_cast<size_t>(*resolved_param_idx) < callee_lvars.size()) {
            callee_locator = group.locator;
            site.var_name = callee_lvars[*resolved_param_idx].name;
            site.old_type = callee_lvars[*resolved_param_idx].type();
        }

        if (!result.can_record_site()) {
            result.mark_incomplete("propagation site limit exceeded");
            return;
        }

        if (apply_type(callee_cfunc, *resolved_param_idx, callee_type)) {
            std::optional<int> current_param_idx;
            cfuncptr_t current_callee = callee_cfunc;
            try {
                current_param_idx = resolve_after_application(
                    group.callee_ea, *callee_locator);
                current_callee = get_cfunc(group.callee_ea);
            } catch (...) {
                bool retained = true;
                if (synthesized_type_transaction.has_value()) {
                    retained = synthesized_type_transaction->commit();
                }
                if (retained) {
                    site.var_idx = -1;
                    result.add_success(std::move(site));
                } else {
                    site.failure_reason =
                        "Callee re-resolution raised and synthesized-type "
                        "retention failed";
                    result.add_failure(std::move(site));
                }
                result.mark_incomplete(
                    "callee re-resolution raised after durable application");
                continue;
            }
            if (synthesized_type_transaction.has_value()) {
                if (!synthesized_type_transaction->commit()) {
                    const bool restored = current_callee &&
                        current_param_idx.has_value() &&
                        !site.old_type.empty() &&
                        apply_exact_type(current_callee, *current_param_idx,
                                         site.old_type);
                    site.failure_reason = restored
                        ? "Synthesized-type commit failed; prior type restored"
                        : "Synthesized-type commit failed and prior type could not be restored";
                    result.add_failure(std::move(site));
                    continue;
                }
            }
            site.var_idx = current_param_idx.value_or(-1);
            result.add_success(std::move(site));
            if (!current_param_idx.has_value()) {
                result.mark_incomplete(
                    "callee parameter locator no longer resolves after application");
                continue;
            }

            // Continue propagation
            propagate_forward(
                group.callee_ea, *current_param_idx,
                callee_type, depth + 1, result);

            // A compiler may reuse a pointer argument as the callee's return
            // register.  Follow that value into caller assignment lvars so a
            // recovered type can continue through call-result chains.
            propagate_return_to_callers(
                group.callee_ea,
                *current_param_idx,
                callee_type,
                depth + 1,
                result);
        } else {
            bool rollback_succeeded = true;
            if (synthesized_type_transaction.has_value()) {
                if (last_application_rollback_failed_) {
                    rollback_succeeded =
                        synthesized_type_transaction->commit();
                    site.failure_reason = rollback_succeeded
                        ? "Failed to apply type and lvar rollback failed; "
                          "synthesized type retained"
                        : "Failed to apply type, lvar rollback failed, and "
                          "synthesized-type retention failed";
                } else {
                    rollback_succeeded =
                        synthesized_type_transaction->rollback();
                }
            }
            if (site.failure_reason.empty()) {
                site.failure_reason = rollback_succeeded
                    ? "Failed to apply type"
                    : "Failed to apply type and synthesized-type rollback failed";
            }
            result.add_failure(std::move(site));
        }
    }
}

void TypePropagator::propagate_backward(
    ea_t func_ea,
    int var_idx,
    const tinfo_t& type,
    int depth,
    PropagationResult& result)
{
    if (depth >= options_.max_propagation_depth) return;

    cfuncptr_t cfunc = get_cfunc(func_ea);
    if (!cfunc) return;

    // Check if this is a parameter
    if (!is_parameter(cfunc, var_idx)) return;

    int param_idx = get_param_index(cfunc, var_idx);
    if (param_idx < 0) return;

    propagate_return_to_callers(func_ea, var_idx, type, depth, result);

    // Find all callers that pass to this parameter
    qvector<CallerArgInfo> callers;
    if (!find_callers_with_param(func_ea, param_idx, callers)) {
        result.mark_incomplete(
            "one or more known callers could not be decompiled");
    }

    std::vector<std::optional<lvar_locator_t>> caller_locators;
    caller_locators.reserve(callers.size());
    for (const auto& info : callers) {
        std::optional<lvar_locator_t> locator;
        if (info.var_idx >= 0) {
            cfuncptr_t initial_caller = get_cfunc(info.caller_ea);
            lvars_t* initial_lvars = initial_caller
                ? initial_caller->get_lvars()
                : nullptr;
            if (initial_lvars != nullptr &&
                static_cast<size_t>(info.var_idx) < initial_lvars->size()) {
                locator = static_cast<const lvar_locator_t&>(
                    initial_lvars->at(static_cast<size_t>(info.var_idx)));
            }
        }
        caller_locators.push_back(std::move(locator));
    }

    for (size_t caller_index = 0; caller_index < callers.size(); ++caller_index) {
        const auto& info = callers[caller_index];
        std::optional<int> resolved_caller_idx;
        if (info.var_idx < 0 && info.global_ea != BADADDR) {
            if (!visited_globals_.insert(info.global_ea).second) {
                continue;
            }
        } else {
            if (!caller_locators[caller_index].has_value()) {
                result.mark_incomplete(
                    "caller variable has no stable locator");
                continue;
            }
            auto key = make_visit_key(
                info.caller_ea, *caller_locators[caller_index]);
            if (visited_.count(key)) continue;
            visited_.insert(std::move(key));
            resolved_caller_idx = resolve_after_application(
                info.caller_ea, *caller_locators[caller_index]);
            if (!resolved_caller_idx.has_value()) {
                result.mark_incomplete("caller locator no longer resolves");
                continue;
            }
        }

        cfuncptr_t caller_cfunc = get_cfunc(info.caller_ea);
        if (!caller_cfunc) {
            result.mark_incomplete("caller decompilation unavailable");
            continue;
        }

        tinfo_t caller_type = type;
        const bool derived_shifted_caller_type =
            derive_caller_type_from_shifted_view(type, info.member_offset, caller_type);
        if (!derived_shifted_caller_type) {
            if (info.by_ref && type.is_ptr()) {
                tinfo_t deref = type.get_pointed_object();
                if (!deref.empty()) {
                    caller_type = deref;
                }
            }
        }

        PropagationSite site;
        site.func_ea = info.caller_ea;
        site.var_idx = resolved_caller_idx.value_or(-1);
        site.new_type = caller_type;
        site.direction = PropagationDirection::Backward;

        if (info.var_idx < 0 && info.global_ea != BADADDR) {
            tinfo_t global_type = caller_type;
            if (global_type.is_ptr()) {
                tinfo_t pointed = global_type.get_pointed_object();
                if (!pointed.empty()) {
                    global_type = pointed;
                }
            }

            site.new_type = global_type;
            get_name(&site.var_name, info.global_ea);
            if (!result.can_record_site()) {
                result.mark_incomplete("propagation site limit exceeded");
                return;
            }
            if (apply_global_object_type(info.global_ea, global_type)) {
                result.add_success(std::move(site));
                propagate_global_forward(
                    caller_cfunc, info.global_ea, global_type, depth + 1, result);
            } else {
                site.failure_reason = "Failed to apply global object type";
                result.add_failure(std::move(site));
            }
            continue;
        }

        lvars_t& caller_lvars = *caller_cfunc->get_lvars();
        std::optional<lvar_locator_t> caller_locator;
        if (resolved_caller_idx.has_value() &&
            static_cast<size_t>(*resolved_caller_idx) < caller_lvars.size()) {
            if (derived_shifted_caller_type &&
                !caller_lvars[*resolved_caller_idx].is_arg_var()) {
                continue;
            }
            caller_locator = caller_locators[caller_index];
            site.var_name = caller_lvars[*resolved_caller_idx].name;
            site.old_type = caller_lvars[*resolved_caller_idx].type();
        }

        if (!result.can_record_site()) {
            result.mark_incomplete("propagation site limit exceeded");
            return;
        }

        if (apply_type(caller_cfunc, *resolved_caller_idx, caller_type)) {
            const auto current_caller_idx = caller_locator.has_value()
                ? resolve_after_application(info.caller_ea, *caller_locator)
                : std::nullopt;
            site.var_idx = current_caller_idx.value_or(-1);
            result.add_success(std::move(site));
            if (!current_caller_idx.has_value()) {
                result.mark_incomplete(
                    "caller locator no longer resolves after application");
                continue;
            }

            // Continue backward propagation
            propagate_backward(
                info.caller_ea, *current_caller_idx,
                caller_type, depth + 1, result);

            // IMPORTANT: Also propagate forward from the caller to reach sibling callees
            // This ensures that if main() calls both init_simple() and process_simple()
            // with the same struct, process_simple() also gets the type
            propagate_forward(
                info.caller_ea, *current_caller_idx,
                caller_type, depth + 1, result);
        } else {
            site.failure_reason = "Failed to apply type";
            result.add_failure(std::move(site));
        }
    }
}

void TypePropagator::propagate_return_to_callers(
    ea_t func_ea,
    int return_var_idx,
    const tinfo_t& type,
    int depth,
    PropagationResult& result)
{
    if (depth >= options_.max_propagation_depth) return;

    cfuncptr_t cfunc = get_cfunc(func_ea);
    if (!cfunc) return;

    // Propagate through return-value assignments: caller_var = func()
    qvector<std::pair<int, sval_t>> return_vars;
    find_return_sources(cfunc, return_vars);
    for (const auto& [source_var_idx, return_delta] : return_vars) {
        // A nonzero delta denotes a shifted return view.  Treating it as the
        // root pointer would be unsound; shifted-return synthesis is handled
        // separately from this direct-value bridge.
        if (source_var_idx != return_var_idx || return_delta != 0) continue;

        qvector<std::pair<ea_t, int>> callers;
        if (!find_callers_with_return(func_ea, callers)) {
            result.mark_incomplete(
                "one or more return callers could not be decompiled");
        }

        std::vector<std::optional<lvar_locator_t>> caller_locators;
        caller_locators.reserve(callers.size());
        for (const auto& [caller_ea, caller_var_idx] : callers) {
            std::optional<lvar_locator_t> locator;
            cfuncptr_t initial_caller = get_cfunc(caller_ea);
            lvars_t* initial_lvars = initial_caller
                ? initial_caller->get_lvars()
                : nullptr;
            if (initial_lvars != nullptr && caller_var_idx >= 0 &&
                static_cast<size_t>(caller_var_idx) < initial_lvars->size()) {
                locator = static_cast<const lvar_locator_t&>(
                    initial_lvars->at(static_cast<size_t>(caller_var_idx)));
            }
            caller_locators.push_back(std::move(locator));
        }

        for (size_t caller_index = 0; caller_index < callers.size(); ++caller_index) {
            const auto& [caller_ea, caller_var_idx] = callers[caller_index];
            if (!caller_locators[caller_index].has_value()) {
                result.mark_incomplete(
                    "return caller has no stable locator");
                continue;
            }
            auto key = make_visit_key(
                caller_ea, *caller_locators[caller_index]);
            if (visited_.count(key)) continue;
            visited_.insert(std::move(key));

            const auto resolved_caller_idx = resolve_after_application(
                caller_ea, *caller_locators[caller_index]);
            cfuncptr_t caller_cfunc = get_cfunc(caller_ea);
            if (!caller_cfunc || !resolved_caller_idx.has_value()) {
                result.mark_incomplete(
                    "return caller locator no longer resolves");
                continue;
            }

            PropagationSite site;
            site.func_ea = caller_ea;
            site.var_idx = *resolved_caller_idx;
            site.new_type = type;
            site.direction = PropagationDirection::Backward;

            lvars_t& caller_lvars = *caller_cfunc->get_lvars();
            std::optional<lvar_locator_t> caller_locator;
            if (*resolved_caller_idx >= 0 &&
                static_cast<size_t>(*resolved_caller_idx) < caller_lvars.size()) {
                caller_locator = caller_locators[caller_index];
                site.var_name = caller_lvars[*resolved_caller_idx].name;
                site.old_type = caller_lvars[*resolved_caller_idx].type();
            }

            if (!result.can_record_site()) {
                result.mark_incomplete("propagation site limit exceeded");
                return;
            }

            if (apply_type(caller_cfunc, *resolved_caller_idx, type)) {
                const auto current_caller_idx = caller_locator.has_value()
                    ? resolve_after_application(caller_ea, *caller_locator)
                    : std::nullopt;
                site.var_idx = current_caller_idx.value_or(-1);
                result.add_success(std::move(site));
                if (!current_caller_idx.has_value()) {
                    result.mark_incomplete(
                        "return caller locator no longer resolves after application");
                    continue;
                }

                // Continue through wrappers that return the assigned value,
                // and forward through uses of the caller result.
                propagate_return_to_callers(
                    caller_ea, *current_caller_idx,
                    type, depth + 1, result);
                propagate_forward(
                    caller_ea, *current_caller_idx,
                    type, depth + 1, result);
            } else {
                site.failure_reason = "Failed to apply type";
                result.add_failure(std::move(site));
            }
        }
    }
}

void TypePropagator::find_callees_with_arg(
    cfunc_t* cfunc,
    int var_idx,
    qvector<CalleeArgInfo>& callees)
{
    if (!cfunc) return;

    // Visit all call expressions
    struct CallVisitor : public ctree_visitor_t {
        int target_var_idx;
        qvector<CalleeArgInfo>& results;
        std::unordered_map<int, std::pair<int, sval_t>> aliases;

        CallVisitor(int var_idx, qvector<CalleeArgInfo>& r)
            : ctree_visitor_t(CV_FAST)
            , target_var_idx(var_idx)
            , results(r) {}

        bool resolve_var_delta(const cexpr_t* expr, int& var_idx, sval_t& delta) const {
            if (!expr) {
                return false;
            }

            switch (expr->op) {
                case cot_var: {
                    auto it = aliases.find(expr->v.idx);
                    if (it != aliases.end()) {
                        var_idx = it->second.first;
                        if (!checked_accumulate(delta, it->second.second)) {
                            return false;
                        }
                    } else {
                        var_idx = expr->v.idx;
                    }
                    return true;
                }
                case cot_cast:
                case cot_ref:
                case cot_ptr:
                    return resolve_var_delta(expr->x, var_idx, delta);
                case cot_call:
                    if (expr->x && expr->x->op == cot_helper && expr->a && expr->a->size() == 1) {
                        return resolve_var_delta(&expr->a->at(0), var_idx, delta);
                    }
                    return false;
                case cot_add:
                    if (expr->y && expr->y->op == cot_num && resolve_var_delta(expr->x, var_idx, delta)) {
                        return checked_accumulate_u64(delta, expr->y->numval());
                    }
                    if (expr->x && expr->x->op == cot_num && resolve_var_delta(expr->y, var_idx, delta)) {
                        return checked_accumulate_u64(delta, expr->x->numval());
                    }
                    return false;
                case cot_sub:
                    if (expr->y && expr->y->op == cot_num && resolve_var_delta(expr->x, var_idx, delta)) {
                        return checked_subtract_u64(delta, expr->y->numval());
                    }
                    return false;
                case cot_memptr:
                case cot_memref:
                    if (resolve_var_delta(expr->x, var_idx, delta)) {
                        return checked_accumulate_u64(delta, expr->m);
                    }
                    return false;
                case cot_idx:
                    return resolve_var_delta(expr->x, var_idx, delta);
                default:
                    return false;
            }
        }

        static const cexpr_t* strip_casts_and_refs(const cexpr_t* expr) {
            while (expr && (expr->op == cot_cast || expr->op == cot_ref)) {
                expr = expr->x;
            }
            return expr;
        }

        static const cexpr_t* find_base_var(const cexpr_t* expr) {
            while (expr) {
                if (expr->op == cot_var) return expr;
                if (expr->op == cot_cast || expr->op == cot_ref || expr->op == cot_ptr) {
                    expr = expr->x;
                } else if (expr->op == cot_call && expr->x && expr->x->op == cot_helper && expr->a && expr->a->size() == 1) {
                    expr = &expr->a->at(0);
                } else if (expr->op == cot_add || expr->op == cot_sub) {
                    const cexpr_t* left = find_base_var(expr->x);
                    if (left) return left;
                    expr = expr->y;
                } else if (expr->op == cot_memref || expr->op == cot_memptr || expr->op == cot_idx) {
                    expr = expr->x;
                } else {
                    break;
                }
            }
            return nullptr;
        }

        static bool is_plain_var_arg(const cexpr_t* expr, int var_idx) {
            expr = strip_casts_and_refs(expr);
            return expr && expr->op == cot_var && expr->v.idx == var_idx;
        }

        tinfo_t extract_member_arg_type(const cexpr_t* expr, int var_idx) const {
            const bool by_ref = contains_ref(expr);
            expr = strip_casts_and_refs(expr);
            if (!expr) {
                return tinfo_t();
            }

            if ((expr->op == cot_memptr || expr->op == cot_memref) && expr->x) {
                int base_var = -1;
                sval_t delta = 0;
                if (resolve_var_delta(expr->x, base_var, delta) && base_var == var_idx && !expr->type.empty()) {
                    tinfo_t member_type = expr->type;
                    if (by_ref) {
                        tinfo_t ptr_type;
                        ptr_type.create_ptr(member_type);
                        return ptr_type;
                    }
                    return member_type;
                }
            }

            return tinfo_t();
        }

        sval_t extract_member_offset(const cexpr_t* expr, int var_idx) const {
            expr = strip_casts_and_refs(expr);
            if (!expr) {
                return -1;
            }

            if ((expr->op == cot_memptr || expr->op == cot_memref) && expr->x) {
                int base_var = -1;
                sval_t delta = 0;
                if (resolve_var_delta(expr->x, base_var, delta) && base_var == var_idx) {
                    return checked_accumulate_u64(delta, expr->m) ? delta : -1;
                }
            }

            return -1;
        }

        static bool contains_ref(const cexpr_t* expr) {
            if (!expr) return false;
            if (expr->op == cot_ref) return true;

            switch (expr->op) {
                case cot_cast:
                case cot_ptr:
                case cot_memref:
                case cot_memptr:
                case cot_idx:
                    return contains_ref(expr->x);
                case cot_call:
                    if (expr->x && expr->x->op == cot_helper && expr->a && expr->a->size() == 1) {
                        return contains_ref(&expr->a->at(0));
                    }
                    return false;
                case cot_add:
                case cot_sub:
                    return contains_ref(expr->x) || contains_ref(expr->y);
                default:
                    return false;
            }
        }

        int idaapi visit_expr(cexpr_t* expr) override {
            if (!expr) return 0;

            if (expr->op == cot_asg && expr->x && expr->x->op == cot_var && expr->y) {
                int base_var = -1;
                sval_t delta = 0;
                if (resolve_var_delta(expr->y, base_var, delta)) {
                    aliases[expr->x->v.idx] = {base_var, delta};
                }
                return 0;
            }

            if (expr->op != cot_call || !expr->a) return 0;

            // Check if target is a direct call
            ea_t callee_ea = BADADDR;
            if (expr->x->op == cot_obj) {
                callee_ea = expr->x->obj_ea;
            } else if (expr->x->op == cot_helper) {
                // Helper function - skip
                return 0;
            }

            if (callee_ea == BADADDR) return 0;

            // Check each argument
            for (size_t i = 0; i < expr->a->size(); ++i) {
                const carg_t& arg = expr->a->at(i);
                int base_var = -1;
                sval_t delta = 0;

                const bool resolved = resolve_var_delta(&arg, base_var, delta);
                bool matches_target = resolved && base_var == target_var_idx;
                if (!matches_target) {
                    const cexpr_t* root = find_base_var(&arg);
                    matches_target = root && root->op == cot_var && root->v.idx == target_var_idx;
                }

                if (matches_target) {
                    CalleeArgInfo info;
                    info.callee_ea = callee_ea;
                    info.param_idx = static_cast<int>(i);
                    info.by_ref = contains_ref(&arg);
                    info.member_offset = extract_member_offset(&arg, target_var_idx);
                    info.passed_type = extract_member_arg_type(&arg, target_var_idx);
                    if (info.passed_type.empty() && !is_plain_var_arg(&arg, target_var_idx) && !arg.type.empty()) {
                        info.passed_type = arg.type;
                    }
                    results.push_back(info);
                    break;
                }
            }

            return 0;
        }
    };

    CallVisitor visitor(var_idx, callees);
    visitor.apply_to(&cfunc->body, nullptr);
}

void TypePropagator::propagate_global_forward(
    cfunc_t* cfunc,
    ea_t global_ea,
    const tinfo_t& type,
    int depth,
    PropagationResult& result)
{
    if (!cfunc || global_ea == BADADDR || depth >= options_.max_propagation_depth) {
        return;
    }

    qvector<CalleeArgInfo> callees;
    find_callees_with_global(cfunc, global_ea, callees);
    propagate_callee_args(callees, type, depth, result);
}

void TypePropagator::find_callees_with_global(
    cfunc_t* cfunc,
    ea_t global_ea,
    qvector<CalleeArgInfo>& callees)
{
    if (!cfunc || global_ea == BADADDR) {
        return;
    }

    struct GlobalCallVisitor : public ctree_visitor_t {
        ea_t target_global;
        qvector<CalleeArgInfo>& results;

        GlobalCallVisitor(ea_t global, qvector<CalleeArgInfo>& r)
            : ctree_visitor_t(CV_FAST)
            , target_global(global)
            , results(r) {}

        static bool contains_ref(const cexpr_t* expr) {
            if (!expr) return false;
            if (expr->op == cot_ref) return true;

            switch (expr->op) {
                case cot_cast:
                case cot_ptr:
                case cot_memref:
                case cot_memptr:
                case cot_idx:
                    return contains_ref(expr->x);
                case cot_add:
                case cot_sub:
                    return contains_ref(expr->x) || contains_ref(expr->y);
                default:
                    return false;
            }
        }

        static bool resolve_global_delta(
            const cexpr_t* expr,
            ea_t& resolved_global,
            sval_t& delta)
        {
            if (!expr) {
                return false;
            }

            switch (expr->op) {
                case cot_obj:
                    resolved_global = expr->obj_ea;
                    return resolved_global != BADADDR;
                case cot_cast:
                case cot_ref:
                case cot_ptr:
                    return resolve_global_delta(expr->x, resolved_global, delta);
                case cot_add:
                    if (expr->y && expr->y->op == cot_num &&
                        resolve_global_delta(expr->x, resolved_global, delta)) {
                        return checked_accumulate_u64(delta, expr->y->numval());
                    }
                    if (expr->x && expr->x->op == cot_num &&
                        resolve_global_delta(expr->y, resolved_global, delta)) {
                        return checked_accumulate_u64(delta, expr->x->numval());
                    }
                    return false;
                case cot_sub:
                    if (expr->y && expr->y->op == cot_num &&
                        resolve_global_delta(expr->x, resolved_global, delta)) {
                        return checked_subtract_u64(delta, expr->y->numval());
                    }
                    return false;
                case cot_memptr:
                case cot_memref:
                    if (resolve_global_delta(expr->x, resolved_global, delta)) {
                        return checked_accumulate_u64(delta, expr->m);
                    }
                    return false;
                case cot_idx:
                    return resolve_global_delta(expr->x, resolved_global, delta);
                default:
                    return false;
            }
        }

        int idaapi visit_expr(cexpr_t* expr) override {
            if (!expr || expr->op != cot_call || !expr->x || !expr->a ||
                expr->x->op != cot_obj) {
                return 0;
            }

            const ea_t callee_ea = expr->x->obj_ea;
            for (size_t i = 0; i < expr->a->size(); ++i) {
                const carg_t& arg = expr->a->at(i);
                ea_t resolved_global = BADADDR;
                sval_t delta = 0;
                if (!resolve_global_delta(&arg, resolved_global, delta) ||
                    resolved_global != target_global) {
                    continue;
                }

                CalleeArgInfo info;
                info.callee_ea = callee_ea;
                info.param_idx = static_cast<int>(i);
                info.by_ref = contains_ref(&arg);
                info.member_offset = delta > 0 ? delta : -1;
                results.push_back(std::move(info));
            }

            return 0;
        }
    };

    GlobalCallVisitor visitor(global_ea, callees);
    visitor.apply_to(&cfunc->body, nullptr);
}

bool TypePropagator::find_callers_with_param(
    ea_t func_ea,
    int param_idx,
    qvector<CallerArgInfo>& callers)
{
    // Get all callers
    qvector<ea_t> caller_funcs = utils::get_callers(func_ea);
    bool complete = true;

    for (ea_t caller_ea : caller_funcs) {
        cfuncptr_t caller_cfunc = get_cfunc(caller_ea);
        if (!caller_cfunc) {
            complete = false;
            continue;
        }

        // Find calls to our function
        struct CallerFinder : public ctree_visitor_t {
            ea_t target_func;
            int target_param;
            qvector<CallerArgInfo>& results;
            ea_t caller_ea;

            CallerFinder(ea_t func, int param, ea_t caller, qvector<CallerArgInfo>& r)
                : ctree_visitor_t(CV_FAST)
                , target_func(func)
                , target_param(param)
                , results(r)
                , caller_ea(caller) {}

            // Helper to extract base variable from complex expressions
            static cexpr_t* find_base_var(cexpr_t* expr) {
                while (expr) {
                    if (expr->op == cot_var || expr->op == cot_obj) return expr;
                    if (expr->op == cot_cast || expr->op == cot_ref || expr->op == cot_ptr) {
                        expr = expr->x;
                    } else if (expr->op == cot_call && expr->x && expr->x->op == cot_helper && expr->a && expr->a->size() == 1) {
                        expr = &expr->a->at(0);
                    } else if (expr->op == cot_add || expr->op == cot_sub) {
                        // Try both sides for (ptr + offset) or (offset + ptr)
                        cexpr_t* left = find_base_var(expr->x);
                        if (left) return left;
                        expr = expr->y;
                    } else if (expr->op == cot_memref || expr->op == cot_memptr) {
                        expr = expr->x;
                    } else if (expr->op == cot_idx) {
                        expr = expr->x;
                    } else {
                        break;
                    }
                }
                return nullptr;
            }

            static bool resolve_global_delta(
                const cexpr_t* expr,
                ea_t& global_ea,
                sval_t& delta)
            {
                if (!expr) {
                    return false;
                }

                switch (expr->op) {
                    case cot_obj:
                        global_ea = expr->obj_ea;
                        return global_ea != BADADDR;
                    case cot_cast:
                    case cot_ref:
                    case cot_ptr:
                        return resolve_global_delta(expr->x, global_ea, delta);
                    case cot_add:
                        if (expr->y && expr->y->op == cot_num &&
                            resolve_global_delta(expr->x, global_ea, delta)) {
                            return checked_accumulate_u64(delta, expr->y->numval());
                        }
                        if (expr->x && expr->x->op == cot_num &&
                            resolve_global_delta(expr->y, global_ea, delta)) {
                            return checked_accumulate_u64(delta, expr->x->numval());
                        }
                        return false;
                    case cot_sub:
                        if (expr->y && expr->y->op == cot_num &&
                            resolve_global_delta(expr->x, global_ea, delta)) {
                            return checked_subtract_u64(delta, expr->y->numval());
                        }
                        return false;
                    case cot_memptr:
                    case cot_memref:
                        if (resolve_global_delta(expr->x, global_ea, delta)) {
                            return checked_accumulate_u64(delta, expr->m);
                        }
                        return false;
                    case cot_idx:
                        return resolve_global_delta(expr->x, global_ea, delta);
                    default:
                        return false;
                }
            }

            static bool contains_ref(const cexpr_t* expr) {
                if (!expr) return false;
                if (expr->op == cot_ref) return true;

                switch (expr->op) {
                    case cot_cast:
                    case cot_ptr:
                    case cot_memref:
                    case cot_memptr:
                    case cot_idx:
                        return contains_ref(expr->x);
                    case cot_add:
                    case cot_sub:
                        return contains_ref(expr->x) || contains_ref(expr->y);
                    default:
                        return false;
                }
            }

            static const cexpr_t* strip_casts_and_refs(const cexpr_t* expr) {
                while (expr && (expr->op == cot_cast || expr->op == cot_ref)) {
                    expr = expr->x;
                }
                return expr;
            }

            bool resolve_var_delta(const cexpr_t* expr, int& var_idx, sval_t& delta) const {
                if (!expr) {
                    return false;
                }

                switch (expr->op) {
                    case cot_var:
                        var_idx = expr->v.idx;
                        return true;
                    case cot_cast:
                    case cot_ref:
                    case cot_ptr:
                        return resolve_var_delta(expr->x, var_idx, delta);
                    case cot_add:
                        if (expr->y && expr->y->op == cot_num && resolve_var_delta(expr->x, var_idx, delta)) {
                            return checked_accumulate_u64(delta, expr->y->numval());
                        }
                        if (expr->x && expr->x->op == cot_num && resolve_var_delta(expr->y, var_idx, delta)) {
                            return checked_accumulate_u64(delta, expr->x->numval());
                        }
                        return false;
                    case cot_sub:
                        if (expr->y && expr->y->op == cot_num && resolve_var_delta(expr->x, var_idx, delta)) {
                            return checked_subtract_u64(delta, expr->y->numval());
                        }
                        return false;
                    case cot_memptr:
                    case cot_memref:
                        if (resolve_var_delta(expr->x, var_idx, delta)) {
                            return checked_accumulate_u64(delta, expr->m);
                        }
                        return false;
                    case cot_idx:
                        return resolve_var_delta(expr->x, var_idx, delta);
                    default:
                        return false;
                }
            }

            sval_t extract_member_offset(const cexpr_t* expr, int var_idx) const {
                expr = strip_casts_and_refs(expr);
                if (!expr) {
                    return -1;
                }

                if (expr->op == cot_memptr || expr->op == cot_memref) {
                    int base_var = -1;
                    sval_t delta = 0;
                    if (resolve_var_delta(expr->x, base_var, delta) && base_var == var_idx) {
                        return checked_accumulate_u64(delta, expr->m) ? delta : -1;
                    }
                }

                int base_var = -1;
                sval_t delta = 0;
                if (resolve_var_delta(expr, base_var, delta) && base_var == var_idx && delta > 0) {
                    return delta;
                }

                return -1;
            }

            int idaapi visit_expr(cexpr_t* expr) override {
                if (expr->op != cot_call || !expr->a) return 0;

                ea_t callee_ea = BADADDR;
                if (expr->x->op == cot_obj) {
                    callee_ea = expr->x->obj_ea;
                }

                if (callee_ea != target_func) return 0;

                // Found a call - check the argument at target_param
                if (static_cast<size_t>(target_param) >= expr->a->size()) return 0;

                const carg_t& arg = expr->a->at(target_param);

                // Use helper to find base variable through complex expressions
                cexpr_t* base_var = find_base_var(const_cast<cexpr_t*>(static_cast<const cexpr_t*>(&arg)));
                if (base_var && base_var->op == cot_var) {
                    CallerArgInfo info;
                    info.caller_ea = caller_ea;
                    info.var_idx = base_var->v.idx;
                    info.by_ref = contains_ref(&arg);
                    info.member_offset = extract_member_offset(&arg, info.var_idx);
                    results.push_back(info);
                } else if (base_var && base_var->op == cot_obj) {
                    CallerArgInfo info;
                    info.caller_ea = caller_ea;
                    info.global_ea = base_var->obj_ea;
                    info.by_ref = contains_ref(&arg);

                    ea_t resolved_global = BADADDR;
                    sval_t delta = 0;
                    if (resolve_global_delta(&arg, resolved_global, delta) &&
                        resolved_global == info.global_ea) {
                        info.member_offset = delta > 0 ? delta : -1;
                    }
                    results.push_back(info);
                }

                return 0;
            }
        };

        CallerFinder finder(func_ea, param_idx, caller_ea, callers);
        finder.apply_to(&caller_cfunc->body, nullptr);
    }
    return complete;
}

void TypePropagator::find_aliased_vars(
    cfunc_t* cfunc,
    int var_idx,
    qvector<int>& aliases)
{
    if (!cfunc) return;

    // Find all assignments of our variable to other variables
    struct AliasVisitor : public ctree_visitor_t {
        int target_var;
        qvector<int>& aliases;

        AliasVisitor(int var, qvector<int>& a)
            : ctree_visitor_t(CV_FAST)
            , target_var(var)
            , aliases(a) {}

        int idaapi visit_expr(cexpr_t* expr) override {
            if (expr->op != cot_asg) return 0;

            // Check if right side is our variable
            cexpr_t* rhs = expr->y;
            while (rhs->op == cot_cast) {
                rhs = rhs->x;
            }

            if (rhs->op == cot_var && rhs->v.idx == target_var) {
                // Check if left side is a variable
                cexpr_t* lhs = expr->x;
                if (lhs->op == cot_var) {
                    aliases.push_back(lhs->v.idx);
                }
            }

            // Also check reverse (our variable assigned from another)
            cexpr_t* lhs = expr->x;
            if (lhs->op == cot_var && lhs->v.idx == target_var) {
                rhs = expr->y;
                while (rhs->op == cot_cast) {
                    rhs = rhs->x;
                }
                if (rhs->op == cot_var) {
                    aliases.push_back(rhs->v.idx);
                }
            }

            return 0;
        }
    };

    AliasVisitor visitor(var_idx, aliases);
    visitor.apply_to(&cfunc->body, nullptr);
}

void TypePropagator::find_assigned_from(
    cfunc_t* cfunc,
    int var_idx,
    qvector<std::pair<ea_t, int>>& sources)
{
    if (!cfunc) return;

    // Find assignments to our variable from call results
    struct SourceVisitor : public ctree_visitor_t {
        int target_var;
        qvector<std::pair<ea_t, int>>& sources;

        SourceVisitor(int var, qvector<std::pair<ea_t, int>>& s)
            : ctree_visitor_t(CV_FAST)
            , target_var(var)
            , sources(s) {}

        int idaapi visit_expr(cexpr_t* expr) override {
            if (expr->op != cot_asg) return 0;

            // Check if left side is our variable
            cexpr_t* lhs = expr->x;
            if (lhs->op != cot_var || lhs->v.idx != target_var) return 0;

            // Check if right side is a call
            cexpr_t* rhs = expr->y;
            while (rhs->op == cot_cast) {
                rhs = rhs->x;
            }

            if (rhs->op == cot_call && rhs->x->op == cot_obj) {
                ea_t callee = rhs->x->obj_ea;
                sources.push_back({callee, -1});  // -1 indicates return value
            }

            return 0;
        }
    };

    SourceVisitor visitor(var_idx, sources);
    visitor.apply_to(&cfunc->body, nullptr);
}

void TypePropagator::find_return_sources(
    cfunc_t* cfunc,
    qvector<std::pair<int, sval_t>>& sources)
{
    if (!cfunc) return;

    struct ReturnVisitor : public ctree_visitor_t {
        qvector<std::pair<int, sval_t>>& sources;

        ReturnVisitor(qvector<std::pair<int, sval_t>>& s)
            : ctree_visitor_t(CV_FAST)
            , sources(s) {}

        int idaapi visit_insn(cinsn_t* insn) override {
            if (!insn || insn->op != cit_return) return 0;
            if (!insn->creturn) return 0;

            cexpr_t* expr = &insn->creturn->expr;
            if (!expr || expr->op == cot_empty) return 0;

            auto info = utils::extract_ptr_arith(expr);
            if (!info.valid || info.var_idx < 0) return 0;

            for (const auto& entry : sources) {
                if (entry.first == info.var_idx && entry.second == info.offset) {
                    return 0;
                }
            }

            sources.push_back({info.var_idx, info.offset});
            return 0;
        }
    };

    ReturnVisitor visitor(sources);
    visitor.apply_to(&cfunc->body, nullptr);
}

bool TypePropagator::has_material_return_consumer(ea_t func_ea)
{
    struct WorkItem {
        ea_t func_ea = BADADDR;
        int depth = 0;
    };

    std::queue<WorkItem> work;
    std::unordered_set<ea_t> visited;
    work.push({func_ea, 0});
    visited.insert(func_ea);

    while (!work.empty()) {
        const WorkItem current = work.front();
        work.pop();
        if (current.depth >= options_.max_propagation_depth) {
            continue;
        }

        qvector<std::pair<ea_t, int>> callers;
        if (!find_callers_with_return(current.func_ea, callers)) {
            return false;
        }
        for (const auto &[caller_ea, caller_var_idx] : callers) {
            cfuncptr_t caller_cfunc = get_cfunc(caller_ea);
            if (!caller_cfunc || caller_var_idx < 0) {
                continue;
            }

            AccessCollector collector(options_);
            const AccessPattern use_pattern =
                    collector.collect(caller_cfunc, caller_var_idx);
            if (!use_pattern.accesses.empty()) {
                return true;
            }

            qvector<CalleeArgInfo> callees;
            find_callees_with_arg(caller_cfunc, caller_var_idx, callees);
            if (!callees.empty()) {
                return true;
            }

            qvector<std::pair<int, sval_t>> return_sources;
            find_return_sources(caller_cfunc, return_sources);
            const bool purely_forwarded = std::any_of(
                    return_sources.begin(), return_sources.end(),
                    [&](const auto &source) {
                        return source.first == caller_var_idx;
                    });
            if (purely_forwarded && visited.insert(caller_ea).second) {
                work.push({caller_ea, current.depth + 1});
            }
        }
    }

    return false;
}

bool TypePropagator::synchronize_function_signature(
    cfunc_t* cfunc,
    int var_idx,
    const tinfo_t& lvar_type)
{
    if (!cfunc || var_idx < 0 || lvar_type.empty()) {
        return false;
    }

    struct ReturnTypeVisitor : public ctree_visitor_t {
        struct ReturnPath {
            int var_idx = -1;
            sval_t offset = 0;
            bool address_of = false;
            bool through_member = false;
            bool pointer_arithmetic = false;
            bool valid = false;
        };

        int target_var_idx;
        const tinfo_t& lvar_type;
        tinfo_t inferred_type;
        bool found_candidate = false;
        bool compatible = true;

        ReturnTypeVisitor(int target, const tinfo_t& type)
            : ctree_visitor_t(CV_FAST)
            , target_var_idx(target)
            , lvar_type(type) {}

        static bool resolve_path(const cexpr_t* expr, ReturnPath& path) {
            if (!expr) {
                return false;
            }

            switch (expr->op) {
                case cot_var:
                    path.var_idx = expr->v.idx;
                    path.valid = true;
                    return true;
                case cot_cast:
                    // Preserve pointer-to-pointer casts used to express byte
                    // arithmetic, but reject integerizing casts such as
                    // (uintptr_t)p.
                    if (!expr->type.is_ptr()) {
                        return false;
                    }
                    return resolve_path(expr->x, path);
                case cot_ref:
                    if (!resolve_path(expr->x, path)) {
                        return false;
                    }
                    path.address_of = true;
                    return true;
                case cot_memref:
                case cot_memptr:
                    if (!resolve_path(expr->x, path)) {
                        return false;
                    }
                    if (!checked_accumulate_u64(path.offset, expr->m)) {
                        return false;
                    }
                    path.through_member = true;
                    return true;
                case cot_add: {
                    const cexpr_t* base = nullptr;
                    const cexpr_t* amount = nullptr;
                    if (expr->y && expr->y->op == cot_num) {
                        base = expr->x;
                        amount = expr->y;
                    } else if (expr->x && expr->x->op == cot_num) {
                        base = expr->y;
                        amount = expr->x;
                    } else {
                        return false;
                    }

                    if (!resolve_path(base, path)) {
                        return false;
                    }

                    const auto raw_delta =
                        checked_sval_from_u64(amount->numval());
                    if (!raw_delta.has_value()) {
                        return false;
                    }
                    sval_t delta = *raw_delta;
                    if (base && base->type.is_ptr()) {
                        const tinfo_t pointed = base->type.get_pointed_object();
                        const size_t element_size = pointed.get_size();
                        if (!pointed.empty() && element_size != BADSIZE && element_size > 0) {
                            const auto scaled = checked_scaled_ctree_constant(
                                amount->numval(), element_size);
                            if (!scaled.has_value()) {
                                return false;
                            }
                            delta = *scaled;
                        }
                    }
                    if (!checked_accumulate(path.offset, delta)) {
                        return false;
                    }
                    path.pointer_arithmetic = true;
                    return true;
                }
                case cot_sub:
                    if (!expr->y || expr->y->op != cot_num ||
                        !resolve_path(expr->x, path)) {
                        return false;
                    } else {
                        const auto raw_delta =
                            checked_sval_from_u64(expr->y->numval());
                        if (!raw_delta.has_value()) {
                            return false;
                        }
                        sval_t delta = *raw_delta;
                        if (expr->x && expr->x->type.is_ptr()) {
                            const tinfo_t pointed = expr->x->type.get_pointed_object();
                            const size_t element_size = pointed.get_size();
                            if (!pointed.empty() && element_size != BADSIZE && element_size > 0) {
                                const auto scaled = checked_scaled_ctree_constant(
                                    expr->y->numval(), element_size);
                                if (!scaled.has_value()) {
                                    return false;
                                }
                                delta = *scaled;
                            }
                        }
                        const auto combined =
                            checked_sval_sub(path.offset, delta);
                        if (!combined.has_value()) {
                            return false;
                        }
                        path.offset = *combined;
                        path.pointer_arithmetic = true;
                        return true;
                    }
                default:
                    return false;
            }
        }

        int idaapi visit_insn(cinsn_t* insn) override {
            if (!insn || insn->op != cit_return || !insn->creturn) {
                return 0;
            }

            const cexpr_t& expr = insn->creturn->expr;
            if (expr.op == cot_empty) {
                compatible = false;
                return 0;
            }
            if (expr.op == cot_num && expr.numval() == 0) {
                return 0;
            }

            ReturnPath path;
            if (!resolve_path(&expr, path) || !path.valid ||
                path.var_idx != target_var_idx || path.offset < 0) {
                compatible = false;
                return 0;
            }

            tinfo_t candidate;
            if (path.offset == 0 && !path.address_of &&
                !path.through_member && !path.pointer_arithmetic) {
                candidate = lvar_type;
            } else if (!(path.address_of && path.offset == 0 && !path.through_member) &&
                       resolve_member_type_from_parent(
                           lvar_type,
                           path.offset,
                           path.address_of || path.pointer_arithmetic,
                           candidate)) {
                // Exact member or pointer-arithmetic target resolved.
            } else {
                compatible = false;
                return 0;
            }

            if (candidate.empty() || !candidate.is_ptr()) {
                compatible = false;
                return 0;
            }

            if (!found_candidate) {
                inferred_type = candidate;
                found_candidate = true;
            } else if (!inferred_type.equals_to(candidate)) {
                compatible = false;
            }
            return 0;
        }
    };

    ReturnTypeVisitor return_visitor(var_idx, lvar_type);
    return_visitor.apply_to(&cfunc->body, nullptr);

    const int param_idx = get_param_index(cfunc, var_idx);
    const bool synchronize_param = param_idx >= 0;
    const bool synchronize_return = return_visitor.compatible &&
                                    return_visitor.found_candidate &&
                                    has_material_return_consumer(cfunc->entry_ea);
    const tinfo_t& desired_return_type = return_visitor.inferred_type;
    if (!synchronize_param && !synchronize_return) {
        return true;
    }

    const ea_t entry_ea = cfunc->entry_ea;
    tinfo_t stored_before;
    bool has_stored_type = false;
    bool prototype_write_attempted = false;
    auto rollback_prototype = [&]() noexcept {
        const bool restored = restore_function_type_snapshot(
            entry_ea, has_stored_type, stored_before);
        if (!restored) {
            last_application_rollback_failed_ = true;
            msg("Structor: CRITICAL: failed to restore function prototype at 0x%llX\n",
                static_cast<unsigned long long>(entry_ea));
        }
        return restored;
    };

    try {
        tinfo_t observed_func_type;
        if (!cfunc->get_func_type(&observed_func_type) ||
            !observed_func_type.is_func()) {
            return false;
        }

        has_stored_type =
            get_tinfo(&stored_before, entry_ea) && stored_before.is_func();

        // Mutate the IDB prototype when one exists so unrelated argument
        // signedness, names, calling convention, and return metadata remain
        // byte-for-byte stable.  The current cfunc type may already contain
        // transient lvar-driven changes in other signature components.
        tinfo_t func_type = has_stored_type ? stored_before : observed_func_type;

        bool modified = false;
        if (synchronize_param) {
            if (param_idx >= func_type.get_nargs()) {
                return false;
            }

            const tinfo_t current_param_type = func_type.get_nth_arg(param_idx);
            if (current_param_type.empty() || !current_param_type.equals_to(lvar_type)) {
                if (func_type.set_funcarg_type(
                        static_cast<size_t>(param_idx), lvar_type) != TERR_OK) {
                    return false;
                }
                modified = true;
            }
        }

        if (synchronize_return) {
            const tinfo_t current_return_type = func_type.get_rettype();
            if (current_return_type.empty() ||
                !current_return_type.equals_to(desired_return_type)) {
                if (func_type.set_func_rettype(desired_return_type) != TERR_OK) {
                    return false;
                }
                modified = true;
            }
        }

        // modify_user_lvar_info() can make cfunc->get_func_type() reflect the
        // new lvar type without persisting the corresponding IDB prototype.
        // Compare against the stored type before deciding no write is needed;
        // direct callers consult that stored prototype when rebuilding calls.
        bool requires_apply = modified;
        if (!has_stored_type) {
            requires_apply = true;
        } else {
            if (synchronize_param) {
                const tinfo_t stored_param = stored_before.get_nth_arg(param_idx);
                requires_apply = requires_apply || stored_param.empty() ||
                                 !stored_param.equals_to(lvar_type);
            }
            if (synchronize_return) {
                const tinfo_t stored_return = stored_before.get_rettype();
                requires_apply = requires_apply || stored_return.empty() ||
                                 !stored_return.equals_to(desired_return_type);
            }
        }

        if (!requires_apply) {
            return true;
        }

        // Recovered prototypes remain guessed so Hex-Rays can refine unrelated
        // arguments (for example, signedness inferred from a recovered field)
        // and pre-existing definitive user prototypes retain precedence.
        prototype_write_attempted = true;
        if (!apply_tinfo(entry_ea, func_type, TINFO_GUESSED | TINFO_STRICT)) {
            rollback_prototype();
            return false;
        }

        tinfo_t stored_type;
        if (!get_tinfo(&stored_type, entry_ea) || !stored_type.is_func()) {
            rollback_prototype();
            return false;
        }
        if (synchronize_param) {
            const tinfo_t stored_param_type = stored_type.get_nth_arg(param_idx);
            if (stored_param_type.empty() || !stored_param_type.equals_to(lvar_type)) {
                rollback_prototype();
                return false;
            }
        }
        if (synchronize_return) {
            const tinfo_t stored_return_type = stored_type.get_rettype();
            if (stored_return_type.empty() ||
                !stored_return_type.equals_to(desired_return_type)) {
                rollback_prototype();
                return false;
            }
        }

        if (options_.debug_mode) {
            qstring stored_text;
            stored_type.print(&stored_text);
            msg("Structor: synchronized signature ea=0x%llX var=%d param=%d return=%s type=%s\n",
                static_cast<unsigned long long>(entry_ea),
                var_idx,
                param_idx,
                synchronize_return ? "true" : "false",
                stored_text.c_str());
        }

        qvector<ea_t> dirty_functions = utils::get_callers(entry_ea);
        dirty_functions.push_back(entry_ea);
        for (ea_t dirty_ea : dirty_functions) {
            (void)mark_cfunc_dirty(dirty_ea, false);
            local_cfunc_cache_.erase(dirty_ea);
            if (shared_cfunc_cache_ != nullptr) {
                shared_cfunc_cache_->erase(dirty_ea);
            }
        }
        return true;
    } catch (const vd_interr_t&) {
        if (prototype_write_attempted) {
            rollback_prototype();
        }
        return false;
    } catch (const vd_failure_t&) {
        if (prototype_write_attempted) {
            rollback_prototype();
        }
        return false;
    } catch (...) {
        if (prototype_write_attempted) {
            rollback_prototype();
        }
        return false;
    }
}

bool TypePropagator::find_callers_with_return(
    ea_t func_ea,
    qvector<std::pair<ea_t, int>>& callers)
{
    qvector<ea_t> caller_funcs = utils::get_callers(func_ea);
    bool complete = true;

    for (ea_t caller_ea : caller_funcs) {
        cfuncptr_t caller_cfunc = get_cfunc(caller_ea);
        if (!caller_cfunc) {
            complete = false;
            continue;
        }

        struct ReturnCallerFinder : public ctree_visitor_t {
            ea_t target_func;
            ea_t caller_ea;
            qvector<std::pair<ea_t, int>>& results;

            ReturnCallerFinder(ea_t func, ea_t caller, qvector<std::pair<ea_t, int>>& r)
                : ctree_visitor_t(CV_FAST)
                , target_func(func)
                , caller_ea(caller)
                , results(r) {}

            static cexpr_t* find_base_var(cexpr_t* expr) {
                while (expr) {
                    if (expr->op == cot_var) return expr;
                    if (expr->op == cot_cast || expr->op == cot_ref) {
                        expr = expr->x;
                    } else if (expr->op == cot_add || expr->op == cot_sub) {
                        cexpr_t* left = find_base_var(expr->x);
                        if (left) return left;
                        expr = expr->y;
                    } else if (expr->op == cot_memref || expr->op == cot_memptr) {
                        expr = expr->x;
                    } else if (expr->op == cot_idx) {
                        expr = expr->x;
                    } else {
                        break;
                    }
                }
                return nullptr;
            }

            int idaapi visit_expr(cexpr_t* expr) override {
                if (!expr || expr->op != cot_asg) return 0;

                cexpr_t* lhs = expr->x;
                cexpr_t* rhs = expr->y;
                while (rhs && rhs->op == cot_cast) {
                    rhs = rhs->x;
                }

                if (!rhs || rhs->op != cot_call || !rhs->x) return 0;
                if (rhs->x->op != cot_obj || rhs->x->obj_ea != target_func) return 0;

                cexpr_t* base = find_base_var(lhs);
                if (!base || base->op != cot_var) return 0;

                results.push_back({caller_ea, base->v.idx});
                return 0;
            }
        };

        ReturnCallerFinder finder(func_ea, caller_ea, callers);
        finder.apply_to(&caller_cfunc->body, nullptr);
    }
    return complete;
}

bool TypePropagator::is_parameter(cfunc_t* cfunc, int var_idx) {
    if (!cfunc || var_idx < 0) return false;

    lvars_t& lvars = *cfunc->get_lvars();
    if (static_cast<size_t>(var_idx) >= lvars.size()) return false;

    return lvars[var_idx].is_arg_var();
}

int TypePropagator::get_param_index(cfunc_t* cfunc, int var_idx) {
    if (!cfunc || var_idx < 0) return -1;

    lvars_t& lvars = *cfunc->get_lvars();
    if (static_cast<size_t>(var_idx) >= lvars.size()) return -1;

    if (!lvars[var_idx].is_arg_var()) return -1;

    // Count parameters before this one
    int param_idx = 0;
    for (int i = 0; i < var_idx; ++i) {
        if (lvars[i].is_arg_var()) {
            ++param_idx;
        }
    }

    return param_idx;
}

} // namespace structor
