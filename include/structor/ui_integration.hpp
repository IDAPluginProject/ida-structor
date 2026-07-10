#pragma once

#include "synth_types.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "access_collector.hpp"
#include "layout_synthesizer.hpp"
#include "vtable_detector.hpp"
#include "type_propagator.hpp"
#include "pseudocode_rewriter.hpp"
#include "structure_persistence.hpp"
#include "type_matcher.hpp"
#include "api.hpp"

namespace structor {

/// Main action handler for structure synthesis
class SynthActionHandler : public action_handler_t {
public:
    int idaapi activate(action_activation_ctx_t* ctx) override;
    action_state_t idaapi update(action_update_ctx_t* ctx) override;
};

/// Action handler for sparse matching against existing structure types.
class MatchExistingActionHandler : public action_handler_t {
public:
    int idaapi activate(action_activation_ctx_t* ctx) override;
    action_state_t idaapi update(action_update_ctx_t* ctx) override;
};

/// Hex-Rays callback handler
struct HexRaysHooks {
    static ssize_t idaapi callback(void* ud, hexrays_event_t event, va_list va);
};

/// UI integration - stateless functions, no singleton
namespace ui {

inline bool g_initialized = false;
inline bool g_synth_action_registered = false;
inline bool g_match_action_registered = false;
inline bool g_hexrays_callback_installed = false;
inline SynthActionHandler g_synth_action_handler;
inline MatchExistingActionHandler g_match_action_handler;

[[nodiscard]] inline std::size_t synthesis_evidence_count(
    const AccessPattern& local_pattern,
    const std::optional<UnifiedAccessPattern>& unified_pattern)
{
    if (unified_pattern.has_value()) {
        return unified_pattern->unique_access_locations();
    }

    return local_pattern.access_count();
}

/// Execute synthesis on selected variable
inline SynthResult do_synthesis(cfunc_t* cfunc, int var_idx, const SynthOptions& opts) {
    if (!cfunc) {
        return SynthResult::make_error(
            SynthError::InternalError, "No decompilation available");
    }

    SynthOptions materialization_options = opts;
    return StructorAPI::instance().synthesize_structure(
        cfunc->entry_ea,
        var_idx,
        MaterializationMode::PersistAndApply,
        &materialization_options);
}

/// Show conflict resolution dialog
inline bool show_conflict_dialog(const qvector<AccessConflict>& conflicts) {
    qstring msg_text;
    msg_text = "The following access conflicts were detected:\n\n";

    for (const auto& conflict : conflicts) {
        msg_text.cat_sprnt("Offset 0x%X: %s\n", static_cast<unsigned>(conflict.offset), conflict.description.c_str());
        for (const auto& acc : conflict.conflicting_accesses) {
            msg_text.cat_sprnt("  - Size %u at EA 0x%llX\n", acc.size, static_cast<unsigned long long>(acc.insn_ea));
        }
        msg_text.append("\n");
    }

    msg_text.append("These will be marked as union candidates.\nContinue with synthesis?");

    return ask_yn(ASKBTN_YES, "%s", msg_text.c_str()) == ASKBTN_YES;
}

struct MatchPreviewResult {
    SynthError error = SynthError::Success;
    qstring error_message;
    AccessPattern local_pattern;
    SynthesisResult synthesis;

    [[nodiscard]] bool success() const noexcept {
        return error == SynthError::Success;
    }
};

inline MatchPreviewResult preview_synthesis(cfunc_t* cfunc, int var_idx, const SynthOptions& opts) {
    MatchPreviewResult result;
    if (!cfunc) {
        result.error = SynthError::InternalError;
        result.error_message = "No decompilation available";
        return result;
    }

    SynthOptions preview_options = opts;
    VariableStructureAnalysisResult analysis =
        StructorAPI::instance().analyze_structure(
            cfunc->entry_ea, var_idx, &preview_options);
    result.error = analysis.error;
    result.error_message = analysis.error_message;
    result.local_pattern = std::move(analysis.local_pattern);
    result.synthesis = std::move(analysis.synthesis);
    return result;
}

#ifndef STRUCTOR_TESTING
class TypeMatchChooser : public chooser_t {
public:
    TypeMatchChooser(const qstring& synth_name, const qstring& path, const qvector<TypeOverlapCandidate>& matches)
        : chooser_t(CH_MODAL | CH_KEEP | CH_QFLT,
                    kColumnCount,
                    kWidths,
                    kHeaders,
                    "Structor: Match Existing Structure")
        , synth_name_(synth_name)
        , path_(path)
        , matches_(matches)
    {
        deflt_col = 1;
        width = 120;
        height = 24;
    }

    size_t idaapi get_count() const override {
        return matches_.size() + 1;
    }

    void idaapi get_row(qstrvec_t* out,
                        int* out_icon,
                        chooser_item_attrs_t* out_attrs,
                        size_t n) const override
    {
        (void)out_icon;
        (void)out_attrs;
        try {
            if (!out || n > matches_.size()) {
                return;
            }

            out->clear();
            if (n == 0) {
                out->push_back("Generated");
                out->push_back(synth_name_);
                out->push_back(path_);
                out->push_back("0.0%");
                out->push_back("Keep the generated structure unchanged");
                out->push_back("");
                return;
            }

            const TypeOverlapCandidate& match = matches_[n - 1];
            qstring size_text;
            if (match.size != 0) {
                size_text.sprnt("0x%X", match.size);
            } else {
                size_text = "?";
            }

            qstring score_text;
            score_text.sprnt("%.1f%%", match.score * 100.0);

            out->push_back("Merge");
            out->push_back(match.name);
            out->push_back(size_text);
            out->push_back(score_text);
            out->push_back(match.summary);

            qstring preview;
            unsigned shown = 0;
            for (const auto& field : match.fields) {
                if (field.is_padding || field.size == 0) {
                    continue;
                }
                if (shown != 0) {
                    preview.append(", ");
                }
                preview.cat_sprnt(
                    "%s@0x%X", field.name.c_str(),
                    static_cast<unsigned>(field.offset));
                if (++shown >= 5) {
                    preview.append(", ...");
                    break;
                }
            }
            out->push_back(preview);
        } catch (...) {
            try {
                if (out) {
                    out->clear();
                }
            } catch (...) {}
        }
    }

private:
    static constexpr int kColumnCount = 6;
    static constexpr int kWidths[kColumnCount] = { 10, 32, 12, 10, 44, 60 };
    static constexpr const char* kHeaders[kColumnCount] = {
        "Action", "Type", "Size", "Score", "Overlap", "Fields"
    };

    qstring synth_name_;
    qstring path_;
    const qvector<TypeOverlapCandidate>& matches_;
};

inline std::optional<TypeOverlapCandidate> choose_existing_match(
    const SynthStruct& synth_struct,
    const qstring& path)
{
    ExistingTypeMatcher matcher;
    qvector<TypeOverlapCandidate> matches = matcher.find_matches(synth_struct, 128, 0.02);
    if (matches.empty()) {
        msg("Structor: No existing type overlap candidates for %s\n", path.c_str());
        return std::nullopt;
    }

    TypeMatchChooser chooser(synth_struct.name, path, matches);
    const ssize_t selected = chooser.choose(matches.empty() ? 0 : 1);
    if (selected <= 0) {
        return std::nullopt;
    }

    return matches[static_cast<size_t>(selected - 1)];
}

inline void recursively_match_existing_types(SynthStruct& structure,
                                             qvector<SubStructInfo>& children,
                                             const qstring& path)
{
    for (auto& child : children) {
        qstring child_path = path;
        child_path.append(".");
        child_path.append(child.field_name.c_str());
        recursively_match_existing_types(child.structure, child.children, child_path);
    }

    std::optional<TypeOverlapCandidate> selected = choose_existing_match(structure, path);
    if (!selected.has_value()) {
        return;
    }

    ExistingTypeMatcher matcher;
    SynthStruct preview = structure;
    TypeMergeResult merge = matcher.merge_existing_type(preview, *selected);
    if (!merge.success) {
        warning("Structor: Could not merge %s: %s", selected->name.c_str(), merge.message.c_str());
        return;
    }

    qstring prompt;
    prompt.sprnt("Merge existing type '%s' into generated '%s'?\n\n%s\n\nExisting field names/types overwrite generated fields at matching offsets. Padding is ignored.",
                 selected->name.c_str(),
                 structure.name.c_str(),
                 merge.message.c_str());
    if (ask_yn(ASKBTN_YES, "%s", prompt.c_str()) != ASKBTN_YES) {
        return;
    }

    structure = std::move(preview);
    msg("Structor: %s\n", merge.message.c_str());
}
#endif

/// Show synthesis result summary
inline void show_result_dialog(const SynthResult& result) {
    qstring msg_text;

    if (result.success()) {
        msg_text.sprnt("Structure synthesis completed successfully.\n\n");
        msg_text.cat_sprnt("Structure TID: 0x%llX\n", static_cast<unsigned long long>(result.struct_tid));
        msg_text.cat_sprnt("Fields created: %d\n", result.fields_created);

        if (result.vtable_tid != BADADDR) {
            msg_text.cat_sprnt("VTable TID: 0x%llX\n", static_cast<unsigned long long>(result.vtable_tid));
            msg_text.cat_sprnt("VTable slots: %d\n", result.vtable_slots);
        }

        if (!result.propagated_to.empty()) {
            msg_text.cat_sprnt("\nType propagated to %zu functions.\n", result.propagated_to.size());
        }
    } else {
        msg_text.sprnt("Structure synthesis failed.\n\n");
        msg_text.cat_sprnt("Error: %s\n", synth_error_str(result.error));
        if (!result.error_message.empty()) {
            msg_text.cat_sprnt("Details: %s\n", result.error_message.c_str());
        }
    }

    info("%s", msg_text.c_str());
}

/// Open structure view for synthesized type
inline void open_struct_view(tid_t tid) {
    open_loctypes_window(static_cast<int>(tid));
}

/// Execute synthesis on selected variable in vdui
inline SynthResult execute_synthesis(vdui_t* vdui) {
    if (!vdui || !vdui->cfunc) {
        return SynthResult::make_error(SynthError::InternalError, "No decompilation available");
    }

    // Get variable at cursor
    auto [var, var_idx] = utils::get_var_at_cursor(vdui);
    if (!var || var_idx < 0) {
        return SynthResult::make_error(SynthError::NoVariableSelected, "No variable at cursor position");
    }
    const lvar_locator_t source_locator =
        static_cast<const lvar_locator_t&>(*var);

    // Check if variable already has a structure type
    const SynthOptions& opts = Config::instance().options();
    tinfo_t var_type = var->type();

    if (!var_type.empty() && var_type.is_ptr()) {
        tinfo_t pointed = var_type.get_pointed_object();
        if (!pointed.empty() && pointed.is_struct()) {
            if (opts.interactive_mode) {
                if (!utils::ask_yes_no("Variable already has a structure type. Retype?")) {
                    return SynthResult::make_error(SynthError::InvalidVariable, "User cancelled retype");
                }
            }
        }
    }

    // In interactive mode, resolve conflicts against a write-free preview.
    // The old path prompted only after persistence, so cancellation could not
    // prevent an IDB mutation.
    if (opts.interactive_mode) {
        SynthOptions preview_options = opts;
        VariableStructureAnalysisResult preview =
            StructorAPI::instance().analyze_structure(
                vdui->cfunc->entry_ea, var_idx, &preview_options);
        if (!preview.success()) {
            return SynthResult::make_error(preview.error, preview.error_message);
        }
        if (!preview.synthesis.conflicts.empty() &&
            !show_conflict_dialog(preview.synthesis.conflicts)) {
            return SynthResult::make_error(
                SynthError::ConflictingAccesses,
                "User cancelled due to conflicts");
        }
    }

    // Perform synthesis through the same hardened API used by automation and
    // embedding consumers.
    SynthResult result = do_synthesis(vdui->cfunc, var_idx, opts);

    // Show result in interactive mode
    if (result.success()) {
        if (opts.auto_open_struct && result.struct_tid != BADADDR) {
            open_struct_view(result.struct_tid);
        }

        // Refresh pseudocode view
        vdui->refresh_view(true);

        if (opts.highlight_changes && result.synthesized_struct) {
            int refreshed_var_idx = -1;
            if (vdui->cfunc && resolve_var_index(
                    vdui->cfunc, source_locator, refreshed_var_idx)) {
                PseudocodeRewriter rewriter(opts);
                RewriteResult rw_result = rewriter.rewrite(
                    vdui->cfunc, refreshed_var_idx,
                    *result.synthesized_struct);
                rewriter.highlight_transforms(vdui, rw_result);
            }
        }

        msg("Structor: Created %s with %d fields",
            result.synthesized_struct ? result.synthesized_struct->name.c_str() : "<unknown>",
            result.fields_created);

        if (result.vtable_slots > 0) {
            msg(", %d vtable slots", result.vtable_slots);
        }

        msg("\n");
    }

    return result;
}

/// Execute synthesis preview, interactively merge matching existing types, then persist/apply.
inline SynthResult execute_match_existing_synthesis(vdui_t* vdui) {
#ifdef STRUCTOR_TESTING
    (void)vdui;
    return SynthResult::make_error(SynthError::InternalError, "Existing type matching UI is unavailable in tests");
#else
    if (!vdui || !vdui->cfunc) {
        return SynthResult::make_error(SynthError::InternalError, "No decompilation available");
    }

    auto [var, var_idx] = utils::get_var_at_cursor(vdui);
    if (!var || var_idx < 0) {
        return SynthResult::make_error(SynthError::NoVariableSelected, "No variable at cursor position");
    }

    const SynthOptions& opts = Config::instance().options();
    MatchPreviewResult preview = preview_synthesis(vdui->cfunc, var_idx, opts);
    if (!preview.success()) {
        return SynthResult::make_error(preview.error, preview.error_message);
    }

    SynthResult result;
    result.conflicts = preview.synthesis.conflicts;
    populate_z3_info_from_synthesis(result, preview.synthesis);
    if (result.has_conflicts() && opts.interactive_mode) {
        if (!show_conflict_dialog(result.conflicts)) {
            return SynthResult::make_error(SynthError::ConflictingAccesses, "User cancelled due to conflicts");
        }
    }

    SynthStruct synth_struct = std::move(preview.synthesis.structure);
    qvector<SubStructInfo> sub_structs = std::move(preview.synthesis.sub_structs);

    if (opts.vtable_detection && !synth_struct.has_vtable()) {
        VTableDetector vtable_detector(opts);
        std::optional<SynthVTable> vtable;
        if (preview.synthesis.unified_pattern.has_value() && preview.synthesis.unified_pattern->has_vtable) {
            vtable = vtable_detector.detect(*preview.synthesis.unified_pattern);
        } else if (preview.local_pattern.has_vtable) {
            vtable = vtable_detector.detect(preview.local_pattern, vdui->cfunc);
        }
        if (vtable) {
            synth_struct.vtable = std::move(vtable);
        }
    }

    qstring root_path = synth_struct.name.empty() ? qstring("<root>") : synth_struct.name;
    recursively_match_existing_types(synth_struct, sub_structs, root_path);

    StructurePersistence persistence(opts);
    auto persistence_transaction = persistence.begin_transaction();
    if (!persistence_transaction.has_value()) {
        return SynthResult::make_error(SynthError::InternalError,
            "Failed to begin merged-structure persistence transaction");
    }
    tid_t struct_tid = sub_structs.empty()
        ? persistence.create_struct(synth_struct)
        : persistence.create_struct_with_substructs(synth_struct, sub_structs);

    if (struct_tid == BADADDR) {
        const bool rollback_succeeded = persistence_transaction->rollback();
        if (!rollback_succeeded) {
            return SynthResult::make_error(SynthError::InternalError,
                "Failed to create merged structure and persistence rollback failed");
        }
        return SynthResult::make_error(SynthError::TypeCreationFailed,
            "Failed to create merged structure in IDB");
    }

    result.struct_tid = struct_tid;
    result.fields_created = synth_struct.field_count();
    if (synth_struct.has_vtable()) {
        result.vtable_tid = synth_struct.vtable->tid;
        result.vtable_slots = synth_struct.vtable->slot_count();
    }

    tinfo_t struct_type;
    if (!struct_type.get_type_by_tid(struct_tid)) {
        const bool rollback_succeeded = persistence_transaction->rollback();
        result.struct_tid = BADADDR;
        result.vtable_tid = BADADDR;
        synth_struct.tid = BADADDR;
        if (synth_struct.vtable.has_value()) {
            synth_struct.vtable->tid = BADADDR;
        }
        result.error = rollback_succeeded
            ? SynthError::TypeCreationFailed
            : SynthError::InternalError;
        result.error_message = rollback_succeeded
            ? "Failed to reload persisted merged structure type; "
              "persistence transaction rolled back"
            : "Failed to reload persisted merged structure type; "
              "persistence transaction rollback failed";
        result.synthesized_struct =
            std::make_unique<SynthStruct>(std::move(synth_struct));
        return result;
    }

    result.propagated_to.reserve(1);
    lvars_t* source_lvars = vdui->cfunc->get_lvars();
    if (source_lvars == nullptr || var_idx < 0 ||
        static_cast<size_t>(var_idx) >= source_lvars->size()) {
        result.error = SynthError::InvalidVariable;
        result.error_message = "Merged source variable no longer resolves";
        const bool rolled_back = persistence_transaction->rollback();
        result.struct_tid = BADADDR;
        result.vtable_tid = BADADDR;
        synth_struct.tid = BADADDR;
        if (synth_struct.vtable.has_value()) {
            synth_struct.vtable->tid = BADADDR;
        }
        result.synthesized_struct =
            std::make_unique<SynthStruct>(std::move(synth_struct));
        if (!rolled_back) {
            result.error = SynthError::InternalError;
            result.error_message.append(
                "; persistence transaction rollback failed");
        }
        return result;
    }
    const ea_t source_func_ea = vdui->cfunc->entry_ea;
    const lvar_locator_t source_locator =
        static_cast<const lvar_locator_t&>(
            source_lvars->at(static_cast<size_t>(var_idx)));
    TypePropagator propagator(opts, nullptr, &persistence);
    const bool inject_apply_failure =
        persistence_invariants::persistence_fault_requested(
            "required_source_apply");
    if (inject_apply_failure ||
        !propagator.apply_type(vdui->cfunc, var_idx, struct_type)) {
        if (!inject_apply_failure &&
            propagator.last_application_rollback_failed()) {
            qstring retain_failed_message =
                "Failed to apply the source type, local-variable rollback "
                "failed, and the persistence transaction could not be retained";
            result.error = SynthError::InternalError;
            result.error_message =
                "Failed to apply the source type and local-variable rollback "
                "failed; persisted types were retained to preserve references";
            result.failed_sites.push_back(vdui->cfunc->entry_ea);
            result.synthesized_struct =
                std::make_unique<SynthStruct>(std::move(synth_struct));
            const bool retained = persistence_transaction->commit();
            if (!retained) {
                result.error_message.swap(retain_failed_message);
            }
            return result;
        }
        const bool rollback_succeeded = persistence_transaction->rollback();
        result.struct_tid = BADADDR;
        result.vtable_tid = BADADDR;
        synth_struct.tid = BADADDR;
        if (synth_struct.vtable.has_value()) {
            synth_struct.vtable->tid = BADADDR;
        }
        result.error = rollback_succeeded
            ? SynthError::PropagationFailed
            : SynthError::InternalError;
        result.error_message = rollback_succeeded
            ? "Failed to apply merged structure type to the source variable; "
              "persistence transaction rolled back"
            : "Failed to apply merged structure type to the source variable; "
              "persistence transaction rollback failed";
        result.failed_sites.push_back(vdui->cfunc->entry_ea);
        result.synthesized_struct =
            std::make_unique<SynthStruct>(std::move(synth_struct));
        return result;
    }
    result.synthesized_struct =
        std::make_unique<SynthStruct>(std::move(synth_struct));
    result.error = SynthError::Success;
    if (!persistence_transaction->commit()) {
        result.struct_tid = BADADDR;
        result.vtable_tid = BADADDR;
        result.synthesized_struct->tid = BADADDR;
        if (result.synthesized_struct->vtable.has_value()) {
            result.synthesized_struct->vtable->tid = BADADDR;
        }
        result.error = SynthError::InternalError;
        result.error_message =
            "Persistence commit failed after source-variable type application";
        result.failed_sites.push_back(vdui->cfunc->entry_ea);
        return result;
    }
    result.propagated_to.push_back(vdui->cfunc->entry_ea);

    try {
    apply_recovered_substruct_types(sub_structs, propagator, result);
    if (opts.auto_propagate) {
        cfuncptr_t propagation_cfunc = utils::get_cfunc(source_func_ea);
        int propagation_var_idx = -1;
        if (!propagation_cfunc || !resolve_var_index(
                propagation_cfunc, source_locator,
                propagation_var_idx)) {
            result.failed_sites.push_back(source_func_ea);
            return result;
        }
        PropagationResult prop_result = propagator.propagate(
            source_func_ea,
            propagation_var_idx,
            struct_type,
            PropagationDirection::Both);

        for (const auto& site : prop_result.sites) {
            if (site.success) {
                result.propagated_to.push_back(site.func_ea);
            } else {
                result.failed_sites.push_back(site.func_ea);
            }
        }
    }

    if (opts.auto_open_struct && result.struct_tid != BADADDR) {
        open_struct_view(result.struct_tid);
    }

    vdui->refresh_view(true);

    if (opts.highlight_changes && result.synthesized_struct) {
        int refreshed_var_idx = -1;
        if (vdui->cfunc && resolve_var_index(
                vdui->cfunc, source_locator, refreshed_var_idx)) {
            PseudocodeRewriter rewriter(opts);
            RewriteResult rw_result = rewriter.rewrite(
                vdui->cfunc, refreshed_var_idx,
                *result.synthesized_struct);
            rewriter.highlight_transforms(vdui, rw_result);
        }
    }

    msg("Structor: Created matched %s with %d fields\n",
        result.synthesized_struct ? result.synthesized_struct->name.c_str() : "<unknown>",
        result.fields_created);
    } catch (...) {
        result.failed_sites.push_back(vdui->cfunc->entry_ea);
    }

    return result;
#endif
}

/// Initialize UI hooks and register actions
inline bool initialize() {
    if (g_initialized) return true;
    if (g_synth_action_registered || g_match_action_registered ||
        g_hexrays_callback_installed) {
        msg("Structor: Refusing UI initialization with retained registrations\n");
        return false;
    }

    // Register synthesis action
    const action_desc_t action_desc = ACTION_DESC_LITERAL(
        ACTION_NAME,
        ACTION_LABEL,
        &g_synth_action_handler,
        Config::instance().hotkey(),
        "Create structure from access patterns",
        -1
    );

    if (!register_action(action_desc)) {
        msg("Structor: Failed to register action\n");
        return false;
    }
    g_synth_action_registered = true;

    const action_desc_t match_action_desc = ACTION_DESC_LITERAL(
        MATCH_ACTION_NAME,
        MATCH_ACTION_LABEL,
        &g_match_action_handler,
        DEFAULT_MATCH_HOTKEY,
        "Preview synthesis and merge matching existing structure fields",
        -1
    );

    if (!register_action(match_action_desc)) {
        msg("Structor: Failed to register existing type match action\n");
        if (unregister_action(ACTION_NAME)) {
            g_synth_action_registered = false;
        }
        return false;
    }
    g_match_action_registered = true;

    // Install Hex-Rays callback
    if (!install_hexrays_callback(HexRaysHooks::callback, nullptr)) {
        msg("Structor: Failed to install Hex-Rays callback\n");
        if (unregister_action(MATCH_ACTION_NAME)) {
            g_match_action_registered = false;
        }
        if (unregister_action(ACTION_NAME)) {
            g_synth_action_registered = false;
        }
        return false;
    }
    g_hexrays_callback_installed = true;

    g_initialized = true;
    return true;
}

/// Cleanup UI hooks and unregister actions
[[nodiscard]] inline bool shutdown() {
    if (g_hexrays_callback_installed &&
        remove_hexrays_callback(HexRaysHooks::callback, nullptr) > 0) {
        g_hexrays_callback_installed = false;
    }
    if (g_match_action_registered && unregister_action(MATCH_ACTION_NAME)) {
        g_match_action_registered = false;
    }
    if (g_synth_action_registered && unregister_action(ACTION_NAME)) {
        g_synth_action_registered = false;
    }

    const bool removed = !g_hexrays_callback_installed &&
        !g_match_action_registered && !g_synth_action_registered;
    if (removed) {
        g_initialized = false;
    }
    return removed;
}

} // namespace ui

// ============================================================================
// SynthActionHandler Implementation
// ============================================================================

inline int SynthActionHandler::activate(action_activation_ctx_t* ctx) {
    try {
        vdui_t* vdui = ctx ? get_widget_vdui(ctx->widget) : nullptr;
        if (!vdui) {
            msg("Structor: No pseudocode view active\n");
            return 0;
        }

        SynthResult result = ui::execute_synthesis(vdui);

        if (!result.success()) {
            msg("Structor: %s\n", synth_error_str(result.error));
            if (!result.error_message.empty()) {
                msg("  Details: %s\n", result.error_message.c_str());
            }
        }
        return 1;
    } catch (const vd_failure_t& e) {
        msg("Structor: synthesis action failed: %s\n", e.desc().c_str());
    } catch (const std::exception& e) {
        msg("Structor: synthesis action failed: %s\n", e.what());
    } catch (...) {
        msg("Structor: synthesis action raised an unknown exception\n");
    }
    return 0;
}

inline int MatchExistingActionHandler::activate(action_activation_ctx_t* ctx) {
    try {
        vdui_t* vdui = ctx ? get_widget_vdui(ctx->widget) : nullptr;
        if (!vdui) {
            msg("Structor: No pseudocode view active\n");
            return 0;
        }

        SynthResult result = ui::execute_match_existing_synthesis(vdui);
        if (!result.success()) {
            msg("Structor: %s\n", synth_error_str(result.error));
            if (!result.error_message.empty()) {
                msg("  Details: %s\n", result.error_message.c_str());
            }
        }
        return 1;
    } catch (const vd_failure_t& e) {
        msg("Structor: type-match action failed: %s\n", e.desc().c_str());
    } catch (const std::exception& e) {
        msg("Structor: type-match action failed: %s\n", e.what());
    } catch (...) {
        msg("Structor: type-match action raised an unknown exception\n");
    }
    return 0;
}

inline action_state_t MatchExistingActionHandler::update(action_update_ctx_t* ctx) {
    SynthActionHandler synth_update;
    return synth_update.update(ctx);
}

inline action_state_t SynthActionHandler::update(action_update_ctx_t* ctx) {
    if (!ctx) {
        return AST_DISABLE;
    }
    try {
        vdui_t* vdui = get_widget_vdui(ctx->widget);
        if (!vdui || !vdui->cfunc) {
            return AST_DISABLE;
        }

        if (!vdui->item.is_citem()) {
            return AST_DISABLE;
        }

        const citem_t* item = vdui->item.it;
        if (!item) {
            return AST_DISABLE;
        }

        const cexpr_t* expr = nullptr;
        if (item->is_expr()) {
            expr = static_cast<const cexpr_t*>(item);
        }

        while (expr) {
            if (expr->op == cot_var) {
                return AST_ENABLE;
            }
            if (expr->op == cot_cast || expr->op == cot_ref ||
                expr->op == cot_ptr || expr->op == cot_memptr ||
                expr->op == cot_idx) {
                expr = expr->x;
            } else {
                break;
            }
        }
    } catch (...) {
        // Action-state callbacks are host ABI boundaries. A disabled action is
        // the only state that remains safe when cursor inspection fails.
    }

    return AST_DISABLE;
}

// ============================================================================
// HexRaysHooks Implementation
// ============================================================================

inline ssize_t HexRaysHooks::callback(void* ud, hexrays_event_t event, va_list va) {
    try {
        switch (event) {
            case hxe_populating_popup: {
                TWidget* widget = va_arg(va, TWidget*);
                TPopupMenu* popup = va_arg(va, TPopupMenu*);
                vdui_t* vdui = va_arg(va, vdui_t*);

                if (vdui && vdui->cfunc) {
                    auto [var, var_idx] = utils::get_var_at_cursor(vdui);
                    if (var && var_idx >= 0) {
                        attach_action_to_popup(widget, popup, ACTION_NAME);
                        attach_action_to_popup(widget, popup, MATCH_ACTION_NAME);
                    }
                }
                break;
            }

            case hxe_double_click:
                break;

            default:
                break;
        }
    } catch (const vd_failure_t& e) {
        msg("Structor: UI Hex-Rays callback failed: %s\n", e.desc().c_str());
    } catch (const std::exception& e) {
        msg("Structor: UI Hex-Rays callback failed: %s\n", e.what());
    } catch (...) {
        msg("Structor: UI Hex-Rays callback raised an unknown exception\n");
    }

    return 0;
}

} // namespace structor
