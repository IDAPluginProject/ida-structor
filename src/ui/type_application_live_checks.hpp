#pragma once

#if defined(STRUCTOR_LIVE_TEST_HOOKS)

#include <structor/z3/type_applicator.hpp>
#include <structor/utils.hpp>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace structor::detail {

// Runs in a disposable IDB. Rejection cases check both saved function types
// and local types; positive controls exercise actual writes through the API.
inline std::vector<std::pair<const char*, bool>> run_live_type_application_check(
    cfunc_t* cfunc, ea_t foreign_ea, const std::string& scenario)
{
    std::vector<std::pair<const char*, bool>> checks;
    constexpr std::array scenarios{
        "foreign_apply", "unknown_apply", "high_address_apply", "foreign_propagate",
        "foreign_signature", "unknown_signature", "failed_signature",
        "same_apply", "same_signature"};
    if (std::find(scenarios.begin(), scenarios.end(), scenario) == scenarios.end()) {
        return checks;
    }
    if (!cfunc || !cfunc->get_lvars() || cfunc->argidx.empty()) {
        return checks;
    }
    const ea_t target_ea = cfunc->entry_ea;
    const bool signature_only = scenario.ends_with("signature");
    const bool positive = scenario.starts_with("same_");
    const bool propagate = scenario == "foreign_propagate";
    const auto& original_lvars = *cfunc->get_lvars();
    const int first_argument = cfunc->argidx[0];
    if (first_argument < 0 || static_cast<size_t>(first_argument) >= original_lvars.size()) {
        return checks;
    }
    const auto first_argument_locator =
        static_cast<const lvar_locator_t&>(original_lvars[first_argument]);
    std::vector<std::pair<lvar_locator_t, tinfo_t>> original_locals;
    for (const auto& local : original_lvars) {
        original_locals.emplace_back(
            static_cast<const lvar_locator_t&>(local), local.type());
    }
    tinfo_t saved_before;
    const bool had_saved_type = get_tinfo(&saved_before, target_ea);

    z3::FunctionTypeInferenceResult inference;
    inference.success = scenario != "failed_signature";
    inference.status = inference.success ? z3::TypeInferenceStatus::Success
                                        : z3::TypeInferenceStatus::SolverFailure;
    inference.func_ea = scenario.starts_with("foreign_") ? foreign_ea : target_ea;
    if (scenario.starts_with("unknown_")) inference.func_ea = BADADDR;
    if (scenario == "high_address_apply") {
        inference.func_ea = target_ea ^ (UINT64_C(1) << 32);
    }

    const auto floating = z3::InferredType::make_base(z3::BaseType::Float32);
    if (signature_only) {
        inference.return_type = floating;
    } else {
        z3::InferredVariableType local;
        local.var_idx = first_argument;
        local.var_name = original_lvars[first_argument].name;
        local.type = z3::InferredType::make_ptr(floating);
        local.confidence = z3::TypeConfidence::High;
        inference.local_types.push_back(std::move(local));
    }

    z3::TypeApplicationConfig config;
    config.overwrite_existing = true;
    config.force_refresh = false;
    config.propagate_types = propagate;
    z3::TypeApplicator applicator(config);
    bool accepted = false;
    bool rejection_has_no_effects = true;
    if (signature_only) {
        accepted = applicator.apply_signature(cfunc, inference);
    } else {
        const auto result = propagate ? applicator.apply_and_propagate(cfunc, inference)
                                      : applicator.apply(cfunc, inference);
        accepted = result.success();
        rejection_has_no_effects = result.applied_count == 0 &&
            result.propagated_count == 0 && !result.signature_requested &&
            !result.error_message.empty() && result.func_ea == target_ea;
    }
    checks.emplace_back("acceptance_matches_identity", accepted == positive);

    tinfo_t saved_after;
    const bool has_saved_type = get_tinfo(&saved_after, target_ea);
    cfuncptr_t current = utils::get_cfunc(target_ea);
    if (!positive) {
        bool locals_unchanged = current && current->get_lvars() &&
            current->get_lvars()->size() == original_locals.size();
        for (const auto& [locator, before] : original_locals) {
            if (!locals_unchanged) break;
            const auto* local = current->get_lvars()->find(locator);
            locals_unchanged = local && local->type().equals_to(before);
        }
        checks.emplace_back("rejected_without_application", rejection_has_no_effects);
        checks.emplace_back("saved_type_unchanged", had_saved_type == has_saved_type &&
            (!had_saved_type || saved_before.equals_to(saved_after)));
        checks.emplace_back("local_types_unchanged", locals_unchanged);
    } else if (signature_only) {
        func_type_data_t function;
        checks.emplace_back("requested_signature_applied", has_saved_type &&
            saved_after.get_func_details(&function) &&
            function.rettype.equals_to(floating.to_tinfo()));
    } else {
        const auto requested = z3::InferredType::make_ptr(floating).to_tinfo();
        const auto* local = current && current->get_lvars()
            ? current->get_lvars()->find(first_argument_locator) : nullptr;
        checks.emplace_back("requested_local_applied", local &&
            local->type().equals_to(requested));
    }
    return checks;
}

} // namespace structor::detail

#endif
