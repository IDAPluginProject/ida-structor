#include <structor/host_integration.hpp>

#include <structor/global_object_analyzer.hpp>
#include <structor/type_fixer.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace structor {

namespace {

#ifndef STRUCTOR_TESTING
template <typename Callable>
class MainThreadRequest final : public exec_request_t {
public:
    explicit MainThreadRequest(Callable&& callable)
        : callable_(std::move(callable)) {}

    ssize_t idaapi execute() override {
        try {
            return callable_();
        } catch (...) {
            return -1;
        }
    }

private:
    Callable callable_;
};

template <typename Callable>
[[nodiscard]] ssize_t dispatch_main_thread(Callable&& callable) {
    using Request = MainThreadRequest<std::decay_t<Callable>>;
    Request request(std::forward<Callable>(callable));
    return execute_sync(request, MFF_WRITE);
}
#endif

void print_type_fix_messages(const TypeFixResult& result, bool include_diagnostics) {
    for (const auto& warning : result.warnings) {
        msg("Structor: %s\n", warning.c_str());
    }

    if (!include_diagnostics) {
        return;
    }

    for (const auto& diagnostic : result.diagnostics) {
        msg("Structor: diagnostic: %s\n", diagnostic.c_str());
    }
}

} // namespace

HostIntegration::HostIntegration(HostIntegrationOptions options)
    : options_(options) {}

HostIntegration::~HostIntegration() {
    shutdown();
    // Destruction with a live callback would leave the host holding a dangling
    // userdata pointer. A failed synchronous dispatch cannot be recovered by
    // invoking the non-thread-safe Hex-Rays removal API from the worker thread.
    if (hook_lifecycle_.is_hooked()) {
        std::terminate();
    }
}

bool HostIntegration::install_hexrays_hooks() {
#ifndef STRUCTOR_TESTING
    if (!is_main_thread()) {
        return dispatch_main_thread([this]() -> ssize_t {
            return install_hexrays_hooks() ? 1 : 0;
        }) > 0;
    }
#endif

    if (!hook_lifecycle_.can_install()) {
        return false;
    }
    if (hook_lifecycle_.is_hooked()) {
        return true;
    }

    if (!install_hexrays_callback(hexrays_callback, this)) {
        return false;
    }

    if (!hook_lifecycle_.mark_installed()) {
        remove_hexrays_callback(hexrays_callback, this);
        return false;
    }
    return true;
}

void HostIntegration::uninstall_hexrays_hooks() {
#ifndef STRUCTOR_TESTING
    if (!is_main_thread()) {
        (void)dispatch_main_thread([this]() -> ssize_t {
            uninstall_hexrays_hooks();
            return 1;
        });
        return;
    }
#endif

    if (!hook_lifecycle_.is_hooked()) {
        return;
    }

    const int removed = remove_hexrays_callback(hexrays_callback, this);
    if (removed <= 0) {
        return;
    }
    hook_lifecycle_.mark_uninstalled();
}

void HostIntegration::shutdown() {
#ifndef STRUCTOR_TESTING
    if (!is_main_thread()) {
        (void)dispatch_main_thread([this]() -> ssize_t {
            shutdown();
            return 1;
        });
        return;
    }
#endif

    const bool first_shutdown = hook_lifecycle_.begin_shutdown();
    // Always enforce the callback invariant, even on repeated shutdown calls.
    uninstall_hexrays_hooks();
    if (!first_shutdown) {
        return;
    }

    processing_functions_.clear();
    processed_functions_.clear();

    if (options_.clear_global_rewrites_on_shutdown) {
        clear_registered_global_rewrite_info();
    }
}

void HostIntegration::reset_processed_functions() {
#ifndef STRUCTOR_TESTING
    if (!is_main_thread()) {
        (void)dispatch_main_thread([this]() -> ssize_t {
            reset_processed_functions();
            return 1;
        });
        return;
    }
#endif

    processing_functions_.clear();
    processed_functions_.clear();
}

void HostIntegration::handle_ctree_maturity(cfunc_t* cfunc, ctree_maturity_t maturity) {
#ifndef STRUCTOR_TESTING
    if (!is_main_thread()) {
        return;
    }
#endif

    if (hook_lifecycle_.is_shutdown() ||
        !cfunc || maturity != CMAT_FINAL || !options_.enable_global_rewrite_callback) {
        return;
    }

    if (Config::instance().options().debug_mode) {
        qstring func_name;
        get_func_name(&func_name, cfunc->entry_ea);
        msg("Structor: hxe_maturity final for %s\n", func_name.c_str());
    }

    (void)rewrite_registered_global_uses(cfunc);
}

void HostIntegration::handle_func_printed(cfunc_t* cfunc) {
#ifndef STRUCTOR_TESTING
    if (!is_main_thread()) {
        return;
    }
#endif

    if (hook_lifecycle_.is_shutdown() ||
        !cfunc || !options_.enable_auto_type_fix_callback) {
        return;
    }

    if (!Config::instance().options().auto_fix_types || auto_type_fixing_suppressed()) {
        return;
    }

    process_decompilation_complete(cfunc);
}

ssize_t idaapi HostIntegration::hexrays_callback(void* ud, hexrays_event_t event, va_list va) {
    auto* self = static_cast<HostIntegration*>(ud);
    if (!self) {
        return 0;
    }

    try {
        switch (event) {
            case hxe_maturity: {
                cfunc_t* cfunc = va_arg(va, cfunc_t*);
                ctree_maturity_t maturity = va_argi(va, ctree_maturity_t);
                self->handle_ctree_maturity(cfunc, maturity);
                break;
            }
            case hxe_func_printed: {
                cfunc_t* cfunc = va_arg(va, cfunc_t*);
                self->handle_func_printed(cfunc);
                break;
            }
            default:
                break;
        }
    } catch (const vd_interr_t& e) {
        msg("Structor: Hex-Rays callback internal error: %s\n", e.desc().c_str());
    } catch (const vd_failure_t& e) {
        msg("Structor: Hex-Rays callback failure: %s\n", e.desc().c_str());
    } catch (const std::exception& e) {
        msg("Structor: Hex-Rays callback exception: %s\n", e.what());
    } catch (...) {
        msg("Structor: Hex-Rays callback raised an unknown exception\n");
    }

    return 0;
}

void HostIntegration::process_decompilation_complete(cfunc_t* cfunc) {
    if (!cfunc) {
        return;
    }

    const ea_t func_ea = cfunc->entry_ea;
    if (processed_functions_.count(func_ea) > 0 ||
        processing_functions_.count(func_ea) > 0) {
        return;
    }

    processing_functions_.insert(func_ea);
    struct ProcessingGuard {
        std::unordered_set<ea_t>& functions;
        ea_t ea;
        ~ProcessingGuard() { functions.erase(ea); }
    } guard{processing_functions_, func_ea};

    if (Config::instance().options().debug_mode) {
        qstring func_name;
        get_func_name(&func_name, func_ea);
        msg("Structor: Processing function %s (0x%llx)\n",
            func_name.c_str(), static_cast<unsigned long long>(func_ea));
    }

    TypeFixerConfig fix_config;
    fix_config.dry_run = false;
    // Function-entry type fixing must not create Local Types. Manual synthesis
    // remains responsible for creating or updating recovered structures.
    fix_config.synthesize_structures = false;
    fix_config.propagate_fixes = Config::instance().options().auto_propagate;
    fix_config.max_propagation_depth = Config::instance().options().max_propagation_depth;
    fix_config.collect_missing_argument_warnings = false;

    TypeFixer fixer(fix_config);
    TypeFixResult result = fixer.fix_function_types(cfunc);
    processed_functions_.insert(func_ea);

    const bool verbose = Config::instance().options().auto_fix_verbose;
    const bool debug = Config::instance().options().debug_mode;
    if (debug) {
        msg("Structor: %s - analyzed %u vars, %u differences, %u fixed\n",
            result.func_name.c_str(),
            result.analyzed,
            result.differences_found,
            result.fixes_applied);
    }

    print_type_fix_messages(result, debug);

    if (verbose && result.fixes_applied > 0) {
        msg("Structor: Auto-fixed %u types in %s\n",
            result.fixes_applied,
            result.func_name.c_str());

        for (const auto& fix : result.variable_fixes) {
            if (fix.applied) {
                msg("  - %s: %s\n",
                    fix.var_name.c_str(),
                    fix.comparison.description.c_str());
            }
        }
    }
}

} // namespace structor
