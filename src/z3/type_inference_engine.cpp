#include "structor/z3/type_inference_engine.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

#ifndef STRUCTOR_TESTING
#include <pro.h>
#include <kernwin.hpp>
#include <name.hpp>
#include <funcs.hpp>
#endif

namespace structor::z3 {

// ============================================================================
// TypeInferenceStats implementation
// ============================================================================

qstring TypeInferenceStats::summary() const {
    qstring result;
    result.sprnt("Type Inference Statistics:\n");
    result.cat_sprnt("  Functions analyzed: %u\n", functions_analyzed);
    result.cat_sprnt("  Variables typed: %u\n", variables_typed);
    result.cat_sprnt("  Constraints: %u hard, %u soft\n", 
                    type_constraints_hard, type_constraints_soft);
    if (constraints_relaxed > 0) {
        result.cat_sprnt("  Constraints relaxed: %u\n", constraints_relaxed);
    }
    result.cat_sprnt("  Alias pairs: %u\n", alias_pairs_found);
    result.cat_sprnt("  Types inferred: %u (ptr=%u, int=%u, float=%u, unknown=%u)\n",
                    types_inferred, types_pointer, types_integer, 
                    types_floating, types_unknown);
    result.cat_sprnt("  Solve iterations: %u\n", solve_iterations);
    result.cat_sprnt("  Timings:\n");
    result.cat_sprnt("    Constraint extraction: %lldms\n",
                    static_cast<long long>(constraint_extraction_time.count()));
    result.cat_sprnt("    Alias analysis: %lldms\n",
                    static_cast<long long>(alias_analysis_time.count()));
    result.cat_sprnt("    Constraint building: %lldms\n",
                    static_cast<long long>(constraint_building_time.count()));
    result.cat_sprnt("    Solving: %lldms\n",
                    static_cast<long long>(solving_time.count()));
    result.cat_sprnt("    Total: %lldms\n",
                    static_cast<long long>(total_time.count()));
    return result;
}

// ============================================================================
// FunctionTypeInferenceResult implementation
// ============================================================================

std::optional<InferredType> FunctionTypeInferenceResult::get_var_type(int var_idx) const {
    for (const auto& vt : local_types) {
        if (vt.var_idx == var_idx) {
            return vt.type;
        }
    }
    return std::nullopt;
}

std::optional<InferredType> FunctionTypeInferenceResult::get_mem_type(
    ea_t base, 
    sval_t offset) const
{
    std::size_t hash = std::hash<ea_t>{}(base) ^ (std::hash<sval_t>{}(offset) << 1);
    auto it = memory_types.find(hash);
    if (it != memory_types.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::unordered_map<int, tinfo_t> FunctionTypeInferenceResult::to_ida_types() const {
    std::unordered_map<int, tinfo_t> result;
    for (const auto& vt : local_types) {
        result[vt.var_idx] = vt.type.to_tinfo();
    }
    return result;
}

// ============================================================================
// TypeInferenceEngine implementation
// ============================================================================

TypeInferenceEngine::TypeInferenceEngine(
    Z3Context& ctx,
    const TypeInferenceConfig& config)
    : ctx_(ctx)
    , config_(config)
    , current_constraints_(ctx)
    , type_encoder_(ctx)
{
    if (config_.enable_experimental_pipeline) {
        initialize_analyzers();
    }
}

void TypeInferenceEngine::initialize_analyzers() {
    semantics_extractor_ = std::make_unique<InstructionSemanticsExtractor>(
        ctx_, config_.semantics_config);
    alias_analyzer_ = std::make_unique<AliasAnalyzer>(ctx_, config_.alias_config);
}

void TypeInferenceEngine::reset_state() {
    current_cfunc_ = nullptr;
    current_constraints_.clear();
    var_to_type_var_.clear();
}

FunctionTypeInferenceResult TypeInferenceEngine::infer_function(cfunc_t* cfunc) {
    FunctionTypeInferenceResult result;
    last_stats_ = TypeInferenceStats();

    if (!config_.enable_experimental_pipeline) {
        result.status = TypeInferenceStatus::ExperimentalDisabled;
        result.error_message =
            "experimental type-inference pipeline is disabled; set "
            "TypeInferenceConfig::enable_experimental_pipeline=true for "
            "explicit per-function experimentation";
        return result;
    }

    if (!cfunc) {
        result.status = TypeInferenceStatus::InvalidInput;
        result.error_message = "null cfunc";
        return result;
    }
    if (ctx_.config().max_memory_mb != 0) {
        result.status = TypeInferenceStatus::UnsupportedOperation;
        result.error_message =
            "configured memory limit cannot be enforced by the type-inference optimizer";
        return result;
    }

    const auto total_start = std::chrono::steady_clock::now();
    const auto finish = [&]() {
        const auto total_end = std::chrono::steady_clock::now();
        last_stats_.total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                total_end - total_start);
        result.stats = last_stats_;
    };

    try {
        // Recreate analyzers so mutations made through config() between calls
        // are reflected instead of silently retaining constructor-time values.
        initialize_analyzers();

        reset_state();
        current_cfunc_ = cfunc;
        result.func_ea = cfunc->entry_ea;

#ifndef STRUCTOR_TESTING
        qstring func_name;
        get_func_name(&func_name, cfunc->entry_ea);
        result.func_name = func_name;
#endif

        last_stats_.functions_analyzed = 1;
        report_progress("Starting", 0,
                        "Initializing experimental type inference");

        if (config_.phase_constraint_extraction) {
            report_progress("Constraints", 10, "Extracting type constraints");
            phase_constraint_extraction(cfunc);
        }

        if (config_.phase_alias_analysis) {
            report_progress("Alias", 30, "Performing alias analysis");
            phase_alias_analysis(cfunc);
        }

        if (config_.phase_soft_constraints) {
            report_progress("Heuristics", 50, "Generating soft constraints");
            phase_soft_constraints(cfunc);
        }

        report_progress("Building", 60, "Building Z3 constraints");
        const auto build_start = std::chrono::steady_clock::now();
        ::z3::optimize opt = build_z3_constraints();
        const auto build_end = std::chrono::steady_clock::now();
        last_stats_.constraint_building_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                build_end - build_start);

        report_progress("Solving", 70, "Solving constraints");
        ::z3::model model(ctx_.ctx());
        const bool solved = phase_solve(opt, model);

        if (!solved) {
            result.status = TypeInferenceStatus::SolverFailure;
            result.error_message =
                "type-inference constraints were unsatisfiable or the solver "
                "returned unknown";
        } else {
            report_progress("Extracting", 90, "Extracting inferred types");
            extract_results(model, result);
            result.status = TypeInferenceStatus::Success;
            result.success = true;
        }

        report_progress("Complete", 100,
                        result.success ? "Experimental type inference complete"
                                       : "Experimental type inference failed");
    } catch (const std::exception& exception) {
        result.status = TypeInferenceStatus::InternalError;
        result.success = false;
        result.error_message.sprnt(
            "experimental type-inference exception: %s", exception.what());
        reset_state();
    } catch (...) {
        result.status = TypeInferenceStatus::InternalError;
        result.success = false;
        result.error_message =
            "unknown exception in experimental type-inference pipeline";
        reset_state();
    }

    finish();
    return result;
}

InferredVariableType TypeInferenceEngine::infer_variable(cfunc_t* cfunc, int var_idx) {
    auto full_result = infer_function(cfunc);

    if (!full_result.success) {
        throw std::runtime_error(full_result.error_message.c_str());
    }

    for (const auto& vt : full_result.local_types) {
        if (vt.var_idx == var_idx) {
            return vt;
        }
    }
    
    InferredVariableType not_found;
    not_found.var_idx = var_idx;
    not_found.type = InferredType::unknown();
    not_found.confidence = TypeConfidence::Low;
    return not_found;
}

std::vector<FunctionTypeInferenceResult> TypeInferenceEngine::infer_cross_function(
    const qvector<cfunc_t*>& cfuncs)
{
    std::vector<FunctionTypeInferenceResult> results;
    results.reserve(cfuncs.size());

    for (auto* cfunc : cfuncs) {
        FunctionTypeInferenceResult result;
        result.func_ea = cfunc ? cfunc->entry_ea : BADADDR;
        result.status = TypeInferenceStatus::UnsupportedOperation;
        result.error_message =
            "interprocedural type-inference fixed-point analysis is not implemented";
        results.push_back(std::move(result));
    }

    return results;
}

void TypeInferenceEngine::phase_constraint_extraction(cfunc_t* cfunc) {
    auto start = std::chrono::steady_clock::now();
    
    // Extract constraints from the ctree
    TypeConstraintSet extracted = semantics_extractor_->extract(cfunc);
    
    // Merge into current constraints
    for (const auto& c : extracted.constraints()) {
        current_constraints_.add(c);
    }
    
    // Record type variables
    for (const auto& tv : extracted.variables()) {
        if (tv.is_local()) {
            var_to_type_var_[tv.var_idx] = tv;
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    last_stats_.constraint_extraction_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
    last_stats_.type_constraints_hard = static_cast<unsigned>(current_constraints_.hard_count());
    last_stats_.type_constraints_soft = static_cast<unsigned>(current_constraints_.soft_count());
    last_stats_.variables_typed = static_cast<unsigned>(var_to_type_var_.size());
}

void TypeInferenceEngine::phase_alias_analysis(cfunc_t* cfunc) {
    auto start = std::chrono::steady_clock::now();
    
    alias_analyzer_->analyze(cfunc);
    
    // Generate type equality constraints for aliasing locations
    auto alias_constraints = alias_analyzer_->generate_type_constraints(var_to_type_var_);
    for (const auto& c : alias_constraints) {
        current_constraints_.add(c);
        last_stats_.alias_pairs_found++;
    }
    
    auto end = std::chrono::steady_clock::now();
    last_stats_.alias_analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
}

void TypeInferenceEngine::phase_soft_constraints(cfunc_t* cfunc) {
    add_type_preferences();
    add_calling_convention_constraints(cfunc);
}

void TypeInferenceEngine::add_type_preferences() {
    // Add soft constraints preferring:
    // 1. Signed integers over unsigned (C default)
    // 2. Concrete types over unknown
    // 3. Smaller types when sufficient
    
    for (const auto& [var_idx, tv] : var_to_type_var_) {
        // Prefer signed integers
        current_constraints_.add(
            TypeConstraint::make_is_signed(tv, BADADDR)
                .soft(config_.weight_signed_over_unsigned)
                .describe("prefer signed integers")
        );
    }
}

void TypeInferenceEngine::add_calling_convention_constraints(cfunc_t* cfunc) {
    if (!cfunc) return;
    
    // Get function type from IDA
    tinfo_t func_type;
    if (!cfunc->get_func_type(&func_type)) return;
    
    func_type_data_t ftd;
    if (!func_type.get_func_details(&ftd)) return;
    
    // Add parameter type constraints
    lvars_t* lvars = cfunc->get_lvars();
    if (!lvars) return;
    
    for (size_t i = 0; i < ftd.size() && i < lvars->size(); ++i) {
        auto it = var_to_type_var_.find(static_cast<int>(i));
        if (it == var_to_type_var_.end()) continue;
        
        auto inferred = InferredType::from_tinfo(ftd[i].type);
        if (inferred.is_unknown()) continue;
        
        current_constraints_.add(
            TypeConstraint::make_one_of(it->second, {inferred}, cfunc->entry_ea)
                .soft(config_.weight_from_signature)
                .describe("parameter type from signature")
        );
    }
}

::z3::optimize TypeInferenceEngine::build_z3_constraints() {
    ::z3::optimize opt(ctx_.ctx());
    
    // Set timeout
    ::z3::params p(ctx_.ctx());
    p.set("timeout", config_.solver_timeout_ms);
    opt.set(p);
    
    // Add hard constraints
    ::z3::expr_vector hard = current_constraints_.to_z3_hard(type_encoder_);
    for (unsigned i = 0; i < hard.size(); ++i) {
        opt.add(hard[i]);
    }
    
    // Add soft constraints with weights
    auto soft = current_constraints_.to_z3_soft(type_encoder_);
    for (const auto& [expr, weight] : soft) {
        opt.add_soft(expr, weight);
    }
    
    return opt;
}

bool TypeInferenceEngine::phase_solve(::z3::optimize& opt, ::z3::model& out_model) {
    auto start = std::chrono::steady_clock::now();
    
    auto result = opt.check();
    last_stats_.solve_iterations++;
    
    auto end = std::chrono::steady_clock::now();
    last_stats_.solving_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
    
    if (result == ::z3::sat) {
        out_model = opt.get_model();
        return true;
    }
    
    // Unknown (including timeout) is not evidence of satisfiability. Never
    // expose a partial model as successful inferred type information.
    return false;
}

void TypeInferenceEngine::extract_results(
    const ::z3::model& model,
    FunctionTypeInferenceResult& result)
{
    for (const auto& [var_idx, tv] : var_to_type_var_) {
        InferredVariableType ivt;
        ivt.var_idx = var_idx;
        ivt.var_name = tv.name;
        
        // Get the Z3 expression for this type variable
        ::z3::expr z3_var = current_constraints_.get_z3_var(tv, type_encoder_);
        
        // Decode the model value
        ivt.type = type_encoder_.decode(z3_var, model);
        
        // Determine confidence based on constraint sources
        // Higher confidence if multiple consistent constraints
        ivt.confidence = TypeConfidence::Medium;
        
        // Categorize for statistics
        if (ivt.type.is_pointer()) {
            last_stats_.types_pointer++;
        } else if (ivt.type.is_base()) {
            if (is_integer(ivt.type.base_type())) {
                last_stats_.types_integer++;
            } else if (is_floating(ivt.type.base_type())) {
                last_stats_.types_floating++;
            } else if (ivt.type.base_type() == BaseType::Unknown) {
                last_stats_.types_unknown++;
            }
        }
        last_stats_.types_inferred++;
        
        result.local_types.push_back(std::move(ivt));
    }
}

TypeVariable TypeInferenceEngine::get_type_var(int var_idx) {
    auto it = var_to_type_var_.find(var_idx);
    if (it != var_to_type_var_.end()) {
        return it->second;
    }
    
    // Use the extractor's exact registry so later constraints refer to the
    // same local, independently of diagnostic index allocation order.
    TypeVariable tv = semantics_extractor_->get_var_type(current_cfunc_, var_idx);
    var_to_type_var_[var_idx] = tv;
    return tv;
}

void TypeInferenceEngine::report_progress(
    const char* phase, 
    int progress, 
    const char* message)
{
    if (progress_callback_) {
        progress_callback_(phase, progress, message);
    }
}

// ============================================================================
// TypeScheme implementation
// ============================================================================

std::pair<InferredType, std::unordered_map<int, TypeVariable>> 
TypeScheme::instantiate(
    int call_site_id, 
    std::function<TypeVariable(int, const char*)> make_var) const
{
    (void)call_site_id;
    (void)make_var;
    if (!type_params.empty()) {
        throw std::logic_error(
            "polymorphic TypeScheme instantiation is unsupported because "
            "InferredType has no type-parameter representation");
    }

    return {body, {}};
}

// ============================================================================
// PolymorphicFunctionDetector implementation
// ============================================================================

PolymorphicFunctionDetector::PolymorphicFunctionDetector(Z3Context& ctx) {
    (void)ctx;
}

bool PolymorphicFunctionDetector::is_polymorphic(ea_t func_ea) const {
    return known_schemes_.find(func_ea) != known_schemes_.end();
}

std::optional<TypeScheme> PolymorphicFunctionDetector::get_type_scheme(
    ea_t func_ea) const {
    auto it = known_schemes_.find(func_ea);
    if (it != known_schemes_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void PolymorphicFunctionDetector::register_polymorphic(ea_t func_ea, TypeScheme scheme) {
    known_schemes_[func_ea] = std::move(scheme);
}

// ============================================================================
// CallingConventionDetector implementation
// ============================================================================

CallingConventionDetector::CallingConventionDetector(Z3Context& ctx) : ctx_(ctx) {}

CallingConventionDetector::Convention CallingConventionDetector::detect(cfunc_t* cfunc) {
    if (!cfunc) return Convention::Unknown;
    
    // Try to detect from function type
    tinfo_t func_type;
    if (cfunc->get_func_type(&func_type)) {
        cm_t cc = func_type.get_cc();
        switch (cc) {
            case CM_CC_CDECL:    return Convention::CDecl;
            case CM_CC_STDCALL:  return Convention::Stdcall;
            case CM_CC_FASTCALL: return Convention::Fastcall;
            case CM_CC_THISCALL: return Convention::Thiscall;
            default: break;
        }
    }
    
    // Platform-based detection
#ifndef STRUCTOR_TESTING
    if (inf_is_64bit()) {
#ifdef __APPLE__
        return Convention::SystemV_x64;
#elif defined(_WIN32)
        return Convention::Microsoft_x64;
#else
        return Convention::SystemV_x64;
#endif
    }
#endif
    
    return Convention::Unknown;
}

qvector<InferredType> CallingConventionDetector::get_param_constraints(
    Convention conv,
    cfunc_t* cfunc)
{
    qvector<InferredType> result;
    
    // Get parameters from function type
    tinfo_t func_type;
    if (!cfunc || !cfunc->get_func_type(&func_type)) return result;
    
    func_type_data_t ftd;
    if (!func_type.get_func_details(&ftd)) return result;
    
    for (const auto& arg : ftd) {
        result.push_back(InferredType::from_tinfo(arg.type));
    }
    
    return result;
}

std::optional<InferredType> CallingConventionDetector::get_return_constraint(
    Convention conv,
    cfunc_t* cfunc)
{
    tinfo_t func_type;
    if (!cfunc || !cfunc->get_func_type(&func_type)) return std::nullopt;
    
    func_type_data_t ftd;
    if (!func_type.get_func_details(&ftd)) return std::nullopt;
    
    return InferredType::from_tinfo(ftd.rettype);
}

std::vector<CallingConventionDetector::ParamLocation> 
CallingConventionDetector::get_param_locations(
    Convention conv,
    const qvector<InferredType>& param_types)
{
    std::vector<ParamLocation> result;
    
    // Convention-specific parameter passing rules
    switch (conv) {
        case Convention::SystemV_x64: {
            // RDI, RSI, RDX, RCX, R8, R9, then stack
            const char* int_regs[] = {"rdi", "rsi", "rdx", "rcx", "r8", "r9"};
            const char* float_regs[] = {"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"};
            
            int int_idx = 0;
            int float_idx = 0;
            sval_t stack_off = 0;
            
            for (const auto& pt : param_types) {
                ParamLocation loc;
                if (pt.is_base() && is_floating(pt.base_type())) {
                    if (float_idx < 8) {
                        loc.is_register = true;
                        loc.reg_name = float_regs[float_idx++];
                    } else {
                        loc.is_register = false;
                        loc.stack_offset = stack_off;
                        stack_off += 8;
                    }
                } else {
                    if (int_idx < 6) {
                        loc.is_register = true;
                        loc.reg_name = int_regs[int_idx++];
                    } else {
                        loc.is_register = false;
                        loc.stack_offset = stack_off;
                        stack_off += 8;
                    }
                }
                result.push_back(loc);
            }
            break;
        }
        
        case Convention::Microsoft_x64: {
            // RCX, RDX, R8, R9, then stack
            const char* regs[] = {"rcx", "rdx", "r8", "r9"};
            sval_t stack_off = 32;  // Shadow space
            
            for (size_t i = 0; i < param_types.size(); ++i) {
                ParamLocation loc;
                if (i < 4) {
                    loc.is_register = true;
                    loc.reg_name = regs[i];
                } else {
                    loc.is_register = false;
                    loc.stack_offset = stack_off;
                    stack_off += 8;
                }
                result.push_back(loc);
            }
            break;
        }
        
        default:
            // Default: all on stack
            sval_t off = 0;
            for (size_t i = 0; i < param_types.size(); ++i) {
                ParamLocation loc;
                loc.is_register = false;
                loc.stack_offset = off;
                off += param_types[i].size(8);
                result.push_back(loc);
            }
            break;
    }
    
    return result;
}

} // namespace structor::z3
