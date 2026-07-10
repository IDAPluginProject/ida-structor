#include <structor/api.hpp>
#include <structor/host_integration.hpp>

#include <string_view>

int main() {
    auto& api = structor::StructorAPI::instance();
    structor::SynthOptions options;
    if (!api.set_options(options)) {
        return 1;
    }

    structor::HostIntegration integration;
    integration.set_auto_type_fixing_suppressed(true);
    integration.handle_ctree_maturity(nullptr, CMAT_FINAL);
    integration.handle_func_printed(nullptr);
    integration.shutdown();

    if (std::string_view(structor::materialization_mode_str(structor::MaterializationMode::Preview)) != "preview") {
        return 2;
    }

    if (std::string_view(structor::materialization_mode_str(structor::MaterializationMode::PersistAndApply)) != "persist_and_apply") {
        return 3;
    }

    return 0;
}
