#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

namespace structor::z3::detail {

struct SignatureArgumentBinding {
    std::size_t parameter_index;
    int local_index;
};

/// cfunc_t::argidx maps prototype arguments to indexes in get_lvars(). A
/// missing, invalid, or ambiguous mapping cannot supply positional type facts.
/// Validation is atomic: no signature facts escape an invalid mapping.
/// Expected O(p) time and O(p) space for p parameters.
[[nodiscard]] inline std::optional<std::vector<SignatureArgumentBinding>>
map_signature_arguments(std::span<const int> argument_indexes,
                        std::size_t parameter_count,
                        std::size_t local_count)
{
    if (argument_indexes.size() != parameter_count) {
        return std::nullopt;
    }
    std::unordered_set<int> seen;
    std::vector<SignatureArgumentBinding> result;
    result.reserve(parameter_count);
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
        const int local = argument_indexes[parameter];
        if (local < 0 || static_cast<std::size_t>(local) >= local_count ||
            !seen.insert(local).second) {
            return std::nullopt;
        }
        result.push_back({parameter, local});
    }
    return result;
}

} // namespace structor::z3::detail
