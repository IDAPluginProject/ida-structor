#pragma once

#include <cstdint>
#include <functional>

namespace structor::detail {

/// Exact cross-function local identity. Hashes select buckets; full addresses
/// and local indices decide equality, including when hashes collide.
struct FunctionVariableKey {
    std::uint64_t function_ea;
    int variable_index;

    bool operator==(const FunctionVariableKey&) const = default;
};

struct FunctionVariableKeyHash {
    std::size_t operator()(const FunctionVariableKey& key) const noexcept {
        return std::hash<std::uint64_t>{}(key.function_ea) ^
               (std::hash<int>{}(key.variable_index) << 1);
    }
};

} // namespace structor::detail
