#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>

namespace structor::z3 {

/// Identity is independent of diagnostic names and extractor-local indices.
/// The allocator-module token separates independently embedded library copies;
/// the scope separates their interners. These session identities must not be
/// serialized into an IDB or retained after their creating module is unloaded.
struct TypeVariableIdentity {
    std::uintptr_t module_token = 0;
    std::uint64_t scope = 0;
    std::uint64_t ordinal = 0;

    bool operator==(const TypeVariableIdentity&) const = default;
    [[nodiscard]] bool valid() const noexcept {
        return module_token != 0 && scope != 0 && ordinal != 0;
    }
};

struct TypeVariableIdentityHash {
    std::size_t operator()(const TypeVariableIdentity& identity) const noexcept {
        return std::hash<std::uintptr_t>{}(identity.module_token) ^
               (std::hash<std::uint64_t>{}(identity.scope) << 1) ^
               (std::hash<std::uint64_t>{}(identity.ordinal) << 2);
    }
};

/// Use this symbol for the solver; human-readable names are diagnostics only.
[[nodiscard]] inline std::string type_variable_solver_symbol(
    const TypeVariableIdentity& identity)
{
    if (!identity.valid()) {
        throw std::invalid_argument("invalid type variable identity");
    }
    return "structor_type_" + std::to_string(identity.module_token) + "_" +
           std::to_string(identity.scope) + "_" + std::to_string(identity.ordinal);
}

struct LocalTypeVariableKey {
    std::uint64_t function_ea;
    int variable_index;
    int ssa_version;

    bool operator==(const LocalTypeVariableKey&) const = default;
};

struct MemoryTypeVariableKey {
    std::uint64_t base;
    std::int64_t offset;
    std::uint32_t size;

    bool operator==(const MemoryTypeVariableKey&) const = default;
};

struct NamedTypeVariableKey {
    std::uint64_t function_ea;
    std::string name;

    bool operator==(const NamedTypeVariableKey&) const = default;
};

struct ExpressionTypeVariableKey {
    std::uint64_t generation;
    const void* node;

    bool operator==(const ExpressionTypeVariableKey&) const = default;
};

using TypeVariableKey = std::variant<LocalTypeVariableKey, MemoryTypeVariableKey,
                                     NamedTypeVariableKey, ExpressionTypeVariableKey>;

/// Hashes select buckets. Complete tagged keys always determine equality.
struct TypeVariableKeyHash {
    std::size_t operator()(const TypeVariableKey& key) const noexcept {
        return std::visit([&](const auto& value) {
            return hash(value) ^ (std::hash<std::size_t>{}(key.index()) << 3);
        }, key);
    }

private:
    static std::size_t hash(const LocalTypeVariableKey& key) noexcept {
        return std::hash<std::uint64_t>{}(key.function_ea) ^
               (std::hash<int>{}(key.variable_index) << 1) ^
               (std::hash<int>{}(key.ssa_version) << 2);
    }

    static std::size_t hash(const MemoryTypeVariableKey& key) noexcept {
        return std::hash<std::uint64_t>{}(key.base) ^
               (std::hash<std::int64_t>{}(key.offset) << 1) ^
               (std::hash<std::uint32_t>{}(key.size) << 2);
    }

    static std::size_t hash(const NamedTypeVariableKey& key) noexcept {
        return std::hash<std::uint64_t>{}(key.function_ea) ^
               (std::hash<std::string>{}(key.name) << 1);
    }

    static std::size_t hash(const ExpressionTypeVariableKey& key) noexcept {
        return std::hash<std::uint64_t>{}(key.generation) ^
               (std::hash<const void*>{}(key.node) << 1);
    }
};

namespace detail {

struct TypeVariableScope {
    std::uintptr_t module_token;
    std::uint64_t scope;
};

[[nodiscard]] inline TypeVariableScope allocate_type_variable_scope() {
    static std::atomic<std::uint64_t> next_scope{1};
    std::uint64_t scope = next_scope.load(std::memory_order_relaxed);
    for (;;) {
        if (scope == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("type variable scopes exhausted");
        }
        if (next_scope.compare_exchange_weak(scope, scope + 1,
                                             std::memory_order_relaxed)) {
            return {reinterpret_cast<std::uintptr_t>(&next_scope), scope};
        }
    }
}

} // namespace detail

/// Construct an independent identity for a public factory call. Reuse an
/// existing identity (or the interner) when two references denote one variable.
[[nodiscard]] inline TypeVariableIdentity fresh_type_variable_identity() {
    const auto scope = detail::allocate_type_variable_scope();
    return {scope.module_token, scope.scope, 1};
}

/// Exact production interning, with injectable hashing for collision tests.
/// A pass retains node addresses only as keys and never dereferences them.
/// Start a new pass whenever the ctree is replaced or mutated. Local, memory,
/// and named identities remain stable for this interner's lifetime.
///
/// Expected interning time is O(1), plus O(L) for a name of L bytes; adversarial
/// hash collisions can require O(V) equality checks. Storage is O(V + E + S),
/// where V is persistent keys, E nodes in the current pass, and S name bytes.
/// Starting a new expression pass takes O(E) time and releases its node keys.
template<class KeyHash = TypeVariableKeyHash>
class ExactTypeVariableInterner {
public:
    ExactTypeVariableInterner() : scope_(detail::allocate_type_variable_scope()) {}
    ExactTypeVariableInterner(const ExactTypeVariableInterner&) = delete;
    ExactTypeVariableInterner& operator=(const ExactTypeVariableInterner&) = delete;
    ExactTypeVariableInterner(ExactTypeVariableInterner&&) = delete;
    ExactTypeVariableInterner& operator=(ExactTypeVariableInterner&&) = delete;

    [[nodiscard]] TypeVariableIdentity local(std::uint64_t function_ea,
                                           int variable_index,
                                           int ssa_version = 0) {
        return intern(persistent_, LocalTypeVariableKey{
            function_ea, variable_index, ssa_version});
    }

    [[nodiscard]] TypeVariableIdentity memory(std::uint64_t base,
                                            std::int64_t offset,
                                            std::uint32_t size) {
        return intern(persistent_, MemoryTypeVariableKey{base, offset, size});
    }

    [[nodiscard]] TypeVariableIdentity named(std::uint64_t function_ea,
                                           std::string_view name) {
        return intern(persistent_, NamedTypeVariableKey{
            function_ea, std::string(name)});
    }

    void begin_expression_pass() {
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("type variable expression passes exhausted");
        }
        ++generation_;
        expressions_.clear();
        expression_pass_active_ = true;
    }

    void end_expression_pass() noexcept {
        expressions_.clear();
        expression_pass_active_ = false;
    }

    [[nodiscard]] TypeVariableIdentity expression(const void* node) {
        if (!node) {
            throw std::invalid_argument("null expression has no node identity");
        }
        if (!expression_pass_active_) {
            throw std::logic_error("begin an expression pass before interning nodes");
        }
        return intern(expressions_, ExpressionTypeVariableKey{generation_, node});
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return persistent_.size() + expressions_.size();
    }

private:
    using Registry = std::unordered_map<TypeVariableKey, TypeVariableIdentity, KeyHash>;

    const detail::TypeVariableScope scope_;
    std::uint64_t ordinal_ = 0;
    std::uint64_t generation_ = 0;
    bool expression_pass_active_ = false;
    Registry persistent_;
    Registry expressions_;

    [[nodiscard]] TypeVariableIdentity intern(Registry& registry, TypeVariableKey key) {
        if (const auto found = registry.find(key); found != registry.end()) {
            return found->second;
        }
        if (ordinal_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("type variable identities exhausted");
        }
        const TypeVariableIdentity identity{
            scope_.module_token, scope_.scope, ++ordinal_};
        registry.emplace(std::move(key), identity);
        return identity;
    }
};

using TypeVariableInterner = ExactTypeVariableInterner<>;

} // namespace structor::z3
