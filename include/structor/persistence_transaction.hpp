#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace structor::persistence_invariants {

/// Test-only fault gate. Production builds compile this to a constant false;
/// test-enabled builds additionally require an explicit harness sentinel.
[[nodiscard]] inline bool persistence_fault_requested(
    const char* stage) noexcept {
#if defined(STRUCTOR_LIVE_TEST_HOOKS)
    const char* harness = std::getenv("STRUCTOR_INTEGRATION_TESTING");
    const char* requested = std::getenv("STRUCTOR_TEST_PERSISTENCE_FAULT");
    return stage != nullptr && harness != nullptr && requested != nullptr &&
           std::strcmp(harness, "1") == 0 && std::strcmp(requested, stage) == 0;
#else
    (void)stage;
    return false;
#endif
}

/// IDA-independent first-touch journal for named-type mutations. The first
/// observed state of a name is immutable for the transaction lifetime; later
/// writes update only the current identity. Complexity: expected O(1) per
/// stage/mark operation, O(n) rollback-order construction, O(n) space.
class PersistenceTransactionJournal {
public:
    struct Entry {
        std::string name;
        bool existed_before = false;
        std::uint64_t original_identity = 0;
        std::uint64_t current_identity = 0;
        bool mutated = false;
    };

    [[nodiscard]] std::optional<std::size_t> stage(
        std::string name,
        bool existed_before,
        std::uint64_t original_identity = 0) {
        if (name.empty()) {
            return std::nullopt;
        }
        const auto existing = indices_.find(name);
        if (existing != indices_.end()) {
            return existing->second;
        }

        const std::size_t index = entries_.size();
        entries_.push_back({
            name, existed_before, original_identity, 0, false});
        try {
            const auto [_, inserted] = indices_.emplace(std::move(name), index);
            if (!inserted) {
                entries_.pop_back();
                return std::nullopt;
            }
        } catch (...) {
            entries_.pop_back();
            throw;
        }
        return index;
    }

    [[nodiscard]] std::optional<std::size_t> find(
        std::string_view name) const {
        const auto found = indices_.find(std::string(name));
        return found == indices_.end()
            ? std::nullopt
            : std::optional<std::size_t>{found->second};
    }

    [[nodiscard]] bool mark_mutated(
        std::size_t index,
        std::uint64_t current_identity) noexcept {
        if (index >= entries_.size() || current_identity == 0) {
            return false;
        }
        entries_[index].current_identity = current_identity;
        entries_[index].mutated = true;
        return true;
    }

    [[nodiscard]] const Entry* entry(std::size_t index) const noexcept {
        return index < entries_.size() ? &entries_[index] : nullptr;
    }

    [[nodiscard]] std::vector<std::size_t> rollback_order() const {
        std::vector<std::size_t> order;
        order.reserve(entries_.size());
        for (std::size_t i = entries_.size(); i > 0; --i) {
            if (entries_[i - 1].mutated) {
                order.push_back(i - 1);
            }
        }
        return order;
    }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    void clear() noexcept {
        entries_.clear();
        indices_.clear();
    }

private:
    std::vector<Entry> entries_;
    std::unordered_map<std::string, std::size_t> indices_;
};

} // namespace structor::persistence_invariants
