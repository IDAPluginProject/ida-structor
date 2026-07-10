#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace structor::detail {

/// Stable index for global rewrite registrations. Re-registering a canonical
/// root reuses its slot and replaces all root/alias lookup keys.
class GlobalRewriteIndex {
public:
    using address_type = std::uint64_t;
    using index_type = std::size_t;

    struct Update {
        index_type index = 0;
        bool replaced = false;
    };

    void clear() noexcept {
        canonical_root_index_.clear();
        head_index_.clear();
        alias_index_.clear();
        slot_heads_.clear();
        next_index_ = 0;
    }

    [[nodiscard]] Update upsert(
        address_type root,
        std::optional<address_type> root_head,
        std::span<const address_type> aliases)
    {
        const auto root_it = canonical_root_index_.find(root);
        const std::optional<index_type> existing =
            root_it == canonical_root_index_.end()
                ? std::nullopt
                : std::optional<index_type>{root_it->second};

        const index_type index = existing.value_or(next_index_);
        if (existing) {
            erase_slot(canonical_root_index_, index);
            erase_slot(alias_index_, index);
        } else {
            ++next_index_;
            slot_heads_.resize(next_index_);
        }

        canonical_root_index_[root] = index;
        slot_heads_[index] = root_head;
        rebuild_head_index();
        for (address_type alias : aliases) {
            alias_index_[alias] = index;
        }

        return Update{index, existing.has_value()};
    }

    [[nodiscard]] std::optional<index_type> find_root(address_type address) const noexcept {
        const auto canonical = canonical_root_index_.find(address);
        if (canonical != canonical_root_index_.end()) {
            return canonical->second;
        }
        const auto head = head_index_.find(address);
        return head == head_index_.end()
            ? std::nullopt
            : std::optional<index_type>{head->second};
    }

    [[nodiscard]] std::optional<index_type> find_alias(address_type address) const noexcept {
        const auto it = alias_index_.find(address);
        if (it == alias_index_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] index_type slot_count() const noexcept {
        return next_index_;
    }

private:
    void rebuild_head_index() {
        head_index_.clear();
        std::unordered_map<address_type, std::size_t> counts;
        for (index_type slot = 0; slot < slot_heads_.size(); ++slot) {
            if (!slot_heads_[slot].has_value()) {
                continue;
            }
            const address_type head = *slot_heads_[slot];
            const std::size_t count = ++counts[head];
            if (count == 1) {
                head_index_[head] = slot;
            } else {
                head_index_.erase(head);
            }
        }
    }

    static void erase_slot(
        std::unordered_map<address_type, index_type>& index,
        index_type slot) noexcept
    {
        for (auto it = index.begin(); it != index.end();) {
            if (it->second == slot) {
                it = index.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::unordered_map<address_type, index_type> canonical_root_index_;
    std::unordered_map<address_type, index_type> head_index_;
    std::unordered_map<address_type, index_type> alias_index_;
    std::vector<std::optional<address_type>> slot_heads_;
    index_type next_index_ = 0;
};

} // namespace structor::detail
