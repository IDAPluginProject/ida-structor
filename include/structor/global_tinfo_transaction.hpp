#pragma once

#include "synth_types.hpp"

#include <array>

namespace structor::detail {

enum class GlobalTinfoTransactionPhase {
    Pending,
    Committed,
    RollbackRequired,
    RolledBack,
    RollbackFailed,
};

struct GlobalTinfoRollbackPlan {
    bool remove_requested_tinfo = false;
    bool remove_temporary_data_item = false;
    bool restore_prior_tinfo = false;
};

/// IDA-independent transaction policy used by apply_global_tinfo().
class GlobalTinfoTransactionState {
public:
    explicit GlobalTinfoTransactionState(bool had_prior_tinfo) noexcept
        : had_prior_tinfo_(had_prior_tinfo) {}

    void record_temporary_data_item() noexcept {
        temporary_data_item_ = true;
    }

    [[nodiscard]] bool finish_application_verification(bool exact_match) noexcept {
        if (phase_ != GlobalTinfoTransactionPhase::Pending) {
            return false;
        }
        phase_ = exact_match
            ? GlobalTinfoTransactionPhase::Committed
            : GlobalTinfoTransactionPhase::RollbackRequired;
        return exact_match;
    }

    [[nodiscard]] GlobalTinfoRollbackPlan rollback_plan() const noexcept {
        if (phase_ != GlobalTinfoTransactionPhase::RollbackRequired) {
            return {};
        }
        return GlobalTinfoRollbackPlan{
            true,
            temporary_data_item_,
            had_prior_tinfo_,
        };
    }

    [[nodiscard]] bool finish_rollback_verification(
        bool tinfo_restored,
        bool data_item_restored) noexcept
    {
        if (phase_ != GlobalTinfoTransactionPhase::RollbackRequired) {
            return false;
        }
        const bool restored = tinfo_restored && data_item_restored;
        phase_ = restored
            ? GlobalTinfoTransactionPhase::RolledBack
            : GlobalTinfoTransactionPhase::RollbackFailed;
        return restored;
    }

    [[nodiscard]] GlobalTinfoTransactionPhase phase() const noexcept {
        return phase_;
    }

private:
    bool had_prior_tinfo_ = false;
    bool temporary_data_item_ = false;
    GlobalTinfoTransactionPhase phase_ = GlobalTinfoTransactionPhase::Pending;
};

using GlobalTinfoReadbackVerifier = bool (*)(
    const tinfo_t& requested,
    const tinfo_t& observed);

enum class GlobalTinfoApplyResult : std::uint8_t {
    Applied,
    FailedRestored,
    RollbackFailed,
};

[[nodiscard]] inline bool exact_global_tinfo_readback(
    const tinfo_t& requested,
    const tinfo_t& observed)
{
    return observed.equals_to(requested);
}

/// Apply an address type as one exact, rollback-verified IDB transaction.
/// The function never reports success from set_tinfo() alone. When pointer
/// storage must be prepared, only a fully unexplored 4-byte or 8-byte range is
/// materialized, and all prior type/item metadata is verified after rollback.
[[nodiscard]] inline GlobalTinfoApplyResult
apply_global_tinfo_transaction_detailed(
    ea_t ea,
    const tinfo_t& type,
    GlobalTinfoReadbackVerifier verifier)
{
    if (ea == BADADDR || type.empty()) {
        return GlobalTinfoApplyResult::FailedRestored;
    }
#ifndef STRUCTOR_TESTING
    if (!is_main_thread() || verifier == nullptr) {
        return GlobalTinfoApplyResult::FailedRestored;
    }

    try {
        if (!is_mapped(ea)) {
            return GlobalTinfoApplyResult::FailedRestored;
        }
        if (type.is_ptr() || type.is_funcptr()) {
            const asize_t pointer_size = static_cast<asize_t>(get_ptr_size());
            if ((pointer_size != 4 && pointer_size != 8) ||
                ea > BADADDR - pointer_size) {
                return GlobalTinfoApplyResult::FailedRestored;
            }
            for (asize_t i = 0; i < pointer_size; ++i) {
                if (!is_mapped(ea + i)) {
                    return GlobalTinfoApplyResult::FailedRestored;
                }
            }
        }
    } catch (...) {
        return GlobalTinfoApplyResult::FailedRestored;
    }

    struct AddressMetadataSnapshot {
        ea_t item_head = BADADDR;
        asize_t item_size = 0;
        flags64_t address_flags = 0;
        flags64_t head_flags = 0;
        std::array<flags64_t, 8> preparation_flags{};
        asize_t preparation_size = 0;
    } metadata;

    tinfo_t prior_type;
    bool had_prior_type = false;
    try {
        had_prior_type = get_tinfo(&prior_type, ea) && !prior_type.empty();
        metadata.item_head = get_item_head(ea);
        metadata.item_size = metadata.item_head == BADADDR
            ? 0
            : get_item_size(metadata.item_head);
        metadata.address_flags = get_flags(ea);
        metadata.head_flags = metadata.item_head == BADADDR
            ? 0
            : get_flags(metadata.item_head);
    } catch (...) {
        return GlobalTinfoApplyResult::FailedRestored;
    }

    GlobalTinfoTransactionState transaction(had_prior_type);

    if (type.is_ptr() || type.is_funcptr()) {
        try {
            const asize_t pointer_size = static_cast<asize_t>(get_ptr_size());
            bool unexplored_range = pointer_size == 4 || pointer_size == 8;
            if (unexplored_range) {
                metadata.preparation_size = pointer_size;
                for (asize_t i = 0; i < pointer_size; ++i) {
                    const flags64_t flags = get_flags(ea + i);
                    metadata.preparation_flags[static_cast<std::size_t>(i)] = flags;
                    if (!is_unknown(flags)) {
                        unexplored_range = false;
                    }
                }
            }

            if (unexplored_range) {
                const bool created = pointer_size == 8
                    ? create_qword(ea, pointer_size, false)
                    : create_dword(ea, pointer_size, false);
                if (created) {
                    transaction.record_temporary_data_item();
                }
            }
        } catch (...) {
        }
    }

    try {
        (void)set_tinfo(ea, &type);
    } catch (...) {
    }

    bool application_verified = false;
    try {
        tinfo_t observed;
        application_verified = get_tinfo(&observed, ea) &&
            !observed.empty() && verifier(type, observed);
    } catch (...) {
    }

    if (transaction.finish_application_verification(application_verified)) {
        return GlobalTinfoApplyResult::Applied;
    }

    const GlobalTinfoRollbackPlan plan = transaction.rollback_plan();
    bool tinfo_restored = false;
    bool data_item_restored = false;
    try {
        if (plan.remove_requested_tinfo) {
            del_tinfo(ea);
        }
        if (plan.remove_temporary_data_item && metadata.preparation_size != 0) {
            (void)del_items(ea, DELIT_SIMPLE, metadata.preparation_size);
        }
        if (plan.restore_prior_tinfo) {
            (void)set_tinfo(ea, &prior_type);
        }

        tinfo_t restored_type;
        const bool has_restored_type =
            get_tinfo(&restored_type, ea) && !restored_type.empty();
        tinfo_restored = had_prior_type
            ? has_restored_type && restored_type.equals_to(prior_type)
            : !has_restored_type;

        const ea_t restored_head = get_item_head(ea);
        const asize_t restored_size = restored_head == BADADDR
            ? 0
            : get_item_size(restored_head);
        data_item_restored =
            restored_head == metadata.item_head &&
            restored_size == metadata.item_size &&
            get_flags(ea) == metadata.address_flags &&
            (restored_head == BADADDR ||
             get_flags(restored_head) == metadata.head_flags);

        if (data_item_restored && plan.remove_temporary_data_item) {
            for (asize_t i = 0; i < metadata.preparation_size; ++i) {
                if (get_flags(ea + i) !=
                    metadata.preparation_flags[static_cast<std::size_t>(i)]) {
                    data_item_restored = false;
                    break;
                }
            }
        }
    } catch (...) {
        tinfo_restored = false;
        data_item_restored = false;
    }

    if (!transaction.finish_rollback_verification(
            tinfo_restored, data_item_restored)) {
        msg("Structor: CRITICAL: failed to restore global type state at "
            "0x%llX\n", static_cast<unsigned long long>(ea));
        return GlobalTinfoApplyResult::RollbackFailed;
    }
    return GlobalTinfoApplyResult::FailedRestored;
#else
    (void)verifier;
    // The mock environment has no persistent IDB type store to read back.
    // Without verification, success cannot be reported.
    return GlobalTinfoApplyResult::FailedRestored;
#endif
}

[[nodiscard]] inline bool apply_global_tinfo_transaction(
    ea_t ea,
    const tinfo_t& type,
    GlobalTinfoReadbackVerifier verifier)
{
    return apply_global_tinfo_transaction_detailed(ea, type, verifier) ==
        GlobalTinfoApplyResult::Applied;
}

} // namespace structor::detail

namespace structor {

[[nodiscard]] inline detail::GlobalTinfoApplyResult
apply_global_tinfo_detailed(ea_t ea, const tinfo_t& type) {
    return detail::apply_global_tinfo_transaction_detailed(
        ea, type, detail::exact_global_tinfo_readback);
}

[[nodiscard]] inline bool apply_global_tinfo(ea_t ea, const tinfo_t& type) {
    return apply_global_tinfo_detailed(ea, type) ==
        detail::GlobalTinfoApplyResult::Applied;
}

} // namespace structor
