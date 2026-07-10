#pragma once

namespace structor::detail {

/// Terminal lifecycle state for a callback registration owned by a host object.
/// Once shutdown begins, no later installation can make the registration live.
class HostHookLifecycle {
public:
    [[nodiscard]] bool can_install() const noexcept {
        return !shutdown_;
    }

    [[nodiscard]] bool is_hooked() const noexcept {
        return hooked_;
    }

    [[nodiscard]] bool is_shutdown() const noexcept {
        return shutdown_;
    }

    /// Records a completed host registration. A registration that completes
    /// after shutdown has started is rejected so the caller can remove it.
    [[nodiscard]] bool mark_installed() noexcept {
        if (shutdown_) {
            return false;
        }
        hooked_ = true;
        return true;
    }

    void mark_uninstalled() noexcept {
        hooked_ = false;
    }

    /// Transitions to the terminal state. Returns true exactly once.
    [[nodiscard]] bool begin_shutdown() noexcept {
        if (shutdown_) {
            return false;
        }
        shutdown_ = true;
        return true;
    }

private:
    bool hooked_ = false;
    bool shutdown_ = false;
};

} // namespace structor::detail
