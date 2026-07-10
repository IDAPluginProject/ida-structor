#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace structor::persistence_invariants {

/// Convert a recovered byte packing cap to IDA's logarithmic UDT pack code.
/// IDA uses 0 for the ABI default and 1..5 for pack(1)..pack(16).
[[nodiscard]] inline constexpr std::optional<std::uint8_t> ida_udt_pack_code(
    const std::optional<std::uint32_t>& packing) noexcept {
    if (!packing.has_value()) {
        return std::uint8_t{0};
    }

    switch (*packing) {
        case 1:  return std::uint8_t{1};
        case 2:  return std::uint8_t{2};
        case 4:  return std::uint8_t{3};
        case 8:  return std::uint8_t{4};
        case 16: return std::uint8_t{5};
        default: return std::nullopt;
    }
}

/// Return the unique highest score at or above threshold. An exact or
/// numerically indistinguishable tie is deliberately ambiguous and therefore
/// has no winner. Complexity: O(n) time and O(1) space.
[[nodiscard]] inline std::optional<std::size_t> unique_best_score_index(
    std::span<const double> scores,
    double threshold,
    double epsilon = 1.0e-12) noexcept {
    if (!std::isfinite(threshold) || !std::isfinite(epsilon) || epsilon < 0.0) {
        return std::nullopt;
    }

    std::optional<std::size_t> best;
    double best_score = threshold;
    bool ambiguous = false;

    for (std::size_t i = 0; i < scores.size(); ++i) {
        const double score = scores[i];
        if (!std::isfinite(score) || score < threshold) {
            continue;
        }

        if (!best.has_value() || score > best_score + epsilon) {
            best = i;
            best_score = score;
            ambiguous = false;
        } else if (std::fabs(score - best_score) <= epsilon) {
            ambiguous = true;
        }
    }

    return best.has_value() && !ambiguous ? best : std::nullopt;
}

} // namespace structor::persistence_invariants
