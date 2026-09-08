#pragma once

#include <z3++.h>
#include <optional>
#include <limits>
#ifdef STRUCTOR_TESTING
#include "mock_ida.hpp"
#endif
#include "structor/synth_types.hpp"
#include "structor/z3/context.hpp"
#include "structor/z3/type_encoding.hpp"
#include "structor/optimized_algorithms.hpp"

namespace structor::z3 {

/// Owned identity of an observation supporting an array view. Keeping the
/// original type (including promoted expression types) allows the candidate
/// generator to recover its source indices without borrowing input pointers.
struct ArrayAccessEvidence {
    sval_t offset = 0;
    uint32_t size = 0;
    tinfo_t observed_type;
    SemanticType semantic = SemanticType::Unknown;

    [[nodiscard]] bool matches(const FieldAccess& access) const {
        return offset == access.offset && size == access.size &&
            semantic == access.semantic_type &&
            (observed_type.empty() ? access.inferred_type.empty()
                                  : observed_type.equals_to(access.inferred_type));
    }
};

/// Represents a detected array pattern
struct ArrayCandidate {
    sval_t base_offset;           // Starting offset of array
    uint32_t stride;              // Bytes between elements (== sizeof(element))
    uint32_t element_count;       // Number of elements
    tinfo_t element_type;         // Type of each element
    qvector<sval_t> member_offsets;  // Original offsets covered by this array
    qvector<ArrayAccessEvidence> member_evidence;  // Sorted by byte offset

    // For stride > access_size cases: synthetic element struct
    bool needs_element_struct = false;
    uint32_t inner_access_offset = 0;  // Offset of accessed subfield within element
    uint32_t inner_access_size = 0;    // Size of the inner access

    // Confidence in the detection
    TypeConfidence confidence = TypeConfidence::Medium;

    ArrayCandidate()
        : base_offset(0)
        , stride(0)
        , element_count(0) {}

    /// C array semantics check: stride must equal element size
    [[nodiscard]] bool is_valid_c_array() const noexcept {
        if (element_type.empty()) return false;
        return stride == element_type.get_size();
    }

    /// Calculate total array size in bytes
    [[nodiscard]] std::optional<uint32_t> checked_total_size() const noexcept {
        if (stride == 0 || element_count == 0) {
            return std::nullopt;
        }
        return checked_u32_product(stride, element_count);
    }

    [[nodiscard]] uint32_t total_size() const noexcept {
        return checked_total_size().value_or(std::numeric_limits<uint32_t>::max());
    }

    [[nodiscard]] std::optional<sval_t> checked_end_offset() const noexcept {
        const auto total = checked_total_size();
        if (!total) {
            return std::nullopt;
        }
        return checked_interval_end(base_offset, *total);
    }

    /// Check if an offset falls within this array
    [[nodiscard]] bool contains_offset(sval_t offset) const noexcept {
        const auto end = checked_end_offset();
        return end.has_value() && offset >= base_offset && offset < *end;
    }

    /// Match only observations belonging to this typed view, not other types
    /// observed at the same address. O(log A+V) for A keys and V views at the
    /// queried offset. The keys remain valid after the input vector is gone.
    [[nodiscard]] bool has_member_evidence(const FieldAccess& access) const {
        auto it = std::lower_bound(member_evidence.begin(), member_evidence.end(), access.offset,
            [](const auto& evidence, sval_t offset) { return evidence.offset < offset; });
        for (; it != member_evidence.end() && it->offset == access.offset; ++it) {
            if (it->matches(access)) return true;
        }
        return false;
    }

    /// Get the element index for an offset
    [[nodiscard]] std::optional<uint32_t> get_element_index(sval_t offset) const noexcept {
        if (stride == 0 || !contains_offset(offset)) return std::nullopt;
        const auto relative = checked_interval_span(base_offset, offset);
        if (!relative ||
            *relative % stride != static_cast<std::uint64_t>(inner_access_offset)) {
            return std::nullopt;  // Not at the expected inner offset
        }
        const std::uint64_t index = *relative / stride;
        if (index >= element_count ||
            index > std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(index);
    }

    /// Generate description string
    [[nodiscard]] qstring description() const {
        qstring desc;
        desc.sprnt("Array[%u] at 0x%llX, stride=%u",
                  element_count,
                  static_cast<unsigned long long>(base_offset),
                  stride);
        if (needs_element_struct) {
            desc.cat_sprnt(" (element struct, inner@0x%X)", inner_access_offset);
        }
        return desc;
    }
};

/// Configuration for array detection
struct ArrayDetectionConfig {
    uint32_t min_elements = 3;               // Minimum elements to form an array
    int max_gap_ratio = 2;                   // Max allowed gap as multiple of stride
    bool require_consistent_types = true;    // All elements must have same type
    bool detect_arrays_of_structs = true;    // When stride > access_size, create element struct
    bool use_symbolic_indices = true;        // Use Z3 for affine index detection
    uint32_t max_stride = 4096;              // Maximum allowed stride
    uint32_t max_elements = 10000;           // Maximum array elements to consider
};

/// Construct the production array-detection configuration from the synthesis
/// controls without silently reverting symbolic/stride bounds to defaults.
[[nodiscard]] inline ArrayDetectionConfig
make_array_detection_config(
    uint32_t min_elements,
    uint32_t max_elements,
    bool use_symbolic_indices,
    uint32_t max_stride) noexcept
{
    ArrayDetectionConfig result;
    result.min_elements = min_elements;
    result.max_elements = max_elements;
    result.use_symbolic_indices = use_symbolic_indices;
    result.max_stride = max_stride;
    return result;
}

/// Builds Z3 constraints for array pattern detection
class ArrayConstraintBuilder {
public:
    ArrayConstraintBuilder(Z3Context& ctx, const ArrayDetectionConfig& config = {});

    /// Analyze accesses and detect array patterns
    /// Returns all detected array candidates, sorted by base offset
    [[nodiscard]] qvector<ArrayCandidate> detect_arrays(
        const qvector<FieldAccess>& accesses
    );

    /// Build Z3 constraint for affine index detection
    /// Solves: offset[i] = base + index[i] * stride + inner_offset
    /// where index[i] are unknown integers
    [[nodiscard]] std::optional<ArrayCandidate> detect_symbolic_array(
        const qvector<const FieldAccess*>& accesses
    );

    /// Handle stride > access_size: create element struct type
    [[nodiscard]] tinfo_t create_element_struct_type(
        uint32_t stride,
        uint32_t inner_offset,
        const tinfo_t& inner_type
    );

    /// Use Z3 to find optimal stride when simple AP detection fails
    [[nodiscard]] std::optional<ArrayCandidate> solve_stride_z3(
        const qvector<const FieldAccess*>& accesses
    );

    /// Get statistics about last detection
    struct DetectionStats {
        int arrays_found = 0;
        int elements_covered = 0;
        int symbolic_detections = 0;
        int struct_element_arrays = 0;
    };
    [[nodiscard]] const DetectionStats& stats() const noexcept { return stats_; }

private:
    Z3Context& ctx_;
    ArrayDetectionConfig config_;
    DetectionStats stats_;

    /// Pre-filter: group accesses by size (potential array elements)
    [[nodiscard]] std::unordered_map<uint32_t, qvector<const FieldAccess*>>
    group_by_size(const qvector<FieldAccess>& accesses);

    /// Fast path: check if offsets form perfect arithmetic progression
    [[nodiscard]] std::optional<std::pair<sval_t, uint32_t>>  // (base, stride)
    find_arithmetic_progression(const qvector<sval_t>& offsets);

    /// Extract a consistent stride hint from accesses
    [[nodiscard]] std::optional<uint32_t>
    extract_stride_hint(const qvector<const FieldAccess*>& accesses) const;

    /// Use a stride hint to validate a progression
    [[nodiscard]] std::optional<std::pair<sval_t, uint32_t>>  // (base, stride)
    find_progression_with_stride(
        const qvector<sval_t>& offsets,
        const qvector<const FieldAccess*>& accesses,
        uint32_t stride_hint) const;

    /// Verify type consistency for potential array elements
    [[nodiscard]] bool verify_type_consistency(
        const qvector<const FieldAccess*>& accesses
    );

    /// Interpret concrete type evidence at the observed storage width. An
    /// empty result denotes an observation without type or semantic evidence.
    [[nodiscard]] tinfo_t normalized_element_type(const FieldAccess& access);

    /// Recover exact-stride runs for each distinct C storage type. Unknown
    /// observations are a separate class, so they cannot bridge conflicting
    /// typed runs. O(A*T + A log A) time and O(A+T) space for A observations
    /// and T distinct normalized types; type comparisons are SDK operations.
    [[nodiscard]] qvector<ArrayCandidate> detect_contiguous_typed_runs(
        const qvector<const FieldAccess*>& accesses,
        uint32_t stride);

    /// Merge compatible array candidates (overlapping ranges)
    void merge_overlapping_arrays(qvector<ArrayCandidate>& candidates);

    /// Create a bounded C array covering each complete observation. Reject
    /// overlapping elements, insufficient distinct observations, and invalid
    /// type/extent representations. O(A log A) time and O(A) space for A
    /// accesses, including deterministic ordering of owned evidence.
    [[nodiscard]] std::optional<ArrayCandidate> create_candidate(
        sval_t base,
        uint32_t stride,
        uint32_t count,
        const qvector<const FieldAccess*>& accesses
    );

    /// Calculate GCD of all offset differences (for stride detection)
    [[nodiscard]] uint32_t calculate_gcd_stride(const qvector<sval_t>& offsets) const;

    /// Check if accesses could form struct-of-arrays pattern
    [[nodiscard]] bool check_struct_element_pattern(
        const qvector<const FieldAccess*>& accesses,
        uint32_t stride,
        uint32_t& inner_offset
    ) const;
};

/// Quick check: can these offsets possibly form an array?
[[nodiscard]] inline bool could_be_array(
    const qvector<sval_t>& offsets,
    uint32_t element_size,
    int min_elements = 3)
{
    if (static_cast<int>(offsets.size()) < min_elements) {
        return false;
    }

    // Check if offsets could form arithmetic progression with stride >= element_size
    if (offsets.size() < 2) return false;

    // Sort offsets
    qvector<sval_t> sorted = offsets;
    std::sort(sorted.begin(), sorted.end());

    // Check first difference as potential stride
    sval_t potential_stride = sorted[1] - sorted[0];
    if (potential_stride <= 0 || potential_stride < static_cast<sval_t>(element_size)) {
        return false;
    }

    // Quick check: all differences should be multiples of potential stride
    for (size_t i = 1; i < sorted.size(); ++i) {
        sval_t diff = sorted[i] - sorted[0];
        if (diff % potential_stride != 0) {
            return false;
        }
    }

    return true;
}

/// Find the GCD of two integers (optimized binary GCD / Stein's algorithm)
[[nodiscard]] inline uint32_t gcd(uint32_t a, uint32_t b) noexcept {
    return algorithms::binary_gcd(a, b);
}

/// Calculate the GCD of a vector of values (optimized with early termination)
[[nodiscard]] inline uint32_t gcd_vector(const qvector<uint32_t>& values) noexcept {
    return algorithms::gcd_array(values);
}

} // namespace structor::z3
