#ifndef STRUCTOR_OPTIMIZED_ALGORITHMS_HPP
#define STRUCTOR_OPTIMIZED_ALGORITHMS_HPP

/**
 * @file optimized_algorithms.hpp
 * @brief High-performance algorithms for Structor
 * 
 * Replaces naive implementations with:
 * - SIMD-accelerated GCD computation
 * - O(n log n) overlap detection via sweep line
 * - Cache-optimized sorting and searching
 * - Batch operations for constraint building
 */

#include "structor/simd.hpp"
#include "structor/optimized_containers.hpp"
#include <algorithm>
#include <future>
#include <thread>
#include <vector>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace structor {
namespace algorithms {

inline size_t default_parallel_worker_count(
    size_t item_count,
    size_t min_items_per_worker = 256) noexcept
{
    min_items_per_worker = std::max<size_t>(1, min_items_per_worker);

    if (item_count < min_items_per_worker * 2) {
        return 1;
    }

    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) {
        hw_threads = 2;
    }

    const size_t useful_threads =
        (item_count + min_items_per_worker - 1) / min_items_per_worker;
    return std::max<size_t>(1, std::min<size_t>(hw_threads, useful_threads));
}

template<typename Fn>
void parallel_for_chunks(
    size_t item_count,
    size_t min_items_per_worker,
    Fn&& fn)
{
    const size_t workers = default_parallel_worker_count(item_count, min_items_per_worker);
    if (workers <= 1) {
        fn(size_t{0}, item_count);
        return;
    }

    const size_t chunk_size = (item_count + workers - 1) / workers;
    std::vector<std::future<void>> tasks;
    tasks.reserve(workers - 1);

    size_t start = 0;
    for (size_t worker = 0; worker + 1 < workers && start < item_count; ++worker) {
        const size_t end = std::min(start + chunk_size, item_count);
        tasks.push_back(std::async(std::launch::async, [&fn, start, end]() {
            fn(start, end);
        }));
        start = end;
    }

    if (start < item_count) {
        fn(start, item_count);
    }

    for (auto& task : tasks) {
        task.get();
    }
}

// ============================================================================
// SIMD-Accelerated GCD Computation
// ============================================================================

/**
 * @brief Binary GCD algorithm (Stein's algorithm)
 * 
 * More efficient than Euclidean algorithm, especially for large numbers.
 * Uses only subtraction and shifts (no division).
 */
STRUCTOR_ALWAYS_INLINE uint32_t binary_gcd(uint32_t u, uint32_t v) noexcept {
    if (u == 0) return v;
    if (v == 0) return u;
    
    // Find common factors of 2
    int shift = 0;
    while (((u | v) & 1) == 0) {
        u >>= 1;
        v >>= 1;
        ++shift;
    }
    
    // Remove remaining factors of 2 from u
    while ((u & 1) == 0) u >>= 1;
    
    do {
        // Remove factors of 2 from v
        while ((v & 1) == 0) v >>= 1;
        
        // Ensure u <= v
        if (u > v) {
            uint32_t t = u;
            u = v;
            v = t;
        }
        
        v -= u;
    } while (v != 0);
    
    return u << shift;
}

/**
 * @brief Compute GCD of array with early termination
 * 
 * Optimizations:
 * - Early termination when GCD becomes 1
 * - Prefetching for sequential access
 * - Branch prediction hints
 */
template<typename Container>
STRUCTOR_ALWAYS_INLINE uint32_t gcd_array(const Container& values) noexcept {
    if (values.empty()) return 0;
    if (values.size() == 1) return values[0];
    
    uint32_t result = values[0];
    
    for (size_t i = 1; i < values.size(); ++i) {
        // Prefetch next value
        if (STRUCTOR_LIKELY(i + 4 < values.size())) {
            STRUCTOR_PREFETCH_READ(&values[i + 4]);
        }
        
        result = binary_gcd(result, static_cast<uint32_t>(values[i]));
        
        // Early termination - GCD of 1 is final
        if (STRUCTOR_UNLIKELY(result == 1)) {
            return 1;
        }
    }
    
    return result;
}

/**
 * @brief Batch GCD computation for multiple pairs
 * 
 * When computing GCDs for many pairs (e.g., all offset differences),
 * batch processing is more cache-efficient.
 */
inline void batch_gcd(
    const uint32_t* STRUCTOR_RESTRICT a,
    const uint32_t* STRUCTOR_RESTRICT b,
    uint32_t* STRUCTOR_RESTRICT results,
    size_t count) noexcept
{
    // Prefetch first cache lines
    simd::prefetch_range(a, std::min(count * sizeof(uint32_t), simd::kCacheLine * 4));
    simd::prefetch_range(b, std::min(count * sizeof(uint32_t), simd::kCacheLine * 4));
    
    for (size_t i = 0; i < count; ++i) {
        results[i] = binary_gcd(a[i], b[i]);
    }
}

// ============================================================================
// O(n log n) Overlap Detection via Sweep Line
// ============================================================================

/**
 * @brief Interval with associated ID
 */
struct Interval {
    int64_t start;
    int64_t end;
    int32_t id;
    
    Interval() : start(0), end(0), id(-1) {}
    Interval(int64_t s, int64_t e, int32_t i) : start(s), end(e), id(i) {}

    [[nodiscard]] bool empty() const noexcept {
        return start >= end;
    }
    
    bool overlaps(const Interval& other) const noexcept {
        if (empty() || other.empty()) return false;
        return start < other.end && other.start < end;
    }
};

/**
 * @brief Find all overlapping interval pairs using sweep line algorithm
 * 
 * Time: O(n log n + k) where k is number of overlapping pairs
 * Space: O(n)
 * 
 * Much faster than O(n²) pairwise comparison for large n with few overlaps.
 */
inline std::vector<std::pair<int32_t, int32_t>> find_overlapping_pairs(
    const std::vector<Interval>& intervals)
{
    std::vector<std::pair<int32_t, int32_t>> result;
    
    if (intervals.size() < 2) return result;
    
    // Events refer to interval indices rather than IDs. IDs are caller-facing
    // labels and are not required for removing the exact active interval.
    struct Event {
        int64_t point;
        bool is_start;
        size_t interval_index;
        
        bool operator<(const Event& other) const noexcept {
            if (point != other.point) return point < other.point;
            // Intervals are half-open [start, end), so an interval ending at a
            // point must be removed before another interval starts there.
            if (is_start != other.is_start) return !is_start;
            return interval_index < other.interval_index;
        }
    };
    
    // Build event list
    std::vector<Event> events;
    events.reserve(intervals.size() * 2);
    
    for (size_t i = 0; i < intervals.size(); ++i) {
        const auto& iv = intervals[i];
        if (iv.empty()) {
            continue;
        }
        events.push_back({iv.start, true, i});
        events.push_back({iv.end, false, i});
    }
    
    // Sort events
    std::sort(events.begin(), events.end());
    
    // Sweep line - track active intervals
    SmallVector<size_t, 16> active;
    
    for (const auto& event : events) {
        if (event.is_start) {
            // Start event: report overlaps with all currently active
            const int32_t event_id = intervals[event.interval_index].id;
            for (size_t active_index : active) {
                const int32_t active_id = intervals[active_index].id;
                result.emplace_back(
                    std::min(active_id, event_id),
                    std::max(active_id, event_id)
                );
            }
            active.push_back(event.interval_index);
        } else {
            // End event: remove from active
            for (size_t i = 0; i < active.size(); ++i) {
                if (active[i] == event.interval_index) {
                    // Swap with last and pop
                    active[i] = active.back();
                    active.pop_back();
                    break;
                }
            }
        }
    }
    
    // Canonicalize result ordering for deterministic solver construction.
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    
    return result;
}

/**
 * @brief Optimized overlap detection for field candidates
 * 
 * Uses sweep line when n > threshold, falls back to O(n²) for small n
 * where constant factors dominate.
 */
template<typename CandidateContainer>
std::vector<std::pair<int32_t, int32_t>> detect_overlaps(
    const CandidateContainer& candidates,
    size_t threshold = 64)
{
    size_t n = candidates.size();
    
    if (n < 2) {
        return {};
    }
    if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("candidate index exceeds int32_t");
    }
    
    // For small n, O(n²) is faster due to lower constant factors
    if (n <= threshold) {
        std::vector<std::pair<int32_t, int32_t>> result;
        
        for (size_t i = 0; i < n; ++i) {
            const int64_t i_start =
                static_cast<int64_t>(candidates[i].offset);
            const auto i_size =
                static_cast<std::uint64_t>(candidates[i].size);
            if (i_size == 0 || i_size > static_cast<std::uint64_t>(
                    std::numeric_limits<int64_t>::max()) ||
                i_start > std::numeric_limits<int64_t>::max() -
                    static_cast<int64_t>(i_size)) {
                continue;
            }
            const int64_t i_end =
                i_start + static_cast<int64_t>(i_size);
            for (size_t j = i + 1; j < n; ++j) {
                const int64_t j_start =
                    static_cast<int64_t>(candidates[j].offset);
                const auto j_size =
                    static_cast<std::uint64_t>(candidates[j].size);
                if (j_size == 0 || j_size > static_cast<std::uint64_t>(
                        std::numeric_limits<int64_t>::max()) ||
                    j_start > std::numeric_limits<int64_t>::max() -
                        static_cast<int64_t>(j_size)) {
                    continue;
                }
                const int64_t j_end =
                    j_start + static_cast<int64_t>(j_size);
                if (i_start < j_end && j_start < i_end) {
                    result.emplace_back(
                        static_cast<int32_t>(i),
                        static_cast<int32_t>(j)
                    );
                }
            }
        }
        
        return result;
    }
    
    // For large n, use sweep line algorithm
    std::vector<Interval> intervals;
    intervals.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        const int64_t offset = static_cast<int64_t>(candidates[i].offset);
        const auto raw_size = static_cast<std::uint64_t>(candidates[i].size);
        if (raw_size > static_cast<std::uint64_t>(
                std::numeric_limits<int64_t>::max()) ||
            offset > std::numeric_limits<int64_t>::max() -
                static_cast<int64_t>(raw_size)) {
            intervals.emplace_back(
                offset, offset, static_cast<int32_t>(i));
            continue;
        }
        intervals.emplace_back(
            offset,
            offset + static_cast<int64_t>(raw_size),
            static_cast<int32_t>(i)
        );
    }
    
    return find_overlapping_pairs(intervals);
}

// ============================================================================
// Arithmetic Progression Detection
// ============================================================================

/**
 * @brief Check if sorted values form an arithmetic progression
 * 
 * Returns (base, stride) if AP detected, nullopt otherwise.
 */
template<typename T>
std::optional<std::pair<T, T>> detect_arithmetic_progression(
    const T* sorted_data,
    size_t count) noexcept
{
    if (count < 2) return std::nullopt;
    
    // First difference is the potential stride
    T stride = sorted_data[1] - sorted_data[0];
    if (stride <= 0) return std::nullopt;
    
    // Verify all subsequent differences match
    for (size_t i = 2; i < count; ++i) {
        T diff = sorted_data[i] - sorted_data[i - 1];
        if (diff != stride) {
            return std::nullopt;
        }
    }
    
    return std::make_pair(sorted_data[0], stride);
}

/**
 * @brief Find longest arithmetic progression subsequence
 * 
 * When data doesn't form a perfect AP, find the longest run that does.
 * Returns (start_index, length, stride).
 */
template<typename T>
std::tuple<size_t, size_t, T> find_longest_ap_run(
    const T* sorted_data,
    size_t count)
{
    if (count < 2) {
        return {0, count, T(0)};
    }
    
    size_t best_start = 0;
    size_t best_length = 1;
    T best_stride = 0;
    
    size_t run_start = 0;
    size_t run_length = 1;
    T run_stride = sorted_data[1] - sorted_data[0];
    
    for (size_t i = 1; i < count; ++i) {
        T diff = sorted_data[i] - sorted_data[i - 1];
        
        if (diff == run_stride && run_stride > 0) {
            ++run_length;
        } else {
            // End of current run
            if (run_length > best_length) {
                best_start = run_start;
                best_length = run_length;
                best_stride = run_stride;
            }
            
            // Start new run
            run_start = i - 1;
            run_length = 2;
            run_stride = diff;
        }
    }
    
    // Check final run
    if (run_length > best_length) {
        best_start = run_start;
        best_length = run_length;
        best_stride = run_stride;
    }
    
    return {best_start, best_length, best_stride};
}

// ============================================================================
// Batch Operations for Constraint Building
// ============================================================================

/**
 * @brief Batch check coverage: which accesses are covered by which candidates
 * 
 * Returns a bitvector for each access indicating covering candidates.
 * More cache-efficient than checking each access individually.
 */
template<typename AccessContainer, typename CandidateContainer>
std::vector<std::vector<int32_t>> compute_coverage_map(
    const AccessContainer& accesses,
    const CandidateContainer& candidates)
{
    std::vector<std::vector<int32_t>> coverage(accesses.size());
    if (candidates.size() >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("candidate index exceeds int32_t");
    }

    const auto checked_end = [](auto offset, auto raw_size) {
        using Offset = std::decay_t<decltype(offset)>;
        using Size = std::decay_t<decltype(raw_size)>;
        static_assert(std::is_integral_v<Offset> && std::is_integral_v<Size>);
        if constexpr (std::is_signed_v<Size>) {
            if (raw_size < 0) {
                return std::optional<Offset>{};
            }
        }
        const auto size = static_cast<std::uintmax_t>(raw_size);
        if (size > static_cast<std::uintmax_t>(
                std::numeric_limits<Offset>::max())) {
            return std::optional<Offset>{};
        }
        const Offset delta = static_cast<Offset>(size);
        if (offset > std::numeric_limits<Offset>::max() - delta) {
            return std::optional<Offset>{};
        }
        return std::optional<Offset>{static_cast<Offset>(offset + delta)};
    };

    // Linear scanning is intentional: candidates are not monotone by end
    // offset when ordered by start, so a lower_bound on end can skip an early
    // long interval. Complexity is O(A*C) time and O(A*C) output space.
    for (size_t ai = 0; ai < accesses.size(); ++ai) {
        const auto& access = accesses[ai];
        const auto access_end = checked_end(access.offset, access.size);
        if (!access_end.has_value() || access.size == 0) {
            continue;
        }
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            const auto& cand = candidates[ci];
            const auto cand_end = checked_end(cand.offset, cand.size);
            if (!cand_end.has_value() || cand.size == 0) {
                continue;
            }
            if (cand.offset <= access.offset && *cand_end >= *access_end) {
                coverage[ai].push_back(static_cast<int32_t>(ci));
            }
        }
    }
    
    return coverage;
}

// ============================================================================
// Radix Sort for Integers (faster than std::sort for many elements)
// ============================================================================

/**
 * @brief Radix sort for 32-bit integers
 * 
 * O(n) time complexity, faster than O(n log n) comparison sort for n > ~1000
 */
inline void radix_sort_u32(uint32_t* data, size_t count, uint32_t* temp) {
    constexpr int kRadix = 8;
    constexpr int kBuckets = 256;
    
    for (int shift = 0; shift < 32; shift += kRadix) {
        size_t buckets[kBuckets] = {0};
        
        // Count occurrences
        for (size_t i = 0; i < count; ++i) {
            ++buckets[(data[i] >> shift) & 0xFF];
        }
        
        // Compute prefix sums
        size_t total = 0;
        for (int b = 0; b < kBuckets; ++b) {
            size_t old_count = buckets[b];
            buckets[b] = total;
            total += old_count;
        }
        
        // Place elements
        for (size_t i = 0; i < count; ++i) {
            uint32_t bucket = (data[i] >> shift) & 0xFF;
            temp[buckets[bucket]++] = data[i];
        }
        
        // Swap buffers
        std::swap(data, temp);
    }
    
    // If odd number of passes, copy back
    // (4 passes for 32-bit, so no copy needed)
}

// ============================================================================
// Parallel-Friendly Partitioning
// ============================================================================

/**
 * @brief Partition data into chunks suitable for parallel processing
 * 
 * Each chunk is cache-line aligned and has similar workload.
 */
template<typename T>
std::vector<std::pair<size_t, size_t>> partition_for_parallel(
    const T* data,
    size_t count,
    size_t num_threads)
{
    std::vector<std::pair<size_t, size_t>> partitions;
    
    if (count == 0 || num_threads == 0) return partitions;
    
    // Calculate elements per thread, rounded to cache line
    size_t elements_per_line = simd::kCacheLine / sizeof(T);
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    chunk_size = std::max(chunk_size, elements_per_line);
    
    for (size_t start = 0; start < count; start += chunk_size) {
        size_t end = std::min(start + chunk_size, count);
        partitions.emplace_back(start, end);
    }
    
    return partitions;
}

} // namespace algorithms
} // namespace structor

#endif // STRUCTOR_OPTIMIZED_ALGORITHMS_HPP
