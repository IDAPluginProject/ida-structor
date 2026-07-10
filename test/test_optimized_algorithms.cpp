/// @file test_optimized_algorithms.cpp
/// @brief Correctness tests for performance-sensitive shared algorithms

#include <gtest/gtest.h>

#include <structor/optimized_algorithms.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace structor::test {
namespace {

using algorithms::Interval;
using Pair = std::pair<int32_t, int32_t>;

std::vector<Pair> reference_overlaps(const std::vector<Interval>& intervals) {
    std::vector<Pair> result;
    for (size_t i = 0; i < intervals.size(); ++i) {
        for (size_t j = i + 1; j < intervals.size(); ++j) {
            if (!intervals[i].overlaps(intervals[j])) {
                continue;
            }

            result.emplace_back(
                std::min(intervals[i].id, intervals[j].id),
                std::max(intervals[i].id, intervals[j].id));
        }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

TEST(OptimizedAlgorithmsTest, HalfOpenIntervalsDoNotOverlapAtBoundary) {
    const Interval left(0, 4, 10);
    const Interval right(4, 8, 11);
    const Interval spanning(3, 5, 12);

    EXPECT_FALSE(left.overlaps(right));
    EXPECT_TRUE(left.overlaps(spanning));
    EXPECT_TRUE(right.overlaps(spanning));

    const std::vector<Interval> intervals{left, right, spanning};
    const std::vector<Pair> expected{{10, 12}, {11, 12}};
    EXPECT_EQ(algorithms::find_overlapping_pairs(intervals), expected);
}

TEST(OptimizedAlgorithmsTest, EmptyAndReversedIntervalsNeverOverlap) {
    const Interval non_empty(-4, 4, 1);
    const Interval empty(0, 0, 2);
    const Interval reversed(3, -3, 3);

    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(reversed.empty());
    EXPECT_FALSE(non_empty.empty());
    EXPECT_FALSE(non_empty.overlaps(empty));
    EXPECT_FALSE(empty.overlaps(non_empty));
    EXPECT_FALSE(non_empty.overlaps(reversed));

    const std::vector<Interval> intervals{non_empty, empty, reversed};
    EXPECT_TRUE(algorithms::find_overlapping_pairs(intervals).empty());
}

TEST(OptimizedAlgorithmsTest, EqualEndpointsAndNegativeOffsetsAreDeterministic) {
    const std::vector<Interval> intervals{
        {-8, 0, 9},
        {-8, 0, 3},
        {0, 8, 7},
        {-4, 4, 5},
        {8, 12, 1},
    };

    const auto expected = reference_overlaps(intervals);
    const auto first = algorithms::find_overlapping_pairs(intervals);
    const auto second = algorithms::find_overlapping_pairs(intervals);
    EXPECT_EQ(first, expected);
    EXPECT_EQ(second, expected);
}

TEST(OptimizedAlgorithmsTest, SweepLineMatchesPairwiseReference) {
    std::mt19937_64 rng(0x5354525543544F52ULL);
    std::uniform_int_distribution<int64_t> start_dist(-64, 64);
    std::uniform_int_distribution<int64_t> length_dist(-2, 16);

    // More than 64 intervals exercises the path used by large solver models.
    constexpr size_t kIntervalCount = 129;
    constexpr size_t kTrials = 128;
    for (size_t trial = 0; trial < kTrials; ++trial) {
        std::vector<Interval> intervals;
        intervals.reserve(kIntervalCount);
        for (size_t i = 0; i < kIntervalCount; ++i) {
            const int64_t start = start_dist(rng);
            const int64_t end = start + length_dist(rng);
            intervals.emplace_back(start, end, static_cast<int32_t>(i));
        }

        EXPECT_EQ(
            algorithms::find_overlapping_pairs(intervals),
            reference_overlaps(intervals))
            << "trial=" << trial;
    }
}

struct TestCandidate {
    int64_t offset = 0;
    uint32_t size = 0;

    [[nodiscard]] bool overlaps(const TestCandidate& other) const noexcept {
        return offset < other.offset + static_cast<int64_t>(other.size) &&
               other.offset < offset + static_cast<int64_t>(size);
    }
};

TEST(OptimizedAlgorithmsTest, DetectOverlapsIsThresholdInvariant) {
    std::vector<TestCandidate> candidates;
    candidates.reserve(96);
    for (int64_t offset = 0; offset < 96 * 4; offset += 4) {
        candidates.push_back({offset, 4});
    }

    // Force the pairwise and sweep-line implementations over identical input.
    const auto pairwise = algorithms::detect_overlaps(candidates, candidates.size());
    const auto sweep_line = algorithms::detect_overlaps(candidates, 0);
    EXPECT_TRUE(pairwise.empty());
    EXPECT_EQ(sweep_line, pairwise);
}

TEST(OptimizedAlgorithmsTest, CoverageMapDoesNotSkipEarlierLongCandidate) {
    const std::vector<TestCandidate> accesses{{10, 2}};
    const std::vector<TestCandidate> candidates{
        {0, 100},
        {9, 1},
        {10, 2},
        {11, 4},
    };

    const auto coverage =
        algorithms::compute_coverage_map(accesses, candidates);
    ASSERT_EQ(coverage.size(), 1U);
    EXPECT_EQ(coverage[0], (std::vector<int32_t>{0, 2}));
}

TEST(OptimizedAlgorithmsTest, CoverageHelpersRejectOverflowAndEmptyIntervals) {
    constexpr int64_t max = std::numeric_limits<int64_t>::max();
    const std::vector<TestCandidate> accesses{{max, 1}, {4, 2}, {4, 0}};
    const std::vector<TestCandidate> candidates{{0, 100}, {4, 2}, {max, 1}};

    const auto generic =
        algorithms::compute_coverage_map(accesses, candidates);
    ASSERT_EQ(generic.size(), 3U);
    EXPECT_TRUE(generic[0].empty());
    EXPECT_EQ(generic[1], (std::vector<int32_t>{0, 1}));
    EXPECT_TRUE(generic[2].empty());

    const int64_t access_offsets[] = {10, max, 5};
    const uint32_t access_sizes[] = {2, 1, 0};
    const int64_t candidate_offsets[] = {0, 9, 10, 11, max};
    const uint32_t candidate_sizes[] = {100, 1, 2, 4, 1};
    std::vector<std::vector<uint32_t>> simd_coverage;
    simd::batch_coverage_check(
        access_offsets, access_sizes, 3,
        candidate_offsets, candidate_sizes, 5,
        simd_coverage);
    ASSERT_EQ(simd_coverage.size(), 3U);
    EXPECT_EQ(simd_coverage[0], (std::vector<uint32_t>{0, 2}));
    EXPECT_TRUE(simd_coverage[1].empty());
    EXPECT_TRUE(simd_coverage[2].empty());

    const int64_t overlap_offsets[] = {0, 4, max};
    const uint32_t overlap_sizes[] = {8, 4, 1};
    EXPECT_EQ(
        simd::batch_overlap_small(overlap_offsets, overlap_sizes, 3),
        (std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}}));

    const std::vector<TestCandidate> sweep_candidates{
        {0, 8}, {4, 4}, {max, 1}, {12, 0},
    };
    EXPECT_EQ(
        algorithms::detect_overlaps(sweep_candidates, 0),
        (std::vector<std::pair<int32_t, int32_t>>{{0, 1}}));
}

} // namespace
} // namespace structor::test
