#include <gtest/gtest.h>

#include <array>
#include <limits>

#include <structor/persistence_invariants.hpp>
#include <structor/persistence_transaction.hpp>

namespace structor::test {

using persistence_invariants::ida_udt_pack_code;
using persistence_invariants::unique_best_score_index;
using persistence_invariants::PersistenceTransactionJournal;

TEST(PersistenceInvariantsTest, MapsBytePackingToIdaUdtCodes) {
    EXPECT_EQ(ida_udt_pack_code(std::nullopt), std::optional<std::uint8_t>{0});
    EXPECT_EQ(ida_udt_pack_code(1), std::optional<std::uint8_t>{1});
    EXPECT_EQ(ida_udt_pack_code(2), std::optional<std::uint8_t>{2});
    EXPECT_EQ(ida_udt_pack_code(4), std::optional<std::uint8_t>{3});
    EXPECT_EQ(ida_udt_pack_code(8), std::optional<std::uint8_t>{4});
    EXPECT_EQ(ida_udt_pack_code(16), std::optional<std::uint8_t>{5});
}

TEST(PersistenceInvariantsTest, RejectsUnsupportedPackingCaps) {
    EXPECT_FALSE(ida_udt_pack_code(0).has_value());
    EXPECT_FALSE(ida_udt_pack_code(3).has_value());
    EXPECT_FALSE(ida_udt_pack_code(32).has_value());
}

TEST(PersistenceInvariantsTest, SelectsOnlyAUniqueBestReuseScore) {
    const std::array<double, 3> scores = {0.85, 0.91, 0.87};
    EXPECT_EQ(unique_best_score_index(scores, 0.85),
              std::optional<std::size_t>{1});
}

TEST(PersistenceInvariantsTest, RejectsAmbiguousReuseScoreTie) {
    const std::array<double, 3> scores = {0.91, 0.87, 0.91};
    EXPECT_FALSE(unique_best_score_index(scores, 0.85).has_value());

    const std::array<double, 2> near_tie = {0.91, 0.91 + 5.0e-13};
    EXPECT_FALSE(unique_best_score_index(near_tie, 0.85).has_value());
}

TEST(PersistenceInvariantsTest, RejectsNonFiniteAndBelowThresholdScores) {
    const std::array<double, 3> scores = {
        0.70, std::numeric_limits<double>::quiet_NaN(), 0.84};
    EXPECT_FALSE(unique_best_score_index(scores, 0.85).has_value());
    const std::array<double, 1> just_below = {0.85 - 5.0e-13};
    EXPECT_FALSE(unique_best_score_index(just_below, 0.85).has_value());
    EXPECT_FALSE(unique_best_score_index(
        scores, std::numeric_limits<double>::infinity()).has_value());
}

TEST(PersistenceInvariantsTest, JournalPreservesFirstTouchedState) {
    PersistenceTransactionJournal journal;
    const auto first = journal.stage("root", true, 11);
    const auto repeated = journal.stage("root", false, 0);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first, repeated);
    ASSERT_NE(journal.entry(*first), nullptr);
    EXPECT_TRUE(journal.entry(*first)->existed_before);
    EXPECT_EQ(journal.entry(*first)->original_identity, 11U);
}

TEST(PersistenceInvariantsTest, JournalRollsBackMutationsInReverseOrder) {
    PersistenceTransactionJournal journal;
    const auto child = journal.stage("child", false);
    const auto root = journal.stage("root", false);
    ASSERT_TRUE(child.has_value());
    ASSERT_TRUE(root.has_value());
    EXPECT_TRUE(journal.mark_mutated(*child, 101));
    EXPECT_TRUE(journal.mark_mutated(*root, 102));
    EXPECT_EQ(journal.rollback_order(),
              (std::vector<std::size_t>{*root, *child}));
}

TEST(PersistenceInvariantsTest, JournalOmitsWritesThatNeverMutatedState) {
    PersistenceTransactionJournal journal;
    const auto failed = journal.stage("failed", false);
    const auto replaced = journal.stage("replaced", true, 20);
    ASSERT_TRUE(failed.has_value());
    ASSERT_TRUE(replaced.has_value());
    EXPECT_TRUE(journal.mark_mutated(*replaced, 21));
    EXPECT_EQ(journal.rollback_order(),
              (std::vector<std::size_t>{*replaced}));
}

TEST(PersistenceInvariantsTest, JournalRejectsInvalidNamesAndIdentities) {
    PersistenceTransactionJournal journal;
    EXPECT_FALSE(journal.stage("", false).has_value());
    const auto valid = journal.stage("valid", false);
    ASSERT_TRUE(valid.has_value());
    EXPECT_FALSE(journal.mark_mutated(*valid, 0));
    EXPECT_FALSE(journal.mark_mutated(99, 1));
    EXPECT_TRUE(journal.rollback_order().empty());
}

} // namespace structor::test
