#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "mock_ida.hpp"

#define __HEXRAYS_HPP
#define __TYPEINF_HPP
#define __PRO_H
#define __IDA_HPP
#define __IDP_HPP
#define __LOADER_HPP
#define __KERNWIN_HPP
#define __STRUCT_HPP
#define __ENUM_HPP
#define __NAME_HPP
#define __BYTES_HPP
#define __FUNCS_HPP
#define __XREF_HPP

#include <structor/global_rewrite_index.hpp>
#include <structor/global_tinfo_transaction.hpp>
#include <structor/host_lifecycle.hpp>

namespace structor::test {

TEST(HostHookLifecycleTest, ShutdownPermanentlyRejectsInstallation) {
    detail::HostHookLifecycle lifecycle;

    EXPECT_TRUE(lifecycle.can_install());
    EXPECT_TRUE(lifecycle.mark_installed());
    EXPECT_TRUE(lifecycle.is_hooked());

    EXPECT_TRUE(lifecycle.begin_shutdown());
    lifecycle.mark_uninstalled();

    EXPECT_TRUE(lifecycle.is_shutdown());
    EXPECT_FALSE(lifecycle.is_hooked());
    EXPECT_FALSE(lifecycle.can_install());
    EXPECT_FALSE(lifecycle.mark_installed());
    EXPECT_FALSE(lifecycle.begin_shutdown());
}

TEST(GlobalRewriteIndexTest, ResynthesisReplacesCanonicalSlotAndRetiresOldKeys) {
    detail::GlobalRewriteIndex index;
    constexpr std::uint64_t root = 0x1000;
    constexpr std::array<std::uint64_t, 2> old_aliases{0x2000, 0x3000};
    constexpr std::array<std::uint64_t, 1> new_aliases{0x4000};

    const auto first = index.upsert(root, 0x0ff0, old_aliases);
    ASSERT_FALSE(first.replaced);
    ASSERT_EQ(first.index, 0U);
    ASSERT_EQ(index.slot_count(), 1U);
    ASSERT_TRUE(index.find_alias(old_aliases[0]).has_value());
    ASSERT_EQ(*index.find_alias(old_aliases[0]), first.index);
    ASSERT_TRUE(index.find_root(0x0ff0).has_value());
    ASSERT_EQ(*index.find_root(0x0ff0), first.index);

    const auto second = index.upsert(root, 0x0fe0, new_aliases);
    EXPECT_TRUE(second.replaced);
    EXPECT_EQ(second.index, first.index);
    EXPECT_EQ(index.slot_count(), 1U);
    ASSERT_TRUE(index.find_root(root).has_value());
    EXPECT_EQ(*index.find_root(root), second.index);
    ASSERT_TRUE(index.find_root(0x0fe0).has_value());
    EXPECT_EQ(*index.find_root(0x0fe0), second.index);
    EXPECT_FALSE(index.find_root(0x0ff0).has_value());
    EXPECT_FALSE(index.find_alias(old_aliases[0]).has_value());
    EXPECT_FALSE(index.find_alias(old_aliases[1]).has_value());
    ASSERT_TRUE(index.find_alias(new_aliases[0]).has_value());
    EXPECT_EQ(*index.find_alias(new_aliases[0]), second.index);
}

TEST(GlobalRewriteIndexTest, DistinctCanonicalRootsReceiveDeterministicSlots) {
    detail::GlobalRewriteIndex index;
    constexpr std::array<std::uint64_t, 0> no_aliases{};

    const auto first = index.upsert(0x1000, std::nullopt, no_aliases);
    const auto second = index.upsert(0x5000, std::nullopt, no_aliases);

    EXPECT_EQ(first.index, 0U);
    EXPECT_EQ(second.index, 1U);
    EXPECT_FALSE(first.replaced);
    EXPECT_FALSE(second.replaced);
    EXPECT_EQ(index.slot_count(), 2U);
}

TEST(GlobalRewriteIndexTest, SharedItemHeadDoesNotReplaceDistinctSubobjectRoots) {
    detail::GlobalRewriteIndex index;
    constexpr std::array<std::uint64_t, 0> no_aliases{};
    constexpr std::uint64_t shared_head = 0x1000;

    const auto first = index.upsert(0x1040, shared_head, no_aliases);
    const auto second = index.upsert(0x1080, shared_head, no_aliases);

    EXPECT_FALSE(first.replaced);
    EXPECT_FALSE(second.replaced);
    EXPECT_EQ(first.index, 0U);
    EXPECT_EQ(second.index, 1U);
    EXPECT_EQ(index.slot_count(), 2U);
    ASSERT_TRUE(index.find_root(0x1040).has_value());
    ASSERT_TRUE(index.find_root(0x1080).has_value());
    EXPECT_EQ(*index.find_root(0x1040), first.index);
    EXPECT_EQ(*index.find_root(0x1080), second.index);
    EXPECT_FALSE(index.find_root(shared_head).has_value());
}

TEST(GlobalTinfoTransactionStateTest, FailedApplicationRestoresPriorType) {
    detail::GlobalTinfoTransactionState transaction(/*had_prior_tinfo=*/true);

    EXPECT_FALSE(transaction.finish_application_verification(false));
    EXPECT_EQ(
        transaction.phase(),
        detail::GlobalTinfoTransactionPhase::RollbackRequired);

    const detail::GlobalTinfoRollbackPlan plan = transaction.rollback_plan();
    EXPECT_TRUE(plan.remove_requested_tinfo);
    EXPECT_FALSE(plan.remove_temporary_data_item);
    EXPECT_TRUE(plan.restore_prior_tinfo);

    EXPECT_TRUE(transaction.finish_rollback_verification(true, true));
    EXPECT_EQ(
        transaction.phase(),
        detail::GlobalTinfoTransactionPhase::RolledBack);
}

TEST(GlobalTinfoTransactionStateTest, TemporaryItemIsRemovedWhenNoPriorTypeExists) {
    detail::GlobalTinfoTransactionState transaction(/*had_prior_tinfo=*/false);
    transaction.record_temporary_data_item();

    EXPECT_FALSE(transaction.finish_application_verification(false));
    const detail::GlobalTinfoRollbackPlan plan = transaction.rollback_plan();
    EXPECT_TRUE(plan.remove_requested_tinfo);
    EXPECT_TRUE(plan.remove_temporary_data_item);
    EXPECT_FALSE(plan.restore_prior_tinfo);

    EXPECT_FALSE(transaction.finish_rollback_verification(true, false));
    EXPECT_EQ(
        transaction.phase(),
        detail::GlobalTinfoTransactionPhase::RollbackFailed);
}

TEST(GlobalTinfoTransactionStateTest, ExactReadbackCommitsWithoutRollbackPlan) {
    detail::GlobalTinfoTransactionState transaction(/*had_prior_tinfo=*/true);

    EXPECT_TRUE(transaction.finish_application_verification(true));
    EXPECT_EQ(
        transaction.phase(),
        detail::GlobalTinfoTransactionPhase::Committed);

    const detail::GlobalTinfoRollbackPlan plan = transaction.rollback_plan();
    EXPECT_FALSE(plan.remove_requested_tinfo);
    EXPECT_FALSE(plan.remove_temporary_data_item);
    EXPECT_FALSE(plan.restore_prior_tinfo);
}

} // namespace structor::test
