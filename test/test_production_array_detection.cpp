#include "mock_ida.hpp"
#include "structor/z3/array_constraints.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <random>

using namespace structor;
using namespace structor::z3;

namespace {

qvector<FieldAccess> integer_accesses(std::initializer_list<sval_t> offsets) {
    qvector<FieldAccess> accesses;
    for (const sval_t offset : offsets) {
        FieldAccess access;
        access.offset = offset;
        access.size = 4;
        access.semantic_type = SemanticType::Integer;
        access.inferred_type.create_simple_type(BTF_INT32);
        accesses.push_back(access);
    }
    return accesses;
}

void assert_covers_all(const ArrayCandidate& candidate,
                     const qvector<FieldAccess>& accesses) {
    assert(candidate.is_valid_c_array());
    const auto end = candidate.checked_end_offset();
    assert(end);
    for (const auto& access : accesses) {
        const auto access_end = checked_interval_end(access.offset, access.size);
        assert(access_end && *access_end <= *end);
        assert(candidate.get_element_index(access.offset));
    }
}

void test_sparse_extent_includes_unobserved_elements() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    for (const auto& offsets : {integer_accesses({0, 4, 12}),
                                integer_accesses({-12, -8, 0}),
                                integer_accesses({32, 36, 44})}) {
        const auto candidates = builder.detect_arrays(offsets);
        assert(candidates.size() == 1);
        assert(candidates[0].element_count == 4);
        assert(candidates[0].total_size() == 16);
        assert_covers_all(candidates[0], offsets);
    }
}

void test_sparse_extent_limit_counts_missing_elements() {
    Z3Context ctx;
    auto config = make_array_detection_config(3, 3, true, 4096);
    ArrayConstraintBuilder builder(ctx, config);
    assert(builder.detect_arrays(integer_accesses({0, 4, 12})).empty());

    config.max_elements = 4;
    ArrayConstraintBuilder exact_limit(ctx, config);
    const auto candidates = exact_limit.detect_arrays(integer_accesses({0, 4, 12}));
    assert(candidates.size() == 1 && candidates[0].element_count == 4);
}

void test_overlapping_scalar_accesses_are_not_array_elements() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    assert(builder.detect_arrays(integer_accesses({0, 2, 4})).empty());
}

void test_incompatible_types_after_unknown_are_not_homogeneous() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 4, 8});
    accesses[0].inferred_type.clear();
    for (auto& access : accesses) access.semantic_type = SemanticType::Unknown;
    accesses[2].inferred_type.create_simple_type(BTF_FLOAT);
    assert(builder.detect_arrays(accesses).empty());
}

void test_exact_runs_respect_element_cap() {
    Z3Context ctx;
    auto config = make_array_detection_config(3, 3, false, 4096);
    ArrayConstraintBuilder builder(ctx, config);
    // Two individually over-limit contiguous runs, separated by a large gap.
    assert(builder.detect_arrays(integer_accesses({0, 4, 8, 12, 128, 132, 136, 140})).empty());
}

void test_duplicate_observations_do_not_satisfy_minimum_elements() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 0, 8});
    for (auto& access : accesses) access.array_stride_hint = 4;
    qvector<const FieldAccess*> pointers;
    for (const auto& access : accesses) pointers.push_back(&access);
    assert(!builder.detect_symbolic_array(pointers));
    assert(!builder.solve_stride_z3(pointers));
}

void test_array_extent_must_be_representable() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    const auto maximum = std::numeric_limits<sval_t>::max();
    assert(builder.detect_arrays(integer_accesses({maximum - 8, maximum - 4, maximum})).empty());
}

void test_overlapping_arrays_with_distinct_element_types_remain_alternatives() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 4, 8, 12});
    auto narrow = integer_accesses({0, 4, 8, 12});
    for (auto& access : narrow) {
        access.size = 2;
        access.inferred_type.create_simple_type(BTF_INT16);
        accesses.push_back(access);
    }
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 2);
    assert(candidates[0].needs_element_struct != candidates[1].needs_element_struct);
    for (const auto& candidate : candidates) {
        assert(candidate.element_count == 4);
        assert(candidate.stride == 4);
        assert(candidate.is_valid_c_array());
    }
}

void test_promoted_expression_type_preserves_stored_element_width() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 2, 4});
    for (auto& access : accesses) access.size = 2;
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 1);
    assert(candidates[0].element_count == 3);
    assert(candidates[0].element_type.get_size() == 2);
    assert(candidates[0].element_type.is_signed());
    assert_covers_all(candidates[0], accesses);
}

void test_promoted_unsigned_load_preserves_storage_signedness() {
    Z3Context ctx;
    ArrayDetectionConfig config;
    config.min_elements = config.max_elements = 2;
    ArrayConstraintBuilder builder(ctx, config);
    auto accesses = integer_accesses({12, 14});
    for (auto& access : accesses) access.size = 2;
    // The first storage type is concrete even though its generic semantic
    // label is signed. The second unsigned load was promoted to signed int.
    accesses[0].inferred_type.create_simple_type(BTF_UINT16);
    accesses[1].semantic_type = SemanticType::UnsignedInteger;
    const tinfo_t expected = accesses[0].inferred_type;
    for (int ordering = 0; ordering < 2; ++ordering) {
        const auto candidates = builder.detect_arrays(accesses);
        assert(candidates.size() == 1);
        assert(candidates[0].base_offset == 12 && candidates[0].stride == 2);
        assert(candidates[0].element_count == 2);
        assert(candidates[0].element_type.equals_to(expected));
        assert(candidates[0].has_member_evidence(accesses[0]));
        assert(candidates[0].has_member_evidence(accesses[1]));
        assert_covers_all(candidates[0], accesses);
        std::reverse(accesses.begin(), accesses.end());
    }
}

void test_exact_width_signed_and_unsigned_views_remain_distinct() {
    Z3Context ctx;
    ArrayDetectionConfig config;
    config.min_elements = config.max_elements = 2;
    ArrayConstraintBuilder builder(ctx, config);
    auto accesses = integer_accesses({12, 14});
    for (auto& access : accesses) {
        access.size = 2;
        access.semantic_type = SemanticType::UnsignedInteger;
    }
    accesses[0].inferred_type.create_simple_type(BTF_UINT16);
    accesses[1].inferred_type.create_simple_type(BTF_INT16);
    for (int ordering = 0; ordering < 2; ++ordering) {
        assert(builder.detect_arrays(accesses).empty());
        std::reverse(accesses.begin(), accesses.end());
    }
}

void test_unknown_first_observation_retains_later_type_evidence() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 4, 8});
    accesses[0].inferred_type.clear();
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 1);
    assert(candidates[0].element_type.equals_to(accesses[1].inferred_type));
    assert_covers_all(candidates[0], accesses);
}

qvector<FieldAccess> float_accesses(std::initializer_list<sval_t> offsets) {
    auto accesses = integer_accesses(offsets);
    for (auto& access : accesses) {
        access.inferred_type.create_simple_type(BTF_FLOAT);
        access.semantic_type = SemanticType::Float;
    }
    return accesses;
}

void test_independent_adjacent_homogeneous_runs_are_recovered() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 4, 8});
    for (const auto& access : float_accesses({12, 16, 20})) accesses.push_back(access);
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 2);
    assert(candidates[0].base_offset == 0 && candidates[0].element_count == 3);
    assert(candidates[0].element_type.equals_to(accesses[0].inferred_type));
    assert(candidates[1].base_offset == 12 && candidates[1].element_count == 3);
    assert(candidates[1].element_type.is_floating());
}

void test_conflicting_views_keep_separate_owned_evidence() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 4, 8});
    for (const auto& access : float_accesses({0, 4, 8})) accesses.push_back(access);
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 2);
    const auto int_evidence = accesses[0];
    const auto float_evidence = accesses[3];
    accesses.clear();  // Returned evidence must not borrow the input vector.
    assert(candidates[0].base_offset == 0 && candidates[1].base_offset == 0);
    assert(candidates[0].element_count == 3 && candidates[1].element_count == 3);
    assert(candidates[0].has_member_evidence(int_evidence));
    assert(!candidates[0].has_member_evidence(float_evidence));
    assert(candidates[1].has_member_evidence(float_evidence));
    assert(!candidates[1].has_member_evidence(int_evidence));
}

void test_mixed_run_detection_is_order_independent() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 4, 8, 32, 36, 40});
    for (const auto& access : float_accesses({0, 4, 8, 12, 16, 20})) accesses.push_back(access);
    accesses.push_back(accesses[0]); // Duplicate observations are not new elements.
    const auto expected = builder.detect_arrays(accesses);
    assert(expected.size() == 3);
    std::mt19937 random(0x53545255);
    for (int permutation = 0; permutation < 32; ++permutation) {
        std::shuffle(accesses.begin(), accesses.end(), random);
        const auto actual = builder.detect_arrays(accesses);
        assert(actual.size() == expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            assert(actual[i].base_offset == expected[i].base_offset);
            assert(actual[i].stride == expected[i].stride);
            assert(actual[i].element_count == expected[i].element_count);
            assert(actual[i].element_type.equals_to(expected[i].element_type));
            assert(actual[i].member_offsets == expected[i].member_offsets);
            for (const auto& access : accesses) {
                assert(actual[i].has_member_evidence(access) ==
                       expected[i].has_member_evidence(access));
            }
        }
    }
}

void test_conflicting_singletons_do_not_bridge_a_typed_run() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 8, 16});
    for (const auto& access : float_accesses({4, 12, 20})) accesses.push_back(access);
    assert(builder.detect_arrays(accesses).empty());

    accesses = integer_accesses({0, 8});
    auto unknown = integer_accesses({4});
    unknown[0].inferred_type.clear();
    unknown[0].semantic_type = SemanticType::Unknown;
    accesses.push_back(unknown[0]);
    for (const auto& access : float_accesses({12, 16, 20})) accesses.push_back(access);
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 1);
    assert(candidates[0].base_offset == 12 && candidates[0].element_type.is_floating());
}

void test_fallback_type_runs_respect_caps_without_discarding_other_types() {
    Z3Context ctx;
    auto config = make_array_detection_config(3, 3, true, 4096);
    ArrayConstraintBuilder builder(ctx, config);
    auto accesses = integer_accesses({0, 4, 8, 12});
    for (const auto& access : float_accesses({16, 20, 24})) accesses.push_back(access);
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 1);
    assert(candidates[0].base_offset == 16 && candidates[0].element_count == 3);
    assert(candidates[0].element_type.is_floating());
}

void test_equal_width_distinct_pointees_form_distinct_runs() {
    Z3Context ctx;
    ArrayConstraintBuilder builder(ctx);
    auto accesses = integer_accesses({0, 8, 16});
    auto other = float_accesses({0, 8, 16});
    for (auto* view : {&accesses, &other}) {
        for (auto& access : *view) {
            access.size = 8;
            access.semantic_type = SemanticType::Pointer;
            const tinfo_t pointee = access.inferred_type;
            access.inferred_type.create_ptr(pointee);
        }
    }
    for (const auto& access : other) accesses.push_back(access);
    const auto candidates = builder.detect_arrays(accesses);
    assert(candidates.size() == 2);
    assert(candidates[0].element_type.equals_to(accesses[0].inferred_type));
    assert(candidates[1].element_type.equals_to(other[0].inferred_type));
    std::reverse(accesses.begin(), accesses.end());
    const auto reversed = builder.detect_arrays(accesses);
    assert(reversed.size() == candidates.size());
    assert(reversed[0].element_type.equals_to(candidates[0].element_type));
    assert(reversed[1].element_type.equals_to(candidates[1].element_type));
}

} // namespace

int main(int argc, char** argv) {
    const std::pair<const char*, void(*)()> cases[] = {
        {"sparse_extent", test_sparse_extent_includes_unobserved_elements},
        {"sparse_limit", test_sparse_extent_limit_counts_missing_elements},
        {"overlapping_elements", test_overlapping_scalar_accesses_are_not_array_elements},
        {"unknown_first_type", test_incompatible_types_after_unknown_are_not_homogeneous},
        {"exact_run_limit", test_exact_runs_respect_element_cap},
        {"duplicate_evidence", test_duplicate_observations_do_not_satisfy_minimum_elements},
        {"extent_overflow", test_array_extent_must_be_representable},
        {"distinct_overlapping_types", test_overlapping_arrays_with_distinct_element_types_remain_alternatives},
        {"promoted_storage_width", test_promoted_expression_type_preserves_stored_element_width},
        {"promoted_unsigned_storage", test_promoted_unsigned_load_preserves_storage_signedness},
        {"exact_storage_signedness", test_exact_width_signed_and_unsigned_views_remain_distinct},
        {"unknown_first_preserve_type", test_unknown_first_observation_retains_later_type_evidence},
        {"independent_typed_runs", test_independent_adjacent_homogeneous_runs_are_recovered},
        {"conflicting_view_evidence", test_conflicting_views_keep_separate_owned_evidence},
        {"order_independent_runs", test_mixed_run_detection_is_order_independent},
        {"untyped_bridge_rejected", test_conflicting_singletons_do_not_bridge_a_typed_run},
        {"typed_run_limits", test_fallback_type_runs_respect_caps_without_discarding_other_types},
        {"distinct_pointee_runs", test_equal_width_distinct_pointees_form_distinct_runs},
    };
    bool ran = false;
    for (const auto& [name, test] : cases) {
        if (argc < 2 || std::string(argv[1]) == name) {
            test();
            ran = true;
        }
    }
    assert(ran);
    std::cout << "[PASS] production array detection tests\n";
}
