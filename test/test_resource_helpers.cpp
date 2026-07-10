/**
 * @file test_resource_helpers.cpp
 * @brief Boundary tests for production candidate/array resource helpers.
 */

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

#include "structor/z3/array_constraints.hpp"
#include "structor/z3/field_candidates.hpp"

#include <cassert>
#include <iostream>
#include <limits>

using structor::z3::ArrayCandidate;
using structor::z3::FieldCandidate;
using structor::z3::make_array_detection_config;

namespace {

void test_candidate_interval_boundary() {
    FieldCandidate boundary;
    boundary.offset = std::numeric_limits<sval_t>::max() - 3;
    boundary.size = 3;
    assert(boundary.checked_end_offset().has_value());
    assert(*boundary.checked_end_offset() == std::numeric_limits<sval_t>::max());

    FieldCandidate invalid = boundary;
    invalid.size = 4;
    assert(!invalid.checked_end_offset().has_value());
    assert(invalid.end_offset() == std::numeric_limits<sval_t>::max());
    assert(!invalid.overlaps(boundary));
    assert(!boundary.overlaps(invalid));
    assert(!invalid.contains(boundary));
}

void test_array_size_and_end_boundary() {
    ArrayCandidate array;
    assert(!array.checked_total_size().has_value());
    assert(!array.checked_end_offset().has_value());

    array.base_offset = 32;
    array.stride = 4096;
    array.element_count = 1024;
    assert(array.checked_total_size() == 4194304u);
    assert(array.checked_end_offset() == 4194336);

    array.stride = std::numeric_limits<std::uint32_t>::max();
    array.element_count = 2;
    assert(!array.checked_total_size().has_value());
    assert(!array.checked_end_offset().has_value());
    assert(!array.contains_offset(32));
}

void test_optional_array_cap_preserves_scalar_eligibility() {
    FieldCandidate oversized_array;
    oversized_array.kind = FieldCandidate::Kind::ArrayField;
    oversized_array.array_element_count = 1025;
    assert(!oversized_array.within_array_element_limit(1024));

    FieldCandidate scalar;
    scalar.kind = FieldCandidate::Kind::DirectAccess;
    assert(scalar.within_array_element_limit(1024));
    assert(scalar.within_array_element_limit(0));
}

void test_optional_confidence_threshold_has_explicit_percent_semantics() {
    FieldCandidate optional;
    optional.confidence = structor::z3::TypeConfidence::Low;
    assert(optional.meets_optional_confidence_threshold(25));
    assert(!optional.meets_optional_confidence_threshold(26));
    optional.confidence = structor::z3::TypeConfidence::Absolute;
    assert(optional.meets_optional_confidence_threshold(100));
}

void test_array_detection_controls_are_mapped_without_default_fallback() {
    const auto config = make_array_detection_config(5, 17, false, 96);
    assert(config.min_elements == 5);
    assert(config.max_elements == 17);
    assert(!config.use_symbolic_indices);
    assert(config.max_stride == 96);
}

} // namespace

int main() {
    test_candidate_interval_boundary();
    test_array_size_and_end_boundary();
    test_optional_array_cap_preserves_scalar_eligibility();
    test_optional_confidence_threshold_has_explicit_percent_semantics();
    test_array_detection_controls_are_mapped_without_default_fallback();
    std::cout << "[PASS] production resource helper tests\n";
    return 0;
}
