#include <stdint.h>

static volatile uint64_t sink;

#define BASE_READS(p) \
    do { \
        sink ^= *(uint32_t *)((uint8_t *)(p) + 0); \
        sink ^= *(uint64_t *)((uint8_t *)(p) + 8); \
    } while (0)
#define INDEX_READ(p, idx) \
    (sink ^= *(uint32_t *)((uint8_t *)(p) + 32 + (idx) * 4))
#define NOINLINE __attribute__((noinline))

NOINLINE void read_unsigned_guard(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, idx);
}

NOINLINE void read_signed_range(void *p, int idx) {
    BASE_READS(p);
    if (idx >= 0 && idx <= 3) INDEX_READ(p, idx);
}

NOINLINE void read_signed_upper_only(void *p, int idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, idx);
}

NOINLINE void read_prior_comparison(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) sink ^= 1;
    INDEX_READ(p, idx);
}

NOINLINE void read_later_comparison(void *p, unsigned idx) {
    BASE_READS(p);
    INDEX_READ(p, idx);
    if (idx < 4) sink ^= 1;
}

NOINLINE void read_unbounded_else(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) sink ^= 1;
    else INDEX_READ(p, idx);
}

NOINLINE void read_bounded_else(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx > 3) sink ^= 1;
    else INDEX_READ(p, idx);
}

NOINLINE void read_narrowed_guard(void *p, uint64_t idx) {
    BASE_READS(p);
    if ((uint8_t)idx < 4) INDEX_READ(p, idx);
}

NOINLINE void read_large_guard(void *p, uint64_t idx) {
    BASE_READS(p);
    if (idx < UINT64_C(0x100000004)) INDEX_READ(p, idx);
}

NOINLINE void read_reassigned_index(void *p, unsigned idx, unsigned replacement) {
    BASE_READS(p);
    if (idx < 4) {
        idx = replacement;
        INDEX_READ(p, idx);
    }
}

NOINLINE void read_alternative_guard(void *p, unsigned idx, int flag) {
    BASE_READS(p);
    if (idx < 4 || flag) INDEX_READ(p, idx);
}

NOINLINE void read_unsigned_loop(void *p, unsigned idx) {
    BASE_READS(p);
    while (idx < 4) {
        INDEX_READ(p, idx);
        ++idx;
    }
}

NOINLINE void read_do_loop(void *p, unsigned idx) {
    BASE_READS(p);
    do {
        INDEX_READ(p, idx);
        ++idx;
    } while (idx < 4);
}

NOINLINE void read_guard_outside_mutating_loop(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) {
        do {
            INDEX_READ(p, idx);
            ++idx;
        } while (idx < 6);
    }
}

NOINLINE void read_nested_guard(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, idx);
    sink ^= *(uint32_t *)((uint8_t *)p + 128 + idx * 4);
}

NOINLINE void read_unknown_index(void *p, unsigned idx) {
    BASE_READS(p);
    INDEX_READ(p, idx);
}

NOINLINE void call_unknown_index(void *p, unsigned idx) {
    typedef void (*callback_t)(void);
    BASE_READS(p);
    ((callback_t *)((uint8_t *)p + 32))[idx]();
}

NOINLINE void read_truncated_index(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, (uint8_t)(idx + 256));
}

NOINLINE void read_wrapping_index(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, (uint8_t)(idx - 1));
}

NOINLINE void read_sparse_index(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, idx * 2);
}

NOINLINE void read_reverse_index(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, 3 - idx);
}

NOINLINE void read_float_cast_guard(void *p, unsigned idx) {
    BASE_READS(p);
    if ((unsigned)(float)idx < 4) INDEX_READ(p, idx);
}

NOINLINE void read_high_finite_range(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx >= 100 && idx < 104) INDEX_READ(p, idx);
}

NOINLINE void read_negative_finite_range(void *p, int idx) {
    BASE_READS(p);
    if (idx >= -2 && idx < 2) INDEX_READ(p, idx);
}

NOINLINE void read_truncated_base(void *p) {
    BASE_READS(p);
    uintptr_t truncated = (uint32_t)(uintptr_t)p;
    sink ^= *(uint32_t *)(truncated + 32);
}

NOINLINE void read_truncated_base_direct(void *p) {
    BASE_READS(p);
    sink ^= *(uint32_t *)((uintptr_t)(uint32_t)(uintptr_t)p + 32);
}

NOINLINE void read_signedness_guard(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx >= 0U && (int32_t)idx < 4) INDEX_READ(p, idx);
}

NOINLINE void read_boolean_guard(void *p, uint8_t idx) {
    BASE_READS(p);
    if ((_Bool)idx < 2) INDEX_READ(p, idx);
}

// Keep these bases untyped by avoiding any zero-offset load. This exercises
// cot_ptr byte arithmetic independently from the cot_idx cases above.
#define REMOTE_READS(p) \
    do { \
        sink ^= *(uint16_t *)((uint8_t *)(p) + 16); \
        sink ^= *(uint64_t *)((uint8_t *)(p) + 24); \
    } while (0)
#define REMOTE_INDEX_READ(p, idx) \
    (sink ^= *(uint32_t *)((uint8_t *)(p) + 64 + (idx) * 4))

NOINLINE void deref_unsigned_guard(void *p, unsigned idx) {
    REMOTE_READS(p);
    if (idx < 4) REMOTE_INDEX_READ(p, idx);
}

NOINLINE void deref_prior_comparison(void *p, unsigned idx) {
    REMOTE_READS(p);
    if (idx < 4) sink ^= 1;
    REMOTE_INDEX_READ(p, idx);
}

NOINLINE void deref_later_comparison(void *p, unsigned idx) {
    REMOTE_READS(p);
    REMOTE_INDEX_READ(p, idx);
    if (idx < 4) sink ^= 1;
}

NOINLINE void deref_signed_upper_only(void *p, int idx) {
    REMOTE_READS(p);
    if (idx < 4) REMOTE_INDEX_READ(p, idx);
}

NOINLINE void deref_narrowed_guard(void *p, uint64_t idx) {
    REMOTE_READS(p);
    if ((uint8_t)idx < 4) REMOTE_INDEX_READ(p, idx);
}

NOINLINE void deref_large_guard(void *p, uint64_t idx) {
    REMOTE_READS(p);
    if (idx < UINT64_C(0x100000004)) REMOTE_INDEX_READ(p, idx);
}

NOINLINE void read_preincrement_index(void *p, unsigned idx) {
    BASE_READS(p);
    if (idx < 4) INDEX_READ(p, ++idx);
}

// Every case remains independently reachable, without cross-function field
// evidence being needed for the collector's assertions.
int main(int argc, char **argv) {
    uint64_t storage[256] = {0};
    (void)argv;
    read_unsigned_guard(storage, 0);
    read_signed_range(storage, 0);
    read_signed_upper_only(storage, 0);
    read_prior_comparison(storage, 0);
    read_later_comparison(storage, 0);
    read_unbounded_else(storage, 4);
    read_bounded_else(storage, 0);
    read_narrowed_guard(storage, 0);
    read_large_guard(storage, 0);
    read_reassigned_index(storage, 0, 5);
    read_alternative_guard(storage, 5, 1);
    read_unsigned_loop(storage, 0);
    read_do_loop(storage, 0);
    read_guard_outside_mutating_loop(storage, 0);
    read_nested_guard(storage, 0);
    read_unknown_index(storage, 0);
    read_truncated_index(storage, 0);
    read_wrapping_index(storage, 1);
    read_sparse_index(storage, 0);
    read_reverse_index(storage, 0);
    read_float_cast_guard(storage, 0);
    read_high_finite_range(storage, 100);
    read_negative_finite_range(storage, 0);
    if (argc == 12346) read_truncated_base(storage);
    if (argc == 12347) read_truncated_base_direct(storage);
    read_signedness_guard(storage, 0);
    read_boolean_guard(storage, 0);
    deref_unsigned_guard(storage, 0);
    deref_prior_comparison(storage, 0);
    deref_later_comparison(storage, 0);
    deref_signed_upper_only(storage, 0);
    deref_narrowed_guard(storage, 0);
    deref_large_guard(storage, 0);
    read_preincrement_index(storage, 0);
    if (argc == 12345) call_unknown_index(storage, 0);
    return (int)sink;
}
