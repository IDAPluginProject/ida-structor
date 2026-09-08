#include <stdint.h>
#include <stddef.h>

static volatile uint64_t sink;
static unsigned *volatile retained;
#define NOINLINE __attribute__((noinline))
#define HEADER_READS(p) do { \
    sink ^= *(uint16_t *)((uint8_t *)(p) + 16); \
    sink ^= *(uint64_t *)((uint8_t *)(p) + 24); \
} while (0)
#define INDEX_LOAD(p, index) (*(uint32_t *)((uint8_t *)(p) + 64 + (index) * 4))
#define AFTER_LOAD(p, index) (*(uint32_t *)((uint8_t *)(p) + 128 + (index) * 4))

NOINLINE void retain_index(unsigned *index) {
    retained = index;
}
NOINLINE void consume_value_pointer(uint32_t value, unsigned *index) {
    sink ^= value;
    *index += 100;
}
NOINLINE void consume_pointer_value(unsigned *index, uint32_t value) {
    sink ^= value;
    *index += 100;
}
NOINLINE void consume_pair(uint32_t first, uint32_t second) {
    sink ^= first + second;
}
NOINLINE unsigned mutate_index(unsigned *index) {
    *index += 100;
    return *index;
}

NOINLINE void *next_pointer_self_assignment(void *base, unsigned hops) {
    void *cursor = base;
    for (unsigned i = 0; i < hops; ++i) {
        sink ^= *(uint16_t *)((uint8_t *)cursor + 24);
        cursor = *(void **)cursor;
    }
    return cursor;
}
NOINLINE void *offset_self_assignment(void *base, unsigned hops) {
    void *cursor = base;
    for (unsigned i = 0; i < hops; ++i) {
        sink ^= *(uint16_t *)((uint8_t *)cursor + 24);
        cursor = *(void **)((uint8_t *)cursor + 8);
    }
    return cursor;
}
NOINLINE void assignment_index_loop(void *base, unsigned index) {
    HEADER_READS(base);
    while (index < 4) {
        index = INDEX_LOAD(base, index);
        sink ^= index;
    }
    sink ^= AFTER_LOAD(base, index);
}
NOINLINE void compound_assignment_index(void *base, unsigned index) {
    HEADER_READS(base);
    while (index < 4) {
        index += INDEX_LOAD(base, index);
        sink ^= index;
    }
    sink ^= AFTER_LOAD(base, index);
}
NOINLINE void assignment_postincrement_index(void *base, unsigned index) {
    HEADER_READS(base);
    while (index < 4) {
        uint32_t value = INDEX_LOAD(base, index++);
        index = value;
        sink ^= index;
    }
    sink ^= AFTER_LOAD(base, index);
}
NOINLINE void assignment_preincrement_index(void *base, unsigned index) {
    HEADER_READS(base);
    while (index < 4) {
        uint32_t value = INDEX_LOAD(base, ++index);
        index = value;
        sink ^= index;
    }
    sink ^= AFTER_LOAD(base, index);
}
NOINLINE void call_previously_escaped_index(void *base, unsigned index) {
    HEADER_READS(base);
    retain_index(&index);
    if (index < 4) {
        HEADER_READS(base);
        consume_value_pointer(INDEX_LOAD(base, index), &index);
        sink ^= AFTER_LOAD(base, index);
    }
}
NOINLINE void call_address_before_value(void *base, unsigned index) {
    HEADER_READS(base);
    retain_index(&index);
    if (index < 4) {
        HEADER_READS(base);
        consume_pointer_value(&index, INDEX_LOAD(base, index));
        sink ^= AFTER_LOAD(base, index);
    }
}
NOINLINE void call_nested_mutator_before_value(void *base, unsigned index) {
    HEADER_READS(base);
    retain_index(&index);
    if (index < 4) {
        HEADER_READS(base);
        consume_pair(mutate_index(&index), INDEX_LOAD(base, index));
        sink ^= AFTER_LOAD(base, index);
    }
}
NOINLINE void call_postincrement_index(void *base, unsigned index) {
    HEADER_READS(base);
    retain_index(&index);
    if (index < 4) {
        HEADER_READS(base);
        consume_value_pointer(INDEX_LOAD(base, index++), &index);
        sink ^= AFTER_LOAD(base, index);
    }
}

int main(void) {
    uint64_t storage[256] = {0};
    storage[0] = (uintptr_t)storage;
    storage[1] = (uintptr_t)storage;
    for (unsigned i = 0; i < 5; ++i) ((uint32_t *)storage)[16 + i] = i + 4;
    next_pointer_self_assignment(storage, 2);
    offset_self_assignment(storage, 2);
    assignment_index_loop(storage, 0);
    compound_assignment_index(storage, 0);
    assignment_postincrement_index(storage, 0);
    assignment_preincrement_index(storage, 0);
    call_previously_escaped_index(storage, 0);
    call_address_before_value(storage, 0);
    call_nested_mutator_before_value(storage, 0);
    call_postincrement_index(storage, 0);
    return 0;
}
