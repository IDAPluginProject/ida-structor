#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static volatile uint64_t sink;

/* The fixture compiler is Clang/GCC; lowering the typedef alignment makes
 * the byte-offset 64-bit accesses defined for the deliberately packed ABI. */
typedef uint64_t unaligned_u64 __attribute__((aligned(1)));
_Static_assert(sizeof(unaligned_u64) == 8, "unaligned_u64 size");
_Static_assert(_Alignof(unaligned_u64) == 1, "unaligned_u64 alignment");

#pragma pack(push, 2)
struct Pack2Truth {
    uint8_t lead;
    uint64_t wide;
    uint8_t tail;
};
#pragma pack(pop)

#pragma pack(push, 4)
struct Pack4Truth {
    uint8_t lead;
    uint64_t wide;
    uint8_t tail;
};
#pragma pack(pop)

/* No packing directive applies to this character-only control structure. */
struct DefaultCharsTruth {
    unsigned char first;
    unsigned char second;
    unsigned char third;
    unsigned char fourth;
    unsigned char fifth;
};

_Static_assert(offsetof(struct Pack2Truth, lead) == 0, "Pack2Truth.lead");
_Static_assert(offsetof(struct Pack2Truth, wide) == 2, "Pack2Truth.wide");
_Static_assert(offsetof(struct Pack2Truth, tail) == 10, "Pack2Truth.tail");
_Static_assert(sizeof(struct Pack2Truth) == 12, "Pack2Truth size");
_Static_assert(_Alignof(struct Pack2Truth) == 2, "Pack2Truth alignment");

_Static_assert(offsetof(struct Pack4Truth, lead) == 0, "Pack4Truth.lead");
_Static_assert(offsetof(struct Pack4Truth, wide) == 4, "Pack4Truth.wide");
_Static_assert(offsetof(struct Pack4Truth, tail) == 12, "Pack4Truth.tail");
_Static_assert(sizeof(struct Pack4Truth) == 16, "Pack4Truth size");
_Static_assert(_Alignof(struct Pack4Truth) == 4, "Pack4Truth alignment");

_Static_assert(offsetof(struct DefaultCharsTruth, first) == 0,
               "DefaultCharsTruth.first");
_Static_assert(offsetof(struct DefaultCharsTruth, second) == 1,
               "DefaultCharsTruth.second");
_Static_assert(offsetof(struct DefaultCharsTruth, third) == 2,
               "DefaultCharsTruth.third");
_Static_assert(offsetof(struct DefaultCharsTruth, fourth) == 3,
               "DefaultCharsTruth.fourth");
_Static_assert(offsetof(struct DefaultCharsTruth, fifth) == 4,
               "DefaultCharsTruth.fifth");
_Static_assert(sizeof(struct DefaultCharsTruth) == 5, "DefaultCharsTruth size");
_Static_assert(_Alignof(struct DefaultCharsTruth) == 1,
               "DefaultCharsTruth alignment");

NOINLINE void seed_pack2(void *object) {
    uint8_t *bytes = (uint8_t *)object;
    *(uint8_t *)(bytes + 0) = UINT8_C(0x12);
    *(unaligned_u64 *)(bytes + 2) = UINT64_C(0x1122334455667788);
    *(uint8_t *)(bytes + 10) = UINT8_C(0x2A);
}

NOINLINE void read_pack2(const void *object) {
    const uint8_t *bytes = (const uint8_t *)object;
    sink ^= *(const uint8_t *)(bytes + 0);
    sink ^= *(const unaligned_u64 *)(bytes + 2);
    sink ^= *(const uint8_t *)(bytes + 10);
}

NOINLINE void seed_pack4(void *object) {
    uint8_t *bytes = (uint8_t *)object;
    *(uint8_t *)(bytes + 0) = UINT8_C(0x34);
    *(unaligned_u64 *)(bytes + 4) = UINT64_C(0x8877665544332211);
    *(uint8_t *)(bytes + 12) = UINT8_C(0x4B);
}

NOINLINE void read_pack4(const void *object) {
    const uint8_t *bytes = (const uint8_t *)object;
    sink ^= *(const uint8_t *)(bytes + 0);
    sink ^= *(const unaligned_u64 *)(bytes + 4);
    sink ^= *(const uint8_t *)(bytes + 12);
}

NOINLINE void seed_default_chars(void *object) {
    uint8_t *bytes = (uint8_t *)object;
    *(uint8_t *)(bytes + 0) = UINT8_C(0x41);
    *(uint8_t *)(bytes + 1) = UINT8_C(0x42);
    *(uint8_t *)(bytes + 2) = UINT8_C(0x43);
    *(uint8_t *)(bytes + 3) = UINT8_C(0x44);
    *(uint8_t *)(bytes + 4) = UINT8_C(0x45);
}

NOINLINE void read_default_chars(const void *object) {
    const uint8_t *bytes = (const uint8_t *)object;
    sink ^= *(const uint8_t *)(bytes + 0);
    sink ^= *(const uint8_t *)(bytes + 1);
    sink ^= *(const uint8_t *)(bytes + 2);
    sink ^= *(const uint8_t *)(bytes + 3);
    sink ^= *(const uint8_t *)(bytes + 4);
}

int main(void) {
    struct Pack2Truth pack2;
    struct Pack4Truth pack4;
    struct DefaultCharsTruth default_chars;

    seed_pack2(&pack2);
    read_pack2(&pack2);
    seed_pack4(&pack4);
    read_pack4(&pack4);
    seed_default_chars(&default_chars);
    read_default_chars(&default_chars);

    (void)sink;
    return 0;
}
