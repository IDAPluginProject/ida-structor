#include <stdint.h>
#include <stdio.h>

/*
 * The storage has the natural alignment required by every typed access below.
 * may_alias makes the two overlay views explicit without relying on C's
 * effective-type aliasing rules, while still producing direct 4-byte loads at
 * -O0 for the structure-recovery fixture.
 */
typedef uint16_t alias_u16 __attribute__((may_alias));
typedef uint32_t alias_u32 __attribute__((may_alias));
typedef uint64_t alias_u64 __attribute__((may_alias));
typedef float alias_f32 __attribute__((may_alias));

enum {
    OVERLAY_MAGIC = 0x00,
    OVERLAY_VERSION = 0x04,
    OVERLAY_KIND = 0x06,
    OVERLAY_VIEW = 0x08,
    OVERLAY_TAG = 0x0C,
    OVERLAY_FOOTER = 0x10,
    OVERLAY_SIZE = 0x18,
};

static volatile uint64_t sink_u64;
static volatile float sink_f32;
static _Alignas(8) uint8_t g_overlay_storage[OVERLAY_SIZE];

__attribute__((noinline))
void seed_global_overlay(void *dst) {
    uint8_t *b = (uint8_t *)dst;
    *(alias_u32 *)(void *)(b + OVERLAY_MAGIC) = 0x4F564C59U;
    *(alias_u16 *)(void *)(b + OVERLAY_VERSION) = 3U;
    *(alias_u16 *)(void *)(b + OVERLAY_KIND) = 7U;
    *(alias_u32 *)(void *)(b + OVERLAY_VIEW) = 0x40490FDBU;
    *(alias_u64 *)(void *)(b + OVERLAY_FOOTER) = UINT64_C(0x1122334455667788);
}

__attribute__((noinline))
uint32_t consume_global_overlay_u32(const void *src) {
    const uint8_t *b = (const uint8_t *)src;
    /* OVERLAY_TAG is BSS-zeroed and gives this consumer independent evidence. */
    return *(const alias_u32 *)(const void *)(b + OVERLAY_VIEW) ^
           *(const alias_u32 *)(const void *)(b + OVERLAY_TAG);
}

__attribute__((noinline))
void consume_global_overlay_float(const void *src) {
    const uint8_t *b = (const uint8_t *)src;
    sink_f32 += *(const alias_f32 *)(const void *)(b + OVERLAY_VIEW);
}

__attribute__((noinline))
void consume_global_overlay_edges(const void *src) {
    const uint8_t *b = (const uint8_t *)src;
    sink_u64 ^= *(const alias_u32 *)(const void *)(b + OVERLAY_MAGIC);
    sink_u64 ^= *(const alias_u16 *)(const void *)(b + OVERLAY_VERSION);
    sink_u64 ^= *(const alias_u16 *)(const void *)(b + OVERLAY_KIND);
    sink_u64 ^= *(const alias_u64 *)(const void *)(b + OVERLAY_FOOTER);
}

__attribute__((noinline))
void initialize_global_overlay(void) {
    seed_global_overlay(g_overlay_storage);
}

__attribute__((noinline))
void inspect_global_overlay(void) {
    consume_global_overlay_edges(g_overlay_storage);
    sink_u64 ^= consume_global_overlay_u32(g_overlay_storage);
    consume_global_overlay_float(g_overlay_storage);
}

int main(void) {
    initialize_global_overlay();
    inspect_global_overlay();
    printf("sink=%llx/%g\n", (unsigned long long)sink_u64, (double)sink_f32);
    return 0;
}
