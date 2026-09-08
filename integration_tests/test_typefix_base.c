#include <stdint.h>
#include <string.h>

#define NOINLINE __attribute__((noinline))

NOINLINE int32_t typefix_scalar(void *base) {
    return *(volatile int32_t *)base;
}

NOINLINE int32_t *typefix_pointer(void *base) {
    return *(int32_t *volatile *)base;
}

NOINLINE int32_t typefix_repeated(void *base) {
    int32_t first = *(volatile int32_t *)base;
    return first + *(volatile int32_t *)base;
}

NOINLINE int32_t typefix_displaced(void *base) {
    return *(volatile int32_t *)((char *)base + 4);
}

NOINLINE int32_t typefix_negative(void *base) {
    return *(volatile int32_t *)((char *)base - 4);
}

NOINLINE int typefix_callback(void *base) {
    return (*(int (*volatile *)(void))base)();
}

NOINLINE uintptr_t typefix_consume(uintptr_t value) {
    __asm__ volatile("" : "+r"(value));
    return value;
}

NOINLINE int32_t typefix_consume_pointer(int32_t *value) {
    return *value;
}

NOINLINE uintptr_t typefix_alias_call(void *base) {
    uintptr_t alias = (uintptr_t)base;
    uintptr_t result = typefix_consume(alias);
    return result + (alias == 0);
}

NOINLINE uintptr_t typefix_alias_compare(void *base, uintptr_t other) {
    uintptr_t alias = (uintptr_t)base;
    uintptr_t result = typefix_consume(other);
    return result + (alias == 0) + (alias & 1);
}

NOINLINE uintptr_t typefix_loaded_call(void *base) {
    uintptr_t loaded = *(volatile uintptr_t *)base;
    return typefix_consume(loaded);
}

NOINLINE int32_t typefix_loaded_pointer_call(void *base) {
    int32_t *loaded = *(int32_t *volatile *)base;
    return typefix_consume_pointer(loaded);
}

NOINLINE size_t typefix_loaded_known_pointer_call(void *base) {
    const char *loaded = *(const char *volatile *)base;
    return strlen(loaded);
}

static int callback(void) {
    return 3;
}

int main(void) {
    int32_t values[2] = {1, 2};
    int32_t *pointer = values;
    const char *text = "pointer evidence";
    int (*function)(void) = callback;
    return typefix_scalar(values) + *typefix_pointer(&pointer) +
           typefix_repeated(values) + typefix_displaced(values) +
           typefix_negative(values + 1) + typefix_callback(&function) +
           typefix_loaded_pointer_call(&pointer) +
           (int)typefix_loaded_known_pointer_call(&text);
}
