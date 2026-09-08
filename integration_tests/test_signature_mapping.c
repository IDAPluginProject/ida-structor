typedef unsigned long long uint64_t;

__attribute__((noinline)) double signature_mapping_carrier(
    uint64_t *pointer, double value, unsigned int count)
{
    volatile uint64_t local = *pointer + count;
    return value + (double)local;
}

int main(void)
{
    uint64_t value = 7;
    return (int)signature_mapping_carrier(&value, 2.5, 3);
}
