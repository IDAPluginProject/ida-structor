#if defined(STRUCTOR_MS_ABI)
#define PROBE_ABI __attribute__((ms_abi))
#elif defined(STRUCTOR_SYSV_ABI)
#define PROBE_ABI __attribute__((sysv_abi))
#else
#define PROBE_ABI
#endif

PROBE_ABI __attribute__((noinline, used)) unsigned long long abi_target_probe(
    unsigned long long first, unsigned long long second, unsigned long long third)
{
    return first ^ (second + 3) ^ (third + 7);
}

int main(void)
{
    return (int)abi_target_probe(1, 2, 3);
}
