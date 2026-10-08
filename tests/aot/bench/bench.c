// Benchmark body: translated vs native. f(x, mem) sorts and hashes; mem has 2M qwords.
static void qs(unsigned long* v, long lo, long hi) {
    while (lo < hi) {
        unsigned long p = v[(lo + hi) / 2]; long i = lo, j = hi;
        while (i <= j) { while (v[i] < p) i++; while (v[j] > p) j--; if (i <= j) { unsigned long t = v[i]; v[i] = v[j]; v[j] = t; i++; j--; } }
        if (j - lo < hi - i) { qs(v, lo, j); lo = i; } else { qs(v, i, hi); hi = j; }
    }
}
unsigned long f(unsigned long x, unsigned long* mem) {
    const long n = 1000000; unsigned long s = x | 1, h = 1469598103934665603ul;
    for (long r = 0; r < 3; ++r) {
        for (long i = 0; i < n; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; mem[i] = s; }
        qs(mem, 0, n - 1);
        for (long i = 0; i < n; i += 97) h = (h ^ mem[i]) * 1099511628211ul;
        float acc = 0; for (long i = 0; i < n; i += 3) acc += (float)(mem[i] & 0xffff) * 0.5f - 1.0f; h ^= (unsigned long)acc;
    }
    return h;
}
