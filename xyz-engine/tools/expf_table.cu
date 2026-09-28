// expf_table -- the exception table that makes the device's expf bit-identical to MSVC's (the host samplers' expf).
// The device computes A = (float) exp((double) x); for every float x in [-104, -0] where A differs from MSVC's expf(x),
// the table holds (x bits, MSVC result bits), sorted by x bits. Written to expf_exc.bin (shipped in xyz-engine/): uint32
// count, then pairs.
// Below -104 both underflow to +0. Build: nvcc -O3 -arch=sm_89 -std=c++17 expf_table.cu (no fast math).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

__global__ void k_exp(const uint32_t first, const uint32_t n, float * a) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = first + i;
    float x;
    memcpy(&x, &u, 4);
    a[i] = (float) exp((double) x);
}

int main(int argc, char ** argv) {
    const char * out = argc > 1 ? argv[1] : "expf_exc.bin";
    const uint32_t first = 0x80000000u, last = 0xC2D00000u;   // -0 .. -104
    const uint32_t chunk = 1u << 26;
    float * da;
    cudaMalloc(&da, (size_t) chunk*4);
    std::vector<float> ha(chunk);
    std::vector<std::pair<uint32_t, uint32_t>> exc;
    uint64_t n_all = 0;
    for (uint64_t s = first; s <= last; s += chunk) {
        const uint32_t n = (uint32_t) std::min<uint64_t>(chunk, (uint64_t) last - s + 1);
        k_exp<<<(n + 255)/256, 256>>>((uint32_t) s, n, da);
        cudaMemcpy(ha.data(), da, (size_t) n*4, cudaMemcpyDeviceToHost);
        const int T = 12;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> part(T);
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) {
            th.emplace_back([&, t] {
                for (uint32_t i = t; i < n; i += T) {
                    const uint32_t u = (uint32_t) s + i;
                    float x;
                    memcpy(&x, &u, 4);
                    const float h = expf(x);
                    uint32_t hb, ab;
                    memcpy(&hb, &h, 4);
                    memcpy(&ab, &ha[i], 4);
                    if (hb != ab) part[t].push_back({u, hb});
                }
            });
        }
        for (auto & x : th) x.join();
        for (auto & p : part) exc.insert(exc.end(), p.begin(), p.end());
        n_all += n;
    }
    std::sort(exc.begin(), exc.end());
    FILE * f = fopen(out, "wb");
    const uint32_t cnt = (uint32_t) exc.size();
    fwrite(&cnt, 4, 1, f);
    for (const auto & e : exc) {
        fwrite(&e.first, 4, 1, f);
        fwrite(&e.second, 4, 1, f);
    }
    fclose(f);
    printf("floats in [-104, -0]: %llu, exceptions %u -> %s\n", (unsigned long long) n_all, cnt, out);
    return 0;
}
