// verify capture-safe config (partitioning disabled): correct sort + capture-safe on gfx1101
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <cstdlib>

using namespace hipcub;

using safe_config = rocprim::segmented_radix_sort_config<8, rocprim::kernel_config<256, 16>,
                                                         rocprim::DisabledWarpSortConfig>;

#define SAFE_SORT(stg, stg_bytes) \
    rocprim::segmented_radix_sort_pairs_desc<safe_config>(stg, stg_bytes, keys, keys, dst, dst, \
                                                          size, nrows, offsets, offsets + 1, \
                                                          0, sizeof(float) * 8, s)

__global__ void k_init(float * k, int * v, int * off, int n, int nrows) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        k[i] = (float) ((i * 2654435761u) % 100000) / 7.0f;  // pseudo-random
        v[i] = i;
    }
    if (i <= nrows) off[i] = i * 256;
}

__global__ void k_check(const float * k, const int * v, int n, int * bad) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i + 1 < n && (i + 1) % 256 != 0) {
        if (k[i] < k[i + 1]) atomicAdd(bad, 1);  // descending expected
    }
}

int main() {
    hipStream_t s;
    hipStreamCreate(&s);
    const int ncols = 256, nrows = 512;
    const int size = ncols * nrows;
    float * keys;
    int * dst, * offsets, * bad;
    void * stg;
    hipMalloc(&keys, size * sizeof(float));
    hipMalloc(&dst, size * sizeof(int));
    hipMalloc(&offsets, (nrows + 1) * sizeof(int));
    hipMalloc(&bad, sizeof(int));
    k_init<<<(size + 255) / 256, 256, 0, s>>>(keys, dst, offsets, size, nrows);

    size_t stg_bytes = 0;
    rocprim::segmented_radix_sort_pairs_desc<safe_config>(nullptr, stg_bytes, keys, keys, dst, dst,
                                                          size, nrows, offsets, offsets + 1,
                                                          0, sizeof(float) * 8, s);
    printf("safe stg=%zu (default was 1052928)\n", stg_bytes);
    hipMalloc(&stg, stg_bytes);

    // DIRECT run, then verify sortedness
    hipMemsetAsync(bad, 0, sizeof(int), s);
    rocprim::segmented_radix_sort_pairs_desc<safe_config>(stg, stg_bytes, keys, keys, dst, dst,
                                                          size, nrows, offsets, offsets + 1,
                                                          0, sizeof(float) * 8, s);
    k_check<<<(size + 255) / 256, 256, 0, s>>>(keys, dst, size, bad);
    int bad_h = -1;
    hipMemcpy(&bad_h, bad, sizeof(int), hipMemcpyDeviceToHost);
    hipStreamSynchronize(s);
    printf("direct: err=%s bad_pairs=%d\n", hipGetErrorString(hipGetLastError()), bad_h);

    // capture run
    hipGraph_t g = nullptr;
    hipError_t e = hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed);
    rocprim::segmented_radix_sort_pairs_desc<safe_config>(stg, stg_bytes, keys, keys, dst, dst,
                                                          size, nrows, offsets, offsets + 1,
                                                          0, sizeof(float) * 8, s);
    printf("in capture: %s\n", hipGetErrorString(hipGetLastError()));
    e = hipStreamEndCapture(s, &g);
    printf("end: %s\n", hipGetErrorString(e));

    if (e == hipSuccess && g) {
        hipGraphExec_t exec;
        e = hipGraphInstantiate(&exec, g, nullptr, nullptr, 0);
        printf("instantiate: %s\n", hipGetErrorString(e));
        e = hipGraphLaunch(exec, s);
        printf("launch: %s\n", hipGetErrorString(e));
        hipStreamSynchronize(s);
        printf("sync: %s\n", hipGetErrorString(hipGetLastError()));
        hipMemsetAsync(bad, 0, sizeof(int), s);
        k_check<<<(size + 255) / 256, 256, 0, s>>>(keys, dst, size, bad);
        hipMemcpy(&bad_h, bad, sizeof(int), hipMemcpyDeviceToHost);
        hipStreamSynchronize(s);
        printf("graph run bad_pairs=%d\n", bad_h);
    }
    return 0;
}
