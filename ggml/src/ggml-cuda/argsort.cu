#include "argsort.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define STRIDED_ITERATOR_AVAILABLE
#        include <cuda/iterator>
#    endif
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

#ifdef GGML_CUDA_USE_HIPCUB
#    include <hipcub/hipcub.hpp>
using namespace hipcub;
#endif  // GGML_CUDA_USE_HIPCUB

static __global__ void init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#endif  // STRIDED_ITERATOR_AVAILABLE

#ifdef GGML_CUDA_USE_CUB

// returns the suggested maximum number of rows to process during one argsort_f32_i32_cuda_cub() call
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows) {
    // perform argsort in chunks up to approximately this size (currently 64MB)
    // to avoid excessive temporary buffers memory usage
    const int chunk_bytes = 1 << 26;

    // calculate how many rows will fit in one chunk (must be at least one)
    const int chunk_nrows = std::max((int) (chunk_bytes / nb01), 1);

    // limit the resulting amount to total nrows
    return std::min((int64_t) chunk_nrows, nrows);
}

void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
    // offset_iterator needs to populate nrows + 1 elements, so we also have to ceildiv nrows + 1 by block_size
    const int                 nrows_offset = nrows + 1;
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows_offset);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows_offset + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    bool is_capturing = false;
#ifdef USE_CUDA_GRAPH
    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does.
    // See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
    // TODO: constrain this to the CCCL versions that have this issue once it's resolved in a future CCCL release.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = (capture_status != cudaStreamCaptureStatusNone);
#endif  // USE_CUDA_GRAPH

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(
                nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                temp_indices, dst,                                  // values (indices)
                ncols * nrows, nrows,                               // num items, num segments
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys,
                                                      temp_keys,             // keys (in-place)
                                                      temp_indices, dst,     // values (indices)
                                                      ncols * nrows, nrows,  // num items, num segments
                                                      offset_iterator, offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                                temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys,          // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                           offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                      temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                      offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                                temp_keys, temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }
}
#endif  // GGML_CUDA_USE_CUB

#ifdef GGML_CUDA_USE_HIPCUB

// capture-safe segmented radix sort configuration: same kernel/radix parameters
// as the runtime default for float/int keys, but with the warp-sort partitioning
// disabled.  rocPRIM's partitioning path performs a device->host sync copy
// (memcpy_and_sync) which the graph capture runtime rejects.
using hipcub_capture_safe_sort_config = rocprim::segmented_radix_sort_config<
    8,
    rocprim::kernel_config<256, 16>,
    rocprim::DisabledWarpSortConfig>;

// hipcub provides the CUB API on ROCm/HIP, keeping top_k/argsort on the GPU
// for long score columns instead of falling back to CPU (llama.cpp #26399)
int argsort_f32_i32_cuda_hipcub_chunk_nrows(const size_t nb01, const int64_t nrows) {
    // perform argsort in chunks up to approximately this size (currently 64MB)
    // to avoid excessive temporary buffers memory usage
    const int chunk_bytes = 1 << 26;

    // calculate how many rows will fit in one chunk (must be at least one)
    const int chunk_nrows = std::max((int) (chunk_bytes / nb01), 1);

    // limit the resulting amount to total nrows
    return std::min((int64_t) chunk_nrows, nrows);
}

void argsort_f32_i32_cuda_hipcub(ggml_cuda_pool & pool,
                                 const float *    x,
                                 int *            dst,
                                 const int        ncols,
                                 const int        nrows,
                                 ggml_sort_order  order,
                                 cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<int>   offsets_alloc(pool, nrows + 1);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();
    int *   offsets      = offsets_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

    const int  nrows_offset = nrows + 1;
    const dim3 offset_grid((nrows_offset + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offsets, ncols, nrows);

    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // rocPRIM's segmented radix sort is not stream-capture compatible when its
    // partitioning path is active: the path performs a device->host sync copy
    // inside the call, which the capture runtime rejects.  gfx1101 builds always
    // hit that path for >= 64 segments because gfx1101 is missing from rocPRIM's
    // target_arch list and the unknown-arch fallback config uses
    // partitioning_threshold=64.  While capturing, fall back to the
    // unpartitioned sort (pure kernel launches, capture-safe); the replayed
    // graph then runs the recorded plain kernels.
    bool is_capturing = false;
#ifdef USE_CUDA_GRAPH
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = capture_status != cudaStreamCaptureStatusNone;
#endif // USE_CUDA_GRAPH

    size_t temp_storage_bytes = 0;

    if (nrows == 1) {
        // single row: plain radix sort is cheaper than the segmented variant
        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        }
    } else if (is_capturing) {
        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(rocprim::segmented_radix_sort_pairs<hipcub_capture_safe_sort_config>(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst,
                (unsigned int) (ncols * nrows), (unsigned int) nrows,
                offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(rocprim::segmented_radix_sort_pairs_desc<hipcub_capture_safe_sort_config>(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst,
                (unsigned int) (ncols * nrows), (unsigned int) nrows,
                offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        }
    } else {
        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows,
                                                           offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                                     temp_indices, dst, ncols * nrows, nrows,
                                                                     offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (nrows == 1) {
        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys,          // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        }
    } else if (is_capturing) {
        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(rocprim::segmented_radix_sort_pairs<hipcub_capture_safe_sort_config>(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst,
                (unsigned int) (ncols * nrows), (unsigned int) nrows,
                offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(rocprim::segmented_radix_sort_pairs_desc<hipcub_capture_safe_sort_config>(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst,
                (unsigned int) (ncols * nrows), (unsigned int) nrows,
                offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        }
    } else {
        if (order == GGML_SORT_ORDER_ASC) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows,
                                                           offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                                     temp_indices, dst, ncols * nrows, nrows,
                                                                     offsets, offsets + 1, 0, sizeof(float) * 8, stream));
        }
    }
}

#endif  // GGML_CUDA_USE_HIPCUB

// Bitonic sort implementation
template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

#ifdef GGML_CUDA_USE_CUB
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    // early return if we can use bitonic argsort
    if (shared_mem <= max_shared_mem && ncols <= 1024) {
        argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
        return;
    }

    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool & pool = ctx.pool();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        argsort_f32_i32_cuda_cub(pool, src0_d, (int *) dst_d, ncols, iter_nrows, order, stream);

        src0_d += ncols * iter_nrows;
        dst_d  += ncols * iter_nrows;
    }
#elif defined(GGML_CUDA_USE_HIPCUB)
    // always use hipcub on HIP: the bitonic kernel needs more shared memory
    // than AMD GPUs provide for large ncols (llama.cpp #24177)
    const int chunk_nrows = argsort_f32_i32_cuda_hipcub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool & pool = ctx.pool();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        argsort_f32_i32_cuda_hipcub(pool, src0_d, (int *) dst_d, ncols, iter_nrows, order, stream);

        src0_d += ncols * iter_nrows;
        dst_d  += ncols * iter_nrows;
    }
#else
    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
#endif
}

void argsort_f32_i32_reserve_capture_pool(ggml_cuda_pool & pool, const ggml_tensor * src0) {
    if (src0 == nullptr || src0->type != GGML_TYPE_F32 || !ggml_is_contiguous(src0)) {
        return;
    }

#ifdef GGML_CUDA_USE_CUB
    // mirror ggml_cuda_op_argsort: bitonic early-out means no pool workspaces
    const int64_t ncols = src0->ne[0];
    const size_t ncols_pad_bytes = next_power_of_2((int) ncols) * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    if (ncols_pad_bytes <= max_shared_mem && ncols <= 1024) {
        return;
    }

    const int64_t nrows = ggml_nrows(src0);
    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        size_t temp_storage_bytes = 0;
        // storage size is identical for both sort orders; take the max of the
        // plain and segmented variants to be safe
        DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, (float *) nullptr, (float *) nullptr,
                                   (int *) nullptr, (int *) nullptr, (int) (ncols * iter_nrows),
                                   0, sizeof(float) * 8);
        DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, (float *) nullptr, (float *) nullptr,
                                            (int *) nullptr, (int *) nullptr, (int) (ncols * iter_nrows), iter_nrows,
                                            (const int *) nullptr, (const int *) nullptr, 0, sizeof(float) * 8);

        // keys + indices live simultaneously with the storage inside one call
        ggml_cuda_pool_alloc<uint8_t> keys(pool, ncols * iter_nrows * sizeof(float));
        ggml_cuda_pool_alloc<uint8_t> idx(pool, ncols * iter_nrows * sizeof(int));
        ggml_cuda_pool_alloc<uint8_t> offs(pool, (iter_nrows + 1) * sizeof(int));
        ggml_cuda_pool_alloc<uint8_t> stg(pool, temp_storage_bytes);
    }
#elif defined(GGML_CUDA_USE_HIPCUB)
    // mirror ggml_cuda_op_argsort: hipcub is always used on HIP
    const int64_t nrows = ggml_nrows(src0);
    const int64_t ncols = src0->ne[0];
    const int chunk_nrows = argsort_f32_i32_cuda_hipcub_chunk_nrows(src0->nb[1], nrows);

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        // null-pointer sizing queries: pure host-side arithmetic in rocPRIM.
        // Reserve both the default (partitioning) and the capture-safe
        // (unpartitioned) storage sizes, since either can be requested at
        // capture time depending on the is_capturing branch in the op.
        size_t temp_storage_bytes = 0;
        DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, (float *) nullptr, (float *) nullptr,
                                            (int *) nullptr, (int *) nullptr, (int) (ncols * iter_nrows), iter_nrows,
                                            (const int *) nullptr, (const int *) nullptr, 0, sizeof(float) * 8);
        size_t capture_safe_storage_bytes = 0;
        rocprim::segmented_radix_sort_pairs<hipcub_capture_safe_sort_config>(
            nullptr, capture_safe_storage_bytes, (float *) nullptr, (float *) nullptr,
            (int *) nullptr, (int *) nullptr, (unsigned int) (ncols * iter_nrows), (unsigned int) iter_nrows,
            (const int *) nullptr, (const int *) nullptr, 0, sizeof(float) * 8);
        temp_storage_bytes = std::max(temp_storage_bytes, capture_safe_storage_bytes);

        ggml_cuda_pool_alloc<uint8_t> keys(pool, ncols * iter_nrows * sizeof(float));
        ggml_cuda_pool_alloc<uint8_t> idx(pool, ncols * iter_nrows * sizeof(int));
        ggml_cuda_pool_alloc<uint8_t> offs(pool, (iter_nrows + 1) * sizeof(int));
        ggml_cuda_pool_alloc<uint8_t> stg(pool, temp_storage_bytes);
    }
#else
    // bitonic-only build: no pool workspaces to reserve
    GGML_UNUSED(pool);
#endif
}
