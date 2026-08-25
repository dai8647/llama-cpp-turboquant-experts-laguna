#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_CUDA_USE_CUB
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows);
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_USE_CUB

#ifdef GGML_CUDA_USE_HIPCUB
int argsort_f32_i32_cuda_hipcub_chunk_nrows(const size_t nb01, const int64_t nrows);
void argsort_f32_i32_cuda_hipcub(ggml_cuda_pool & pool,
                                 const float *    x,
                                 int *            dst,
                                 const int        ncols,
                                 const int        nrows,
                                 ggml_sort_order  order,
                                 cudaStream_t     stream);
#endif  // GGML_CUDA_USE_HIPCUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);

// Pre-populate the pool with every workspace buffer the cub/hipcub argsort
// path will allocate for this op, so that a graph capture of the op never
// triggers a device malloc (illegal while a stream is capturing).
// No-op when the op would take the bitonic path.
void argsort_f32_i32_reserve_capture_pool(ggml_cuda_pool & pool, const ggml_tensor * src0);
