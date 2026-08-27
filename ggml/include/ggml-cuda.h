#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// extension: background copy stream for MoE expert-slot prefetching.
// Copies issued here run on a dedicated stream that does not serialize with
// the compute stream; completion is tracked with the returned event handles.
// All functions accept NULL backend / return NULL when unsupported, so
// callers must fall back to synchronous copies.
GGML_API void * ggml_backend_cuda_ext_copy_stream(ggml_backend_t backend);
GGML_API void   ggml_backend_cuda_ext_h2d_async(ggml_backend_t backend, struct ggml_tensor * tensor,
                                                size_t offset, const void * host, size_t size);
GGML_API void * ggml_backend_cuda_ext_event_create(ggml_backend_t backend);
GGML_API void   ggml_backend_cuda_ext_event_record(ggml_backend_t backend, void * event);
GGML_API bool   ggml_backend_cuda_ext_event_query(void * event);
GGML_API void   ggml_backend_cuda_ext_event_synchronize(void * event);
GGML_API void   ggml_backend_cuda_ext_event_destroy(ggml_backend_t backend, void * event);

// extension: per-context switch for CUDA graph capture/replay.  MoE
// expert-slot paging regimes (q*/global-LRU) decide slot residency inside
// eval-time host ops; a captured graph would freeze those decisions at
// warmup values, so the caller disables graphs for its accel backends
// instead of exporting GGML_CUDA_DISABLE_GRAPHS globally.
GGML_API void   ggml_backend_cuda_ext_set_graphs_enabled(ggml_backend_t backend, bool enable);

// extension: pinned host buffer alloc/free for the MoE expert-slot bank
// staging path. returns NULL on failure (caller falls back to pageable H2D).
// on success *ext_backend_out is the first CUDA backend suitable for
// ggml_backend_cuda_ext_h2d_async; it may be NULL on CPU-only builds.
GGML_API void * ggml_backend_cuda_pinned_host_malloc(size_t size, ggml_backend_t * ext_backend_out);
GGML_API void   ggml_backend_cuda_pinned_host_free(void * ptr);

#ifdef  __cplusplus
}
#endif
