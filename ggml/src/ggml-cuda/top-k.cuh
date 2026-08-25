#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Pre-populate the pool with the workspaces ggml_cuda_op_top_k will allocate
// for this op (see argsort_f32_i32_reserve_capture_pool for the rationale).
void top_k_f32_i32_reserve_capture_pool(ggml_cuda_pool & pool, const ggml_tensor * src0, int64_t k);
