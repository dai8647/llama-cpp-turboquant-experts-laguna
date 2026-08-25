#include "llama.h"

#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"
#include "llama-moe-stats.h"

#include <string>

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

const char * llama_load_mode_name(enum llama_load_mode load_mode) {
    switch (load_mode) {
        case LLAMA_LOAD_MODE_NONE:
            return "none";
        case LLAMA_LOAD_MODE_MMAP:
            return "mmap";
        case LLAMA_LOAD_MODE_MLOCK:
            return "mlock";
        case LLAMA_LOAD_MODE_MMAP_MLOCK:
            return "mmap+mlock";
        case LLAMA_LOAD_MODE_DIRECT_IO:
            return "dio";
    }
    GGML_ABORT("fatal error");
}

enum llama_load_mode llama_load_mode_from_str(const char * str) {
    if (std::strcmp(str, "none") == 0)       { return LLAMA_LOAD_MODE_NONE;       }
    if (std::strcmp(str, "mmap") == 0)       { return LLAMA_LOAD_MODE_MMAP;       }
    if (std::strcmp(str, "mlock") == 0)      { return LLAMA_LOAD_MODE_MLOCK;      }
    if (std::strcmp(str, "mmap+mlock") == 0) { return LLAMA_LOAD_MODE_MMAP_MLOCK; }
    if (std::strcmp(str, "dio") == 0)        { return LLAMA_LOAD_MODE_DIRECT_IO;  }
    throw std::invalid_argument(std::string("unknown load mode: ") + str);
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// returns true on success
static bool llama_prepare_model_devices(const llama_model_params & params, llama_model * model) {
    // create list of devices to use with this model
    if (params.devices) {
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            size_t n_devs = 0;
            while (params.devices[n_devs]) {
                n_devs++;
            }
            if (n_devs == 0) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, n_devs);
            for (size_t i = 0; i < n_devs; ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s\n", __func__, i, ggml_backend_dev_name(params.devices[i]));
            }
            model->get_split_state_ud.n_devices = n_devs;
            model->get_split_state_ud.model = model;
            model->devices.push_back({
                true, ggml_backend_meta_device(
                params.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
                model->devices.push_back({false, *dev});
            }
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<llama_device> gpus;
        std::vector<llama_device> igpus;
        std::vector<llama_device> rpc_servers;

        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(ggml_backend_dev_count());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
                    LLAMA_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    continue;
                }
                devs.push_back(dev);
            }
            if (devs.empty()) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }

            LLAMA_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, devs.size());
            for (size_t i = 0; i < devs.size(); ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, i, ggml_backend_dev_name(devs[i]), ggml_backend_dev_description(devs[i]));
            }

            GGML_ASSERT(!devs.empty());
            model->get_split_state_ud.n_devices = devs.size();
            model->get_split_state_ud.model     = model;
            gpus.push_back({
                true, ggml_backend_meta_device(
                devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_servers.push_back({false, dev});
                        } else {
                            // check if there is already a GPU with the same device id
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(dev, &props);
                            auto it = std::find_if(gpus.begin(), gpus.end(), [&props](const llama_device & d) {
                                ggml_backend_dev_props d_props;
                                ggml_backend_dev_get_props(d.dev, &d_props);
                                if (props.device_id && d_props.device_id) {
                                    return strcmp(props.device_id, d_props.device_id) == 0;
                                }
                                return false;
                            });

                            if (it != gpus.end()) {
                                LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                        __func__,
                                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                        props.device_id ? props.device_id : "unknown id",
                                        ggml_backend_dev_name(it->dev), ggml_backend_dev_description(it->dev));
                            } else {
                                gpus.push_back({false, dev});
                            }
                        }
                        break;
                    }

                    case GGML_BACKEND_DEVICE_TYPE_IGPU:
                        if (igpus.empty()) {
                            igpus.push_back({false, dev});
                        }
                        break;
                    case GGML_BACKEND_DEVICE_TYPE_META:
                        GGML_ABORT("fatal error");
                }
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no discrete GPUs were found
        // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
        if (gpus.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE && !model->devices.empty()) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                return false;
            }
            llama_device main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (const auto & dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static bool llama_model_has_moe_experts(const llama_model & model) {
    for (const auto & layer : model.layers) {
        if (layer.ffn_gate_exps     ||
            layer.ffn_down_exps     ||
            layer.ffn_up_exps       ||
            layer.ffn_gate_up_exps  ||
            layer.ffn_gate_exps_b   ||
            layer.ffn_down_exps_b   ||
            layer.ffn_up_exps_b     ||
            layer.ffn_gate_up_exps_b) {
            return true;
        }
    }
    return false;
}

static const char * llama_moe_expert_layout_name(const llama_layer & layer) {
    const bool has_gate_up = layer.ffn_gate_up_exps || layer.ffn_gate_up_exps_b;
    const bool has_sep = layer.ffn_gate_exps || layer.ffn_up_exps || layer.ffn_down_exps ||
                         layer.ffn_gate_exps_b || layer.ffn_up_exps_b || layer.ffn_down_exps_b;

    if (has_gate_up && has_sep) {
        return "unknown";
    }
    if (has_gate_up) {
        return "fused";
    }
    if (has_sep) {
        return "separate";
    }
    return "unknown";
}

static int32_t llama_moe_expert_count_from_tensor(const ggml_tensor * t) {
    if (t == nullptr) {
        return -1;
    }

    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (t->ne[i] > 1) {
            return (int32_t) t->ne[i];
        }
    }

    return -1;
}

static int32_t llama_moe_expert_count_from_layer(const llama_layer & layer) {
    const ggml_tensor * candidates[] = {
        layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_up_exps, layer.ffn_gate_up_exps,
        layer.ffn_gate_exps_b, layer.ffn_down_exps_b, layer.ffn_up_exps_b, layer.ffn_gate_up_exps_b,
        layer.ffn_gate_exps_s, layer.ffn_down_exps_s, layer.ffn_up_exps_s,
    };

    for (const ggml_tensor * t : candidates) {
        const int32_t n = llama_moe_expert_count_from_tensor(t);
        if (n > 0) {
            return n;
        }
    }

    return -1;
}

struct llama_moe_layer_preload_info {
    int32_t layer_id = -1;
    int32_t n_experts = 0;
};

static int32_t llama_moe_gpu_expert_slot_effective_count(const llama_model & model, int32_t requested_slots) {
    if (requested_slots < 0 || model.hparams.n_expert == 0) {
        return 0;
    }

    const int32_t active_experts = (int32_t) model.hparams.n_expert_used;
    const int32_t total_experts  = (int32_t) model.hparams.n_expert;

    if (active_experts <= 0 || total_experts <= 0) {
        return 0;
    }

    if (requested_slots == 0) {
        return active_experts;
    }

    return std::clamp(requested_slots, active_experts, total_experts);
}

static ggml_backend_dev_t llama_moe_gpu_expert_slot_device(const llama_model & model) {
    for (const auto & dev : model.devices) {
        if (dev.is_meta) {
            continue;
        }
        if (ggml_backend_dev_buffer_type(dev.dev) != ggml_backend_cpu_buffer_type()) {
            return dev.dev;
        }
    }
    return nullptr;
}

static int32_t llama_moe_gpu_expert_dim(const ggml_tensor * src, int32_t n_experts) {
    if (src == nullptr || n_experts <= 0) {
        return -1;
    }

    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (src->ne[i] == n_experts) {
            return i;
        }
    }

    return -1;
}

static bool llama_moe_gpu_expert_slot_slice_shape(
        const ggml_tensor * src,
        int32_t n_experts,
        int32_t expert_id,
        int64_t dst_ne[GGML_MAX_DIMS],
        size_t & src_offset,
        size_t & src_nbytes) {
    if (src == nullptr || n_experts <= 0 || expert_id < 0 || expert_id >= n_experts) {
        return false;
    }

    const int32_t expert_dim = llama_moe_gpu_expert_dim(src, n_experts);
    if (expert_dim < 0 || expert_id >= src->ne[expert_dim]) {
        return false;
    }

    int dst_dim = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i == expert_dim) {
            continue;
        }
        dst_ne[dst_dim++] = src->ne[i];
    }
    while (dst_dim < GGML_MAX_DIMS) {
        dst_ne[dst_dim++] = 1;
    }

    src_offset = (size_t) expert_id * src->nb[expert_dim];
    src_nbytes = src->nb[expert_dim];
    return src_nbytes > 0;
}

static ggml_tensor * llama_moe_gpu_expert_bank_new_tensor(
        ggml_context * ctx,
        const ggml_tensor * src,
        int32_t expert_dim,
        int32_t n_slots) {
    int64_t ne[GGML_MAX_DIMS] = {
        src->ne[0],
        src->ne[1],
        src->ne[2],
        src->ne[3],
    };
    ne[expert_dim] = n_slots;

    if (ne[3] > 1) {
        return ggml_new_tensor_4d(ctx, src->type, ne[0], ne[1], ne[2], ne[3]);
    }
    if (ne[2] > 1) {
        return ggml_new_tensor_3d(ctx, src->type, ne[0], ne[1], ne[2]);
    }
    if (ne[1] > 1) {
        return ggml_new_tensor_2d(ctx, src->type, ne[0], ne[1]);
    }
    return ggml_new_tensor_1d(ctx, src->type, ne[0]);
}

static bool llama_moe_gpu_expert_bank_copy_tensor(
        const llama_moe_gpu_expert_bank_tensor & bank_tensor,
        int32_t expert_id,
        int32_t slot_id) {
    if (bank_tensor.src == nullptr || bank_tensor.dev == nullptr || bank_tensor.expert_dim < 0) {
        return false;
    }

    const size_t src_offset = (size_t) expert_id * bank_tensor.src->nb[bank_tensor.expert_dim];
    const size_t dst_offset = (size_t) slot_id   * bank_tensor.dev->nb[bank_tensor.expert_dim];
    const size_t nbytes     = bank_tensor.nbytes_per_expert;

    if (nbytes == 0 || src_offset + nbytes > ggml_nbytes(bank_tensor.src) || dst_offset + nbytes > ggml_nbytes(bank_tensor.dev)) {
        LLAMA_LOG_WARN("%s: MoE GPU expert bank tensor slice mismatch: %s expert=%d slot=%d bytes=%zu\n",
                __func__, bank_tensor.src->name, expert_id, slot_id, nbytes);
        return false;
    }

    // host-backed sources (model weights always are) can feed the H2D copy
    // directly; the staging detour only exists for exotic non-host buffers
    if (bank_tensor.src->buffer != nullptr && ggml_backend_buffer_is_host(bank_tensor.src->buffer)) {
        const uint8_t * host_ptr = (const uint8_t *) bank_tensor.src->data + src_offset;
        ggml_backend_tensor_set(bank_tensor.dev, host_ptr, dst_offset, nbytes);
        return true;
    }

    std::vector<uint8_t> data(nbytes);
    ggml_backend_tensor_get(bank_tensor.src, data.data(), src_offset, nbytes);
    ggml_backend_tensor_set(bank_tensor.dev, data.data(), dst_offset, nbytes);
    return true;
}

static bool llama_moe_gpu_expert_bank_ensure(
        llama_model & model,
        int32_t layer_id,
        int32_t n_experts) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.enabled()) {
        return false;
    }
    const int32_t n_bank_slots = std::min(cache.size(), n_experts);

    auto & bank = cache.bank_for_layer(layer_id);
    if (bank.buf && !bank.tensors.empty() && bank.n_experts == n_experts && bank.n_slots == n_bank_slots) {
        return true;
    }

    ggml_backend_dev_t dev = llama_moe_gpu_expert_slot_device(model);
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU backend device available for MoE GPU expert bank allocation\n", __func__);
        return false;
    }

    bank.clear_storage();
    bank.layer_id  = layer_id;
    bank.n_experts = n_experts;
    bank.n_slots   = n_bank_slots;

    const llama_layer & layer = model.layers.at(layer_id);
    const ggml_tensor * sources[] = {
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_up_exps,
        layer.ffn_gate_up_exps,
        layer.ffn_gate_exps_b,
        layer.ffn_down_exps_b,
        layer.ffn_up_exps_b,
        layer.ffn_gate_up_exps_b,
        layer.ffn_gate_exps_s,
        layer.ffn_down_exps_s,
        layer.ffn_up_exps_s,
    };

    int n_bank_tensors = 0;
    for (const ggml_tensor * src : sources) {
        if (llama_moe_gpu_expert_dim(src, n_experts) >= 0) {
            ++n_bank_tensors;
        }
    }
    if (n_bank_tensors == 0) {
        LLAMA_LOG_WARN("%s: no source tensors found for MoE GPU expert bank: layer=%d\n", __func__, layer_id);
        return false;
    }

    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_bank_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    bank.ctx.reset(ggml_init(ctx_params));
    if (!bank.ctx) {
        LLAMA_LOG_WARN("%s: failed to create MoE GPU expert bank context\n", __func__);
        bank.clear_storage();
        return false;
    }

    for (const ggml_tensor * src : sources) {
        const int32_t expert_dim = llama_moe_gpu_expert_dim(src, n_experts);
        if (expert_dim < 0) {
            continue;
        }

        ggml_tensor * dst = llama_moe_gpu_expert_bank_new_tensor(bank.ctx.get(), src, expert_dim, bank.n_slots);
        ggml_format_name(dst, "moe_slot_bank.%d.%s", (int) bank.tensors.size(), src->name);
        bank.tensors.push_back({ dst->name, const_cast<ggml_tensor *>(src), dst, expert_dim, (size_t) src->nb[expert_dim] });
    }
    if (bank.tensors.empty()) {
        bank.clear_storage();
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    bank.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(bank.ctx.get(), buft));
    if (!bank.buf) {
        LLAMA_LOG_WARN("%s: failed to allocate MoE GPU expert bank buffer: layer=%d slots=%d tensors=%zu\n",
                __func__, layer_id, bank.n_slots, bank.tensors.size());
        bank.clear_storage();
        return false;
    }

    for (const auto & bank_tensor : bank.tensors) {
        cache.register_compute_tensor(bank_tensor.src, bank_tensor.dev);
    }

    LLAMA_LOG_INFO("%s: MoE GPU expert slot bank allocated: layer=%d slots=%d tensors=%zu buffer=%.2f MiB device=%s\n",
            __func__, layer_id, bank.n_slots, bank.tensors.size(),
            ggml_backend_buffer_get_size(bank.buf.get()) / 1024.0 / 1024.0,
            ggml_backend_dev_name(dev));
    return true;
}

static ggml_tensor * llama_moe_gpu_expert_slot_new_tensor(
        ggml_context * ctx,
        const ggml_tensor * src,
        const int64_t ne[GGML_MAX_DIMS]) {
    if (ne[3] > 1) {
        return ggml_new_tensor_4d(ctx, src->type, ne[0], ne[1], ne[2], ne[3]);
    }
    if (ne[2] > 1) {
        return ggml_new_tensor_3d(ctx, src->type, ne[0], ne[1], ne[2]);
    }
    if (ne[1] > 1) {
        return ggml_new_tensor_2d(ctx, src->type, ne[0], ne[1]);
    }
    return ggml_new_tensor_1d(ctx, src->type, ne[0]);
}

static void llama_moe_gpu_expert_slot_add_tensor(
        llama_moe_gpu_expert_slot & slot,
        const ggml_tensor * src,
        int32_t n_experts,
        int32_t expert_id) {
    int64_t dst_ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    size_t src_offset = 0;
    size_t src_nbytes = 0;
    if (!llama_moe_gpu_expert_slot_slice_shape(src, n_experts, expert_id, dst_ne, src_offset, src_nbytes)) {
        return;
    }

    ggml_tensor * dst = llama_moe_gpu_expert_slot_new_tensor(slot.ctx.get(), src, dst_ne);
    ggml_format_name(dst, "moe_slot.%d.%s", (int) slot.tensors.size(), src->name);
    slot.tensors.push_back({ dst->name, const_cast<ggml_tensor *>(src), dst, src_nbytes });
}

static bool llama_moe_gpu_expert_slot_copy_tensor(
        const llama_moe_gpu_expert_slot_tensor & slot_tensor,
        int32_t n_experts,
        int32_t expert_id) {
    int64_t dst_ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    size_t src_offset = 0;
    size_t src_nbytes = 0;
    if (!llama_moe_gpu_expert_slot_slice_shape(slot_tensor.src, n_experts, expert_id, dst_ne, src_offset, src_nbytes)) {
        return false;
    }

    const size_t dst_nbytes = ggml_nbytes(slot_tensor.dev);
    if (src_nbytes != dst_nbytes || slot_tensor.nbytes != dst_nbytes) {
        LLAMA_LOG_WARN("%s: MoE GPU expert slot tensor size mismatch: %s source=%zu slot=%zu\n",
                __func__, slot_tensor.src->name, src_nbytes, dst_nbytes);
        return false;
    }

    std::vector<uint8_t> data(dst_nbytes);
    ggml_backend_tensor_get(slot_tensor.src, data.data(), src_offset, dst_nbytes);
    ggml_backend_tensor_set(slot_tensor.dev, data.data(), 0, dst_nbytes);
    return true;
}

static bool llama_moe_gpu_expert_slot_stats_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("LLAMA_MOE_SLOT_STATS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

static bool llama_moe_gpu_expert_slot_materialize(
        llama_model & model,
        int32_t slot_id,
        int32_t layer_id,
        int32_t expert_id,
        int32_t n_experts) {
    if (slot_id < 0 || slot_id >= model.moe_gpu_expert_cache.size()) {
        return false;
    }

    auto & cache = model.moe_gpu_expert_cache;

    // a background prefill copy may still be writing this slot's bank region
    // (the slot is deliberately non-resident while in flight, so find_free
    // can hand it to a new owner). wait for it before touching the region -
    // if the new owner is the same expert the copy already wrote the data.
    if (cache.prefill_pf_enabled) {
        std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
        for (auto it = cache.prefill_pf_inflight.begin(); it != cache.prefill_pf_inflight.end(); ++it) {
            if (it->layer_id != layer_id || it->slot_id != slot_id) {
                continue;
            }
            ggml_backend_cuda_ext_event_synchronize(it->event);
            ggml_backend_cuda_ext_event_destroy(cache.prefill_pf_backend, it->event);
            const bool matches = it->expert_id == expert_id;
            cache.prefill_pf_inflight.erase(it);
            if (matches) {
                auto * slot = cache.slot_at(layer_id, slot_id);
                if (slot != nullptr) {
                    slot->bank_backed = true;
                }
                return true;
            }
            break;
        }
    }

    ggml_backend_dev_t dev = llama_moe_gpu_expert_slot_device(model);
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU backend device available for MoE GPU expert slot allocation\n", __func__);
        return false;
    }

    auto * slot_ptr = model.moe_gpu_expert_cache.slot_at(layer_id, slot_id);
    if (slot_ptr == nullptr) {
        LLAMA_LOG_WARN("%s: invalid MoE GPU expert slot: layer=%d slot=%d expert=%d\n", __func__, layer_id, slot_id, expert_id);
        return false;
    }

    auto & slot = *slot_ptr;
    if (slot.resident &&
        slot.layer_id == layer_id &&
        slot.expert_id == expert_id) {
        const auto * bank = static_cast<const llama_moe_gpu_expert_cache &>(model.moe_gpu_expert_cache).bank_for_layer(layer_id);
        if ((slot.bank_backed && bank != nullptr && bank->buf && !bank->tensors.empty()) || (slot.buf && !slot.tensors.empty())) {
            return true;
        }
    }

    if (llama_moe_gpu_expert_bank_ensure(model, layer_id, n_experts)) {
        auto & bank = model.moe_gpu_expert_cache.bank_for_layer(layer_id);
        if (slot_id >= bank.n_slots) {
            return false;
        }

        size_t copied_bytes = 0;
        const bool stats  = llama_moe_gpu_expert_slot_stats_enabled();
        const auto stats_t0 = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (const auto & bank_tensor : bank.tensors) {
            if (llama_moe_gpu_expert_bank_copy_tensor(bank_tensor, expert_id, slot_id)) {
                copied_bytes += bank_tensor.nbytes_per_expert;
            }
        }
        if (stats) {
            auto & cache = model.moe_gpu_expert_cache;
            const int64_t copy_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - stats_t0).count();
            std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
            ++cache.n_copy;
            cache.copy_bytes += (int64_t) copied_bytes;
            cache.copy_ns    += copy_ns;
            if (cache.n_copy % 4096 == 0) {
                LLAMA_LOG_INFO("%s: MoE GPU slot stats: copies=%lld hit=%lld miss=%lld evict=%lld copy=%.1f MiB avg=%.2f ms\n",
                        __func__, (long long) cache.n_copy, (long long) cache.n_hit, (long long) cache.n_miss,
                        (long long) cache.n_evict, cache.copy_bytes / 1048576.0,
                        cache.copy_ns / 1e6 / (double) std::max<int64_t>(cache.n_copy, 1));
            }
        }

        slot.clear_storage();
        slot.layer_id  = layer_id;
        slot.expert_id = expert_id;
        slot.last_used = ++model.moe_gpu_expert_cache.clock;
        slot.resident  = true;
        slot.bank_backed = true;

        LLAMA_LOG_INFO("%s: MoE GPU expert slot bank materialized: slot=%d layer=%d expert=%d tensors=%zu bytes=%.2f MiB buffer=%.2f MiB device=%s\n",
                __func__, slot_id, layer_id, expert_id, bank.tensors.size(),
                copied_bytes / 1024.0 / 1024.0,
                ggml_backend_buffer_get_size(bank.buf.get()) / 1024.0 / 1024.0,
                ggml_backend_dev_name(dev));
        return true;
    }

    slot.clear_storage();

    const llama_layer & layer = model.layers.at(layer_id);
    const ggml_tensor * sources[] = {
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_up_exps,
        layer.ffn_gate_up_exps,
        layer.ffn_gate_exps_b,
        layer.ffn_down_exps_b,
        layer.ffn_up_exps_b,
        layer.ffn_gate_up_exps_b,
        layer.ffn_gate_exps_s,
        layer.ffn_down_exps_s,
        layer.ffn_up_exps_s,
    };

    int n_slot_tensors = 0;
    for (const ggml_tensor * src : sources) {
        int64_t dst_ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        size_t src_offset = 0;
        size_t src_nbytes = 0;
        if (llama_moe_gpu_expert_slot_slice_shape(src, n_experts, expert_id, dst_ne, src_offset, src_nbytes)) {
            ++n_slot_tensors;
        }
    }
    if (n_slot_tensors == 0) {
        LLAMA_LOG_WARN("%s: no source tensors found for MoE GPU expert slot: layer=%d expert=%d\n", __func__, layer_id, expert_id);
        return false;
    }

    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_slot_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    slot.ctx.reset(ggml_init(ctx_params));
    if (!slot.ctx) {
        LLAMA_LOG_WARN("%s: failed to create MoE GPU expert slot context\n", __func__);
        return false;
    }

    for (const ggml_tensor * src : sources) {
        llama_moe_gpu_expert_slot_add_tensor(slot, src, n_experts, expert_id);
    }
    if (slot.tensors.empty()) {
        slot.clear_storage();
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    slot.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(slot.ctx.get(), buft));
    if (!slot.buf) {
        LLAMA_LOG_WARN("%s: failed to allocate MoE GPU expert slot buffer: layer=%d expert=%d tensors=%zu\n",
                __func__, layer_id, expert_id, slot.tensors.size());
        slot.clear_storage();
        return false;
    }

    size_t copied_bytes = 0;
    for (const auto & slot_tensor : slot.tensors) {
        if (llama_moe_gpu_expert_slot_copy_tensor(slot_tensor, n_experts, expert_id)) {
            copied_bytes += slot_tensor.nbytes;
        }
    }

    LLAMA_LOG_INFO("%s: MoE GPU expert slot materialized: slot=%d layer=%d expert=%d tensors=%zu bytes=%.2f MiB buffer=%.2f MiB device=%s\n",
            __func__, slot_id, layer_id, expert_id, slot.tensors.size(),
            copied_bytes / 1024.0 / 1024.0,
            ggml_backend_buffer_get_size(slot.buf.get()) / 1024.0 / 1024.0,
            ggml_backend_dev_name(dev));
    return true;
}

static bool llama_moe_gpu_expert_slot_materialize_cb(
        void * userdata,
        int32_t slot_id,
        int32_t layer_id,
        int32_t expert_id,
        int32_t n_experts) {
    if (userdata == nullptr) {
        return false;
    }

    return llama_moe_gpu_expert_slot_materialize(
            *static_cast<llama_model *>(userdata), slot_id, layer_id, expert_id, n_experts);
}

void llama_moe_gpu_expert_slot_prefetch(struct llama_model & model, double budget_ms) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.enabled() || budget_ms <= 0.0) {
        return;
    }

    // runs between decode steps, while no graph is in flight, so plain
    // synchronous H2D copies cannot race with backend compute
    const auto t_start = std::chrono::steady_clock::now();
    for (const auto & [layer_id, ids] : cache.take_last_selections()) {
        const auto * bank = static_cast<const llama_moe_gpu_expert_cache &>(cache).bank_for_layer(layer_id);
        if (bank == nullptr) {
            continue;
        }

        for (const int32_t expert_id : ids) {
            const auto t_now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double, std::milli>(t_now - t_start).count() >= budget_ms) {
                return;
            }
            if (cache.find(layer_id, expert_id) >= 0) {
                continue;
            }

            int32_t slot = cache.preload_or_assign_slot(layer_id, expert_id, cache.next_clock());
            if (slot < 0) {
                continue;
            }
            if (!llama_moe_gpu_expert_slot_materialize(model, slot, layer_id, expert_id, bank->n_experts)) {
                cache.release_slot(layer_id, slot);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// prefill double buffering: background H2D copies on the backend's dedicated
// copy stream, issued between prefill ubatches while the previous graph is
// still running. prediction = per-layer router selections of the previous
// chunk (take_last_selections). opt-in: LLAMA_MOE_PREFILL_PF=1.
// ---------------------------------------------------------------------------

static void llama_moe_gpu_expert_slot_prefill_configure(llama_moe_gpu_expert_cache & cache) {
    if (const char * pf = getenv("LLAMA_MOE_PREFILL_PF")) {
        cache.prefill_pf_enabled = pf[0] != '\0' && pf[0] != '0';
    }
    if (const char * mb = getenv("LLAMA_MOE_PREFILL_PF_MB")) {
        const int64_t v = (int64_t) (atof(mb) * 1048576.0);
        if (v > 0) {
            cache.prefill_pf_budget_bytes = v;
        }
    }
    if (const char * n = getenv("LLAMA_MOE_PREFILL_PF_INFLIGHT")) {
        const int64_t v = atoll(n);
        if (v > 0) {
            cache.prefill_pf_max_inflight = v;
        }
    }
    if (cache.prefill_pf_enabled) {
        LLAMA_LOG_INFO("%s: MoE prefill double buffering enabled (budget=%.2f MiB/ubatch, max_inflight=%lld)\n",
                __func__, cache.prefill_pf_budget_bytes / 1048576.0,
                (long long) cache.prefill_pf_max_inflight);
    }
}

void llama_moe_gpu_expert_slot_prefill_prefetch(struct llama_model & model, struct ggml_backend * backend) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.enabled() || !cache.prefill_pf_enabled || backend == nullptr) {
        return;
    }
    if (ggml_backend_cuda_ext_copy_stream(backend) == nullptr) {
        return; // unsupported backend: misses stay synchronous
    }
    cache.prefill_pf_backend = backend;

    // 1. poll background copies: completed copies flip their slot to resident
    {
        std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
        auto & inflight = cache.prefill_pf_inflight;
        for (auto it = inflight.begin(); it != inflight.end(); ) {
            if (!ggml_backend_cuda_ext_event_query(it->event)) {
                ++it;
                continue;
            }
            auto * slot = cache.slot_at(it->layer_id, it->slot_id);
            if (slot != nullptr && slot->layer_id == it->layer_id &&
                slot->expert_id == it->expert_id && !slot->resident) {
                slot->resident = true; // n_resident_global was bumped at assign
            }
            ggml_backend_cuda_ext_event_destroy(backend, it->event);
            it = inflight.erase(it);
        }
    }

    // 2. enqueue predicted experts of the next chunk. assignment, the
    // non-resident flip and the copy enqueue happen under one cache_mutex hold
    // so the eval remap can never observe a resident-but-unwritten slot. the
    // hold is per expert (pageable weight copies block the host), keeping lock
    // waits of concurrently running remap ops bounded.
    int64_t enqueued_bytes = 0;
    for (const auto & [layer_id, ids] : cache.take_last_selections()) {
        if (enqueued_bytes >= cache.prefill_pf_budget_bytes ||
            (int64_t) cache.prefill_pf_inflight.size() >= cache.prefill_pf_max_inflight) {
            return;
        }

        const llama_layer & layer = model.layers.at(layer_id);
        const int32_t n_experts = llama_moe_expert_count_from_layer(layer);
        if (n_experts <= 0) {
            continue;
        }

        const auto * bank = static_cast<const llama_moe_gpu_expert_cache &>(cache).bank_for_layer(layer_id);
        if (bank == nullptr || bank->tensors.empty()) {
            {
                std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
                if (!llama_moe_gpu_expert_bank_ensure(model, layer_id, n_experts)) {
                    continue;
                }
            }
            bank = static_cast<const llama_moe_gpu_expert_cache &>(cache).bank_for_layer(layer_id);
            if (bank == nullptr || bank->tensors.empty()) {
                continue;
            }
        }

        // all bank tensors must be host-backed to feed a straight H2D
        bool host_backed = true;
        for (const auto & bt : bank->tensors) {
            if (bt.src->buffer == nullptr || !ggml_backend_buffer_is_host(bt.src->buffer)) {
                host_backed = false;
                break;
            }
        }
        if (!host_backed) {
            continue; // sync materialize will handle it at remap time
        }

        for (const int32_t expert_id : ids) {
            if (expert_id < 0 || expert_id >= n_experts) {
                continue;
            }
            if (enqueued_bytes >= cache.prefill_pf_budget_bytes ||
                (int64_t) cache.prefill_pf_inflight.size() >= cache.prefill_pf_max_inflight) {
                return;
            }

            std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
            if (cache.find(layer_id, expert_id) >= 0) {
                continue;
            }
            bool in_flight = false;
            for (const auto & c : cache.prefill_pf_inflight) {
                if (c.layer_id == layer_id && c.expert_id == expert_id) {
                    in_flight = true;
                    break;
                }
            }
            if (in_flight) {
                continue;
            }

            int32_t slot = cache.preload_or_assign_slot(layer_id, expert_id, cache.next_clock());
            if (slot < 0) {
                continue;
            }
            auto * s = cache.slot_at(layer_id, slot);
            if (s == nullptr) {
                continue;
            }

            // find_free hands out non-resident slots, so the assigned slot may
            // still own an in-flight copy of a different expert: writing the
            // new expert over it would corrupt the pending copy, and a later
            // materialize drain for the old expert would then see the wrong
            // data. release the slot and skip - the sync path takes over.
            bool slot_busy = false;
            for (const auto & c : cache.prefill_pf_inflight) {
                if (c.layer_id == layer_id && c.slot_id == slot && c.expert_id != expert_id) {
                    slot_busy = true;
                    break;
                }
            }
            if (slot_busy) {
                cache.release_slot(layer_id, slot);
                continue;
            }

            void * event = ggml_backend_cuda_ext_event_create(backend);
            if (event == nullptr) {
                cache.release_slot(layer_id, slot);
                continue;
            }
            for (const auto & bt : bank->tensors) {
                const uint8_t * host = (const uint8_t *) bt.src->data +
                        (size_t) expert_id * bt.src->nb[bt.expert_dim];
                const size_t dst_offset = (size_t) slot * bt.dev->nb[bt.expert_dim];
                ggml_backend_cuda_ext_h2d_async(backend, bt.dev, dst_offset, host, bt.nbytes_per_expert);
                enqueued_bytes += bt.nbytes_per_expert;
            }
            ggml_backend_cuda_ext_event_record(backend, event);

            // non-resident until the copy completes: find() must miss so the
            // graph never reads an unwritten bank region
            s->resident = false;
            cache.prefill_pf_inflight.push_back({ layer_id, slot, expert_id, event });
        }
    }
}

void llama_moe_gpu_expert_slot_prefill_shutdown(struct llama_model & model) {
    auto & cache = model.moe_gpu_expert_cache;
    if (cache.prefill_pf_inflight.empty()) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
    for (auto & c : cache.prefill_pf_inflight) {
        ggml_backend_cuda_ext_event_synchronize(c.event);
        ggml_backend_cuda_ext_event_destroy(cache.prefill_pf_backend, c.event);
    }
    cache.prefill_pf_inflight.clear();
}

// ---------------------------------------------------------------------------
// elastic VRAM sizing: resolve --moe-gpu-expert-slot-num auto from free VRAM
// after model weights + KV caches + compute buffers are resident. the cache
// was left uninitialized at load time (auto_pending); this is the deferred
// half of the load-time init block.
// ---------------------------------------------------------------------------

static void llama_moe_gpu_expert_slot_preload(const llama_model & model); // defined below (shared with manual init)

void llama_moe_gpu_expert_slot_auto_init(struct llama_model & model) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.auto_pending) {
        return;
    }
    cache.auto_pending = false;

    if (model.hparams.n_expert <= 0 || !llama_model_has_moe_experts(model)) {
        LLAMA_LOG_INFO("%s: MoE GPU expert slot auto sizing skipped: model has no MoE experts\n", __func__);
        cache.clear();
        return;
    }

    ggml_backend_dev_t dev = llama_moe_gpu_expert_slot_device(model);
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: MoE GPU expert slot auto sizing skipped: no GPU backend device\n", __func__);
        cache.clear();
        return;
    }

    size_t vram_free = 0;
    size_t vram_total = 0;
    ggml_backend_dev_memory(dev, &vram_free, &vram_total);

    // one-expert cost across ALL banks: each layer bank holds n_slots copies
    // of every expert tensor, so a slot's footprint is the sum of per-layer
    // per-expert strides
    int64_t unit_bytes = 0;
    const int32_t n_experts = (int32_t) model.hparams.n_expert;
    for (const llama_layer & layer : model.layers) {
        const ggml_tensor * sources[] = {
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_up_exps,
            layer.ffn_gate_up_exps,
            layer.ffn_gate_exps_b,
            layer.ffn_down_exps_b,
            layer.ffn_up_exps_b,
            layer.ffn_gate_up_exps_b,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s,
            layer.ffn_up_exps_s,
        };
        for (const ggml_tensor * src : sources) {
            const int32_t expert_dim = llama_moe_gpu_expert_dim(src, n_experts);
            if (expert_dim >= 0) {
                unit_bytes += (int64_t) src->nb[expert_dim];
            }
        }
    }
    if (unit_bytes <= 0) {
        LLAMA_LOG_WARN("%s: MoE GPU expert slot auto sizing skipped: could not compute expert unit cost\n", __func__);
        cache.clear();
        return;
    }

    double margin = 0.80;
    if (const char * m = getenv("LLAMA_MOE_AUTO_VMARGIN_FRACTION")) {
        margin = atof(m);
        margin = std::clamp(margin, 0.0, 1.0);
    }
    const int64_t budget = (int64_t) ((double) vram_free * margin);

    int32_t slots_max = (int32_t) model.hparams.n_expert;
    if (const char * cap = getenv("LLAMA_MOE_AUTO_SLOT_CAP")) {
        const int32_t v = atoi(cap);
        if (v > 0) {
            slots_max = std::min(slots_max, v);
        }
    }
    const int32_t slots_min = (int32_t) model.hparams.n_expert_used;
    int32_t slots = (int32_t) (budget / unit_bytes);
    slots = std::clamp(slots, slots_min, slots_max);

    LLAMA_LOG_INFO("%s: MoE GPU expert slot auto sizing: free_vram=%.2f MiB total_vram=%.2f MiB margin=%.2f budget=%.2f MiB unit=%.2f MiB/slot slots=%d (clamped to [%d, %d])\n",
            __func__, vram_free / 1048576.0, vram_total / 1048576.0, margin,
            budget / 1048576.0, unit_bytes / 1048576.0, slots, slots_min, slots_max);

    if (slots <= 0) {
        LLAMA_LOG_INFO("%s: MoE GPU expert slot auto sizing: budget too small, slot mode disabled\n", __func__);
        cache.clear();
        return;
    }

    cache.init(slots);
    cache.materialize_cb       = llama_moe_gpu_expert_slot_materialize_cb;
    cache.materialize_userdata = (llama_model *) &model;
    cache.owner_model          = &model;

    // same env knobs as the manual path
    if (const char * glru = getenv("LLAMA_MOE_GLOBAL_LRU")) {
        cache.global_lru_enabled = glru[0] != '\0' && glru[0] != '0';
    }
    if (const char * pf = getenv("LLAMA_MOE_PREFETCH_MS")) {
        cache.prefetch_budget_ms = atof(pf);
    }
    if (const char * qs = getenv("LLAMA_MOE_QSTAR")) {
        cache.qstar_enabled = qs[0] != '\0' && qs[0] != '0';
    }
    if (const char * qt = getenv("LLAMA_MOE_QSTAR_THREADS")) {
        cache.qstar_threads = std::max(1, atoi(qt));
    }
    if (const char * qb = getenv("LLAMA_MOE_QSTAR_BUDGET_US")) {
        cache.qstar_budget_us = atof(qb);
    }
    if (const char * qst = getenv("LLAMA_MOE_QSTAR_STATS")) {
        cache.qstar_stats = qst[0] != '\0' && qst[0] != '0';
    }
    if (cache.qstar_enabled) {
        LLAMA_LOG_INFO("%s: q* bandwidth-adaptive policy enabled (threads=%d budget=%.0fus)\n",
                __func__, cache.qstar_threads, cache.qstar_budget_us);
        llama_moe_gpu_qstar_calibrate(model);
    }
    llama_moe_gpu_expert_slot_prefill_configure(cache);

    llama_moe_gpu_expert_slot_preload(model);
    LLAMA_LOG_INFO("%s: MoE GPU expert slot cache initialized with %d slots (auto)\n", __func__, slots);
}

// ---------------------------------------------------------------------------
// q* bandwidth-adaptive policy: host-side expert execution engine.
// one tiny static ggml graph per MoE layer; weight tensors alias the original
// host weight storage (zero copies), activations live in an owned scratch
// block. computed on a dedicated threadpool so it can nest inside the eval-
// time remap custom op without touching the outer inference pool.
// ---------------------------------------------------------------------------

// forward decl: the auto-init path above calls this before its definition
static void llama_moe_gpu_expert_slot_preload(const llama_model & model);

// metadata-only tensor aliasing src's storage, strides and type; requires a
// 3d [ne0, ne1, n_experts] layout (mul_mat_id contract)
static ggml_tensor * llama_moe_qstar_alias_weight(ggml_context * ctx, const ggml_tensor * src) {
    const int64_t ne[3] = { src->ne[0], src->ne[1], src->ne[2] };
    ggml_tensor * t = ggml_new_tensor(ctx, src->type, 3, ne);
    memcpy(t->nb, src->nb, sizeof(t->nb));
    t->data = src->data;
    return t;
}

static ggml_tensor * llama_moe_qstar_owned_tensor(
        ggml_context * ctx,
        ggml_type       type,
        int64_t         n0,
        int64_t         n1,
        int64_t         n2,
        std::vector<uint8_t> & mem,
        size_t & off) {
    const int64_t ne[3] = { n0, n1, n2 };
    ggml_tensor * t = ggml_new_tensor(ctx, type, 3, ne);
    const size_t nbytes = ggml_nbytes(t);
    if (off + nbytes > mem.size()) {
        mem.resize(off + nbytes);
    }
    t->data = mem.data() + off;
    off += nbytes;
    return t;
}

// place an op-produced tensor's storage inside the engine scratch block
static void llama_moe_qstar_own_node(ggml_tensor * t, std::vector<uint8_t> & mem, size_t & off) {
    const size_t nbytes = ggml_nbytes(t);
    if (off + nbytes > mem.size()) {
        mem.resize(off + nbytes);
    }
    t->data = mem.data() + off;
    off += nbytes;
}

static int64_t llama_moe_qstar_expert_bytes(const llama_moe_qstar_layer_exec & ex) {
    return (int64_t)(ex.fused_gate_up ? ex.gateup_stride + ex.down_stride
                                      : ex.gate_stride + ex.up_stride + ex.down_stride);
}

static bool llama_moe_qstar_exec_build(llama_model & model, int32_t layer_id) {
    auto & cache = model.moe_gpu_expert_cache;
    std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);

    llama_moe_qstar_layer_exec & ex = cache.qstar_exec_by_layer[layer_id];
    if (ex.ready) {
        return ex.supported;
    }
    ex.ready = true;

    const llama_layer & layer = model.layers.at(layer_id);
    const int32_t n_experts = (int32_t) model.hparams.n_expert;
    if (n_experts <= 1 || model.hparams.n_expert_used < 2) {
        ex.supported = false;
        return false;
    }

    // biases would need per-expert adds inside the graph; non-scalar scales
    // would need per-expert multiplies. both fall back to transfer-always.
    const ggml_tensor * bias_sources[] = {
        layer.ffn_gate_exps_b, layer.ffn_down_exps_b, layer.ffn_up_exps_b, layer.ffn_gate_up_exps_b,
    };
    for (const ggml_tensor * b : bias_sources) {
        if (b != nullptr) {
            ex.supported = false;
            return false;
        }
    }
    auto scalar_scale = [](const ggml_tensor * s) {
        return s == nullptr || ggml_nelements(s) == 1;
    };
    // no fused gate_up scale tensor exists: merged layouts carry their scale
    // in ffn_up_exps_s (the same tensor build_moe_ffn hands the fused matmul)
    if (!scalar_scale(layer.ffn_gate_exps_s) || !scalar_scale(layer.ffn_up_exps_s) ||
        !scalar_scale(layer.ffn_down_exps_s)) {
        ex.supported = false;
        return false;
    }

    const ggml_tensor * w_down = layer.ffn_down_exps;
    if (w_down == nullptr || llama_moe_gpu_expert_dim(w_down, n_experts) != 2) {
        ex.supported = false;
        return false;
    }
    const int32_t n_ff   = (int32_t) w_down->ne[0];
    const int32_t n_embd = (int32_t) w_down->ne[1];

    const ggml_tensor * w_gate_up = layer.ffn_gate_up_exps;
    const ggml_tensor * w_gate    = layer.ffn_gate_exps;
    const ggml_tensor * w_up      = layer.ffn_up_exps;
    const bool fused = w_gate_up != nullptr;
    const ggml_tensor * w_in0 = fused ? w_gate_up : w_gate; // [n_embd, n_ff or 2*n_ff, E]
    const ggml_tensor * w_in1 = fused ? w_gate_up : w_up;

    for (const ggml_tensor * w : { w_in0, w_in1 }) {
        if (w == nullptr || llama_moe_gpu_expert_dim(w, n_experts) != 2 ||
            w->ne[0] != n_embd || w->type != w_down->type || w->ne[3] != 1) {
            LLAMA_LOG_WARN("%s: q* host exec unsupported at layer %d: incompatible expert tensor layout\n",
                    __func__, layer_id);
            ex.supported = false;
            return false;
        }
    }

    ex.n_ff          = n_ff;
    ex.n_embd        = n_embd;
    ex.r_max         = (size_t) model.hparams.n_expert_used;
    ex.fused_gate_up = fused;
    ex.wtype         = w_down->type;
    ex.gate_data     = fused ? nullptr : w_gate->data;
    ex.up_data       = fused ? nullptr : w_up->data;
    ex.down_data     = w_down->data;
    ex.gateup_data   = fused ? w_gate_up->data : nullptr;
    ex.gate_stride   = fused ? 0 : w_gate->nb[2];
    ex.up_stride     = fused ? 0 : w_up->nb[2];
    ex.down_stride   = w_down->nb[2];
    ex.gateup_stride = fused ? w_gate_up->nb[2] : 0;
    if (layer.ffn_gate_exps_s) ex.scale_gate = ggml_get_f32_1d(layer.ffn_gate_exps_s, 0);
    if (layer.ffn_up_exps_s)   ex.scale_up   = ggml_get_f32_1d(layer.ffn_up_exps_s, 0);
    if (layer.ffn_down_exps_s) ex.scale_down = ggml_get_f32_1d(layer.ffn_down_exps_s, 0);

    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*32 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ex.ctx.reset(ggml_init(ctx_params));
    if (!ex.ctx) {
        ex.supported = false;
        return false;
    }
    ggml_context * ctx = ex.ctx.get();

    // scratch plan (all f32): gout/uout or fused gu [n_ff|2*n_ff, r], h
    // [n_ff, r], y [n_embd, r], x [n_embd]
    size_t off = 0;
    const size_t r_max = ex.r_max;

    // expert-id lists: metadata only, data pointers swapped per call; [r,1]
    // and [1,r] share one linear buffer (r consecutive i32 either way)
    int64_t ne_ids_a[2] = { (int64_t) r_max, 1 };
    int64_t ne_ids_b[2] = { 1, (int64_t) r_max };
    ggml_tensor * t_ids_up   = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, ne_ids_a);
    ggml_tensor * t_ids_down = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, ne_ids_b);

    ggml_tensor * w_in0_t  = llama_moe_qstar_alias_weight(ctx, w_in0);
    ggml_tensor * w_in1_t  = fused ? w_in0_t : llama_moe_qstar_alias_weight(ctx, w_up);
    ggml_tensor * w_down_t = llama_moe_qstar_alias_weight(ctx, w_down);

    ggml_tensor * t_x = llama_moe_qstar_owned_tensor(ctx, GGML_TYPE_F32, n_embd, 1, 1, ex.mem, off);

    // stage 1: gate/up projections of all r experts against the shared token
    ggml_tensor * gout = ggml_mul_mat_id(ctx, w_in0_t, t_x, t_ids_up);      // [n_ff, r, 1]
    ggml_tensor * uout = fused ? gout : ggml_mul_mat_id(ctx, w_in1_t, t_x, t_ids_up);
    ggml_tensor * gate_v = gout;
    ggml_tensor * up_v   = uout;
    ggml_tensor * h      = nullptr;
    if (fused) {
        // split merged [2*n_ff, r, 1] output into gate/up halves
        gate_v = ggml_view_3d(ctx, gout, n_ff, r_max, 1, gout->nb[1], gout->nb[2], 0);
        up_v   = ggml_view_3d(ctx, gout, n_ff, r_max, 1, gout->nb[1], gout->nb[2], n_ff*gout->nb[0]);
    }
    h = ggml_swiglu_split(ctx, gate_v, up_v);                               // [n_ff, r, 1]

    // stage 2: down projection; h reinterpreted as r column vectors so each
    // expert multiplies exactly its own hidden activation
    ggml_tensor * h_cols = ggml_view_3d(ctx, h, n_ff, 1, r_max, sizeof(float), h->nb[1], 0);
    ggml_tensor * t_y = ggml_mul_mat_id(ctx, w_down_t, h_cols, t_ids_down); // [n_embd, r, 1]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_y);

    // assign scratch storage to every computed node (weights alias sources,
    // ids swap per call, everything else lands in ex.mem)
    llama_moe_qstar_own_node(gout, ex.mem, off);
    if (!fused) {
        llama_moe_qstar_own_node(uout, ex.mem, off);
    } else {
        GGML_UNUSED(w_in1_t);
    }
    llama_moe_qstar_own_node(h, ex.mem, off);
    llama_moe_qstar_own_node(t_y, ex.mem, off);

    ex.gf    = gf;
    ex.t_ids = t_ids_up;
    ex.t_ids_down = t_ids_down;
    ex.t_x   = t_x;
    ex.t_y   = t_y;
    cache.qstar_expert_bytes = llama_moe_qstar_expert_bytes(ex);

    LLAMA_LOG_INFO("%s: q* host exec engine ready: layer=%d n_embd=%d n_ff=%d r=%zu fused=%d type=%d bytes/expert=%.2f MiB\n",
            __func__, layer_id, n_embd, n_ff, r_max, fused ? 1 : 0, (int) ex.wtype,
            cache.qstar_expert_bytes / 1048576.0);
    return true;
}

bool llama_moe_gpu_qstar_cpu_exec(struct llama_model & model, int32_t layer_id,
        const int32_t * experts, int32_t n, const float * x, float * partials_out) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.qstar_enabled || n <= 0 || x == nullptr || partials_out == nullptr) {
        return false;
    }

    if (!llama_moe_qstar_exec_build(model, layer_id)) {
        return false;
    }

    std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
    const auto it = cache.qstar_exec_by_layer.find(layer_id);
    if (it == cache.qstar_exec_by_layer.end() || !it->second.supported || it->second.gf == nullptr) {
        return false;
    }
    llama_moe_qstar_layer_exec & ex = it->second;

    const int32_t r = std::min<int32_t>(n, (int32_t) ex.r_max);
    std::vector<int32_t> ids_buf(r);
    for (int32_t i = 0; i < r; ++i) {
        ids_buf[i] = experts[i];
    }
    ex.t_ids->data           = ids_buf.data(); // [r, 1]
    ex.t_ids_down->data      = ids_buf.data(); // [1, r]: same linear layout
    memcpy(ex.t_x->data, x, ex.n_embd*sizeof(float));

    // dedicated nested-compute pool; falls back to a single inline worker
    if (cache.qstar_tp == nullptr && cache.qstar_threads > 1) {
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(cache.qstar_threads);
        cache.qstar_tp = ggml_threadpool_new(&tpp);
    }

    struct ggml_cplan cplan = ggml_graph_plan(ex.gf,
            cache.qstar_tp != nullptr ? cache.qstar_threads : 1, cache.qstar_tp);
    if (cplan.work_size > ex.work.size()) {
        ex.work.resize(cplan.work_size);
    }
    cplan.work_data = ex.work.data();

    const int64_t bytes = (int64_t) r * cache.qstar_expert_bytes;
    const auto t0 = std::chrono::steady_clock::now();
    const enum ggml_status st = ggml_graph_compute(ex.gf, &cplan);
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
    if (st != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_WARN("%s: q* host exec graph failed at layer %d: %d\n", __func__, layer_id, (int) st);
        return false;
    }

    if (ns > 0 && bytes > 0) {
        const double bps = bytes / (ns / 1e9);
        cache.qstar_cpu_bps = cache.qstar_cpu_bps <= 0 ? bps : 0.875*cache.qstar_cpu_bps + 0.125*bps;
    }

    // accumulate unweighted outputs: y[i*n_embd + d], folded with scalar scales
    const float s = ex.scale_gate*ex.scale_up*ex.scale_down;
    const float * y = (const float *) ex.t_y->data;
    for (int32_t i = 0; i < r; ++i) {
        float * out = partials_out + (size_t) i*ex.n_embd;
        const float * yi = y + (size_t) i*ex.n_embd;
        for (int32_t d = 0; d < ex.n_embd; ++d) {
            out[d] += yi[d]*s;
        }
    }
    return true;
}

bool llama_moe_qstar_exec_prepare(struct llama_model & model, int32_t layer_id) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.qstar_enabled) {
        return false;
    }
    if (!llama_moe_qstar_exec_build(model, layer_id)) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(cache.cache_mutex);
    const auto it = cache.qstar_exec_by_layer.find(layer_id);
    return it != cache.qstar_exec_by_layer.end() && it->second.supported && it->second.gf != nullptr;
}

void llama_moe_gpu_qstar_calibrate(struct llama_model & model) {
    auto & cache = model.moe_gpu_expert_cache;
    if (!cache.enabled() || !cache.qstar_enabled) {
        return;
    }

    const int32_t n_experts = (int32_t) model.hparams.n_expert;
    int32_t layer_id = -1;
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const llama_layer & l = model.layers[i];
        if (l.ffn_down_exps && (l.ffn_gate_exps || l.ffn_gate_up_exps)) {
            layer_id = (int32_t) i;
            break;
        }
    }
    if (layer_id < 0) {
        return;
    }

    // build the host-exec engine first: it reports per-expert weight bytes
    // that the H2D probe below relies on
    llama_moe_qstar_exec_prepare(model, layer_id);

    // effective H2D bandwidth: recopy a real expert slice through the direct
    // host->VRAM path three times, keep the best rate
    const int32_t probe = n_experts - 1;
    {
        const int32_t slot = cache.preload_or_assign_slot(layer_id, probe, cache.next_clock());
        if (slot >= 0 && llama_moe_gpu_expert_bank_ensure(model, layer_id, n_experts)) {
            auto & bank = cache.bank_for_layer(layer_id);
            if (slot < bank.n_slots) {
                double best_bps = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (const auto & bt : bank.tensors) {
                        llama_moe_gpu_expert_bank_copy_tensor(bt, probe, slot);
                    }
                    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
                    const int64_t bytes = (int64_t) cache.qstar_expert_bytes;
                    if (ns > 0) {
                        best_bps = std::max(best_bps, bytes / (ns / 1e9));
                    }
                }
                if (best_bps > 0) {
                    cache.qstar_h2d_bps = best_bps;
                }
            }
        } else {
            cache.release_slot(layer_id, slot);
        }
    }

    // host GEMV rate: three warmup executions also seed the runtime EMA
    {
        const int32_t n_embd = (int32_t) model.layers[layer_id].ffn_down_exps->ne[1];
        const int32_t one = probe;
        std::vector<float> x(n_embd, 0.5f);
        std::vector<float> partials((size_t) model.hparams.n_expert_used*n_embd, 0.0f);
        for (int rep = 0; rep < 3; ++rep) {
            if (!llama_moe_gpu_qstar_cpu_exec(model, layer_id, &one, 1, x.data(), partials.data())) {
                break;
            }
        }
    }

    LLAMA_LOG_INFO("%s: q* calibrated: h2d=%.1f GB/s cpu=%.1f GB/s expert=%.2f MiB threads=%d budget_us=%.0f\n",
            __func__,
            cache.qstar_h2d_bps / 1e9,
            cache.qstar_cpu_bps / 1e9,
            cache.qstar_expert_bytes / 1048576.0,
            cache.qstar_threads,
            cache.qstar_budget_us);
}

static void llama_moe_gpu_expert_slot_preload(const llama_model & model) {
    auto & cache = const_cast<llama_model &>(model).moe_gpu_expert_cache;
    llama_model & model_mut = const_cast<llama_model &>(model);
    std::vector<llama_moe_layer_preload_info> moe_layers;
    int64_t step = 1;
    int32_t max_layer_experts = 0;

    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto & layer = model.layers[i];
        if (!(layer.ffn_gate_exps || layer.ffn_down_exps || layer.ffn_up_exps || layer.ffn_gate_up_exps ||
              layer.ffn_gate_exps_b || layer.ffn_down_exps_b || layer.ffn_up_exps_b || layer.ffn_gate_up_exps_b ||
              layer.ffn_gate_exps_s || layer.ffn_down_exps_s || layer.ffn_up_exps_s)) {
            continue;
        }

        const char * layout = llama_moe_expert_layout_name(layer);
        const int32_t n_experts = llama_moe_expert_count_from_layer(layer);

        LLAMA_LOG_INFO("%s: layer %zu: MoE tensors found\n", __func__, i);
        LLAMA_LOG_INFO("%s: layer %zu: expert tensor layout = %s\n", __func__, i, layout);
        if (n_experts > 0) {
            LLAMA_LOG_INFO("%s: layer %zu: n_experts = %d\n", __func__, i, n_experts);
            moe_layers.push_back({(int32_t) i, n_experts});
            max_layer_experts = std::max(max_layer_experts, n_experts);
        } else {
            LLAMA_LOG_INFO("%s: layer %zu: n_experts = unknown\n", __func__, i);
        }
    }

    if (!cache.enabled()) {
        return;
    }

    // collection mode (Pass 1 of the frequency workflow: no whitelist, slots <
    // experts): only record access at eval. use_moe_gpu_slot_cache is false in
    // this mode, so the banks would never be used - skip the preload and avoid
    // allocating ~n_slots * n_layers worth of VRAM at load time
    if (cache.frequency_whitelist.empty() && cache.size() < max_layer_experts) {
        LLAMA_LOG_INFO("%s: collection mode (slots=%d < experts=%d, no whitelist): skipping GPU expert bank preload\n",
                __func__, cache.size(), max_layer_experts);
        return;
    }

    if (!cache.frequency_whitelist.empty()) {
        LLAMA_LOG_INFO("%s: frequency-based placement mode: %zu experts in whitelist\n",
                __func__, cache.frequency_whitelist.size());
    }

    int32_t n_preloaded = 0;
    for (const llama_moe_layer_preload_info & info : moe_layers) {
        if (cache.size() >= info.n_experts) {
            LLAMA_LOG_INFO("%s: layer=%d has %d experts and %d GPU expert slots; materializing a packed GPU expert bank\n",
                    __func__, info.layer_id, info.n_experts, cache.size());
        }

        const int32_t layer_id = info.layer_id;

        if (!cache.frequency_whitelist.empty()) {
            // frequency-based placement: preload only whitelisted experts
            for (const auto& [wl_layer, wl_expert] : cache.frequency_whitelist) {
                if (wl_layer != layer_id || wl_expert >= info.n_experts) {
                    continue;
                }

                int32_t slot = cache.find(layer_id, wl_expert);
                if (slot >= 0) {
                    slot = cache.preload_or_assign_slot(layer_id, wl_expert, step++);
                    LLAMA_LOG_INFO("%s: MoE GPU expert slot preload reuse: layer=%d expert=%d slot=%d\n", __func__, layer_id, wl_expert, slot);
                    if (!llama_moe_gpu_expert_slot_materialize(model_mut, slot, layer_id, wl_expert, info.n_experts)) {
                        cache.release_slot(layer_id, slot);
                    }
                    ++n_preloaded;
                    continue;
                }

                slot = cache.find_free(layer_id);
                if (slot < 0) {
                    slot = cache.find_lru_victim(layer_id);
                    const auto * victim = cache.slot_at(layer_id, slot);
                    if (victim != nullptr && victim->resident) {
                        LLAMA_LOG_INFO("%s: MoE GPU expert slot preload replace: slot=%d old_layer=%d old_expert=%d new_layer=%d new_expert=%d\n",
                                __func__, slot, victim->layer_id, victim->expert_id, layer_id, wl_expert);
                    }
                }
                const int32_t assigned = cache.preload_or_assign_slot(layer_id, wl_expert, step++);
                LLAMA_LOG_INFO("%s: MoE GPU expert slot preload: layer=%d expert=%d slot=%d\n", __func__, layer_id, wl_expert, assigned >= 0 ? assigned : slot);
                if (!llama_moe_gpu_expert_slot_materialize(model_mut, assigned >= 0 ? assigned : slot, layer_id, wl_expert, info.n_experts)) {
                    cache.release_slot(layer_id, assigned >= 0 ? assigned : slot);
                }
                ++n_preloaded;
            }
        } else {
            // full-slot mode: preload all experts sequentially
            for (int32_t expert_id = 0; expert_id < max_layer_experts && expert_id < cache.size(); ++expert_id) {
                if (expert_id >= info.n_experts) {
                    continue;
                }

                int32_t slot = cache.find(layer_id, expert_id);
                if (slot >= 0) {
                    slot = cache.preload_or_assign_slot(layer_id, expert_id, step++);
                    LLAMA_LOG_INFO("%s: MoE GPU expert slot preload reuse: layer=%d expert=%d slot=%d\n", __func__, layer_id, expert_id, slot);
                    if (!llama_moe_gpu_expert_slot_materialize(model_mut, slot, layer_id, expert_id, info.n_experts)) {
                        cache.release_slot(layer_id, slot);
                    }
                    ++n_preloaded;
                    continue;
                }

                slot = cache.find_free(layer_id);
                if (slot < 0) {
                    slot = cache.find_lru_victim(layer_id);
                    const auto * victim = cache.slot_at(layer_id, slot);
                    if (victim != nullptr && victim->resident) {
                        LLAMA_LOG_INFO("%s: MoE GPU expert slot preload replace: slot=%d old_layer=%d old_expert=%d new_layer=%d new_expert=%d\n",
                                __func__, slot, victim->layer_id, victim->expert_id, layer_id, expert_id);
                    }
                }
                const int32_t assigned = cache.preload_or_assign_slot(layer_id, expert_id, step++);
                LLAMA_LOG_INFO("%s: MoE GPU expert slot preload: layer=%d expert=%d slot=%d\n", __func__, layer_id, expert_id, assigned >= 0 ? assigned : slot);
                if (!llama_moe_gpu_expert_slot_materialize(model_mut, assigned >= 0 ? assigned : slot, layer_id, expert_id, info.n_experts)) {
                    cache.release_slot(layer_id, assigned >= 0 ? assigned : slot);
                }
                ++n_preloaded;
            }
        }
    }

    LLAMA_LOG_INFO("%s: MoE GPU expert slot preload complete: %d experts preloaded across %zu MoE layers\n", __func__, n_preloaded, moe_layers.size());
}

// Build a fingerprint that identifies a specific model for MoE frequency
// reports. Derived from GGUF metadata (general.name), the model description,
// the parameter count and the total model size: stable across runs of the same
// file, and different between different models or quantizations. Sanitized so
// the value is safe to embed in the JSON report.
static std::string llama_moe_model_fingerprint(const llama_model & model) {
    auto sanitize = [](const char * s) -> std::string {
        std::string out;
        if (s == nullptr) {
            return out;
        }
        for (const char * p = s; *p; ++p) {
            const unsigned char c = (unsigned char) *p;
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
                out += (char) c;
            } else {
                out += '_';
            }
        }
        return out;
    };
    char buf[512];
    std::string name;
    if (llama_model_meta_val_str(&model, "general.name", buf, sizeof(buf)) > 0) {
        name = sanitize(buf);
    }
    std::string desc;
    if (llama_model_desc(&model, buf, sizeof(buf)) > 0) {
        desc = sanitize(buf);
    }
    char fp[1024];
    snprintf(fp, sizeof(fp), "%s|%s|%llu|%llu",
             name.c_str(), desc.c_str(),
             (unsigned long long) llama_model_n_params(&model),
             (unsigned long long) llama_model_size(&model));
    return std::string(fp);
}

static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.load_mode,
            params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        {
            int32_t requested_slots = params.n_moe_gpu_expert_slot_num;
            // frequency mode auto-activates when slot is default -1
            if (requested_slots < 0) {
                const bool is_freq_mode = params.moe_expert_placement &&
                                          std::string(params.moe_expert_placement) == "frequency";
                if (is_freq_mode) {
                    requested_slots = INT32_MAX;
                }
            }
            if (params.moe_gpu_expert_slot_auto) {
                // elastic sizing: expert weights were already routed off-GPU at
                // create_tensor time; the concrete slot count needs free VRAM
                // AFTER weights + KV are resident, so resolve it in llama_context
                if (params.no_alloc || !llama_model_has_moe_experts(*model)) {
                    LLAMA_LOG_INFO("%s: MoE GPU expert slot auto sizing skipped (no_alloc=%d, has_moe=%d)\n",
                            __func__, (int) params.no_alloc, (int) llama_model_has_moe_experts(*model));
                    model->moe_gpu_expert_cache.clear();
                } else {
                    model->moe_gpu_expert_cache.auto_pending = true;
                    LLAMA_LOG_INFO("%s: MoE GPU expert slot auto sizing requested - deferring until after KV allocation\n", __func__);
                }
            } else if (requested_slots < 0) {
                model->moe_gpu_expert_cache.clear();
            } else if (params.no_alloc) {
                LLAMA_LOG_INFO("%s: MoE GPU expert slot mode requested but no_alloc=true - skipping slot preload (tensor data not loaded)\n", __func__);
                model->moe_gpu_expert_cache.clear();
            } else if (!llama_model_has_moe_experts(*model)) {
                LLAMA_LOG_INFO("%s: MoE GPU expert slot mode requested, but model has no MoE experts; ignoring\n", __func__);
                model->moe_gpu_expert_cache.clear();
            } else {
                const int32_t effective_slots = llama_moe_gpu_expert_slot_effective_count(*model, requested_slots);
                if (effective_slots <= 0) {
                    LLAMA_LOG_INFO("%s: MoE GPU expert slot mode disabled (effective slot count = %d)\n", __func__, effective_slots);
                    model->moe_gpu_expert_cache.clear();
                } else {
                    model->moe_gpu_expert_cache.init(effective_slots);
                    model->moe_gpu_expert_cache.materialize_cb       = llama_moe_gpu_expert_slot_materialize_cb;
                    model->moe_gpu_expert_cache.materialize_userdata = (llama_model *) model;
                    if (const char * pf = getenv("LLAMA_MOE_PREFETCH_MS")) {
                        model->moe_gpu_expert_cache.prefetch_budget_ms = atof(pf);
                    }
                    llama_moe_gpu_expert_slot_prefill_configure(model->moe_gpu_expert_cache);
                    LLAMA_LOG_INFO("%s: initialized MoE GPU expert slot cache with %d slots (requested %d)\n", __func__, effective_slots, requested_slots);

                    // frequency-based placement: set whitelist from freq report
                    const bool is_freq_mode = params.moe_expert_placement &&
                                              std::string(params.moe_expert_placement) == "frequency";
                    if (is_freq_mode && params.moe_freq_report_in) {
                        llama_moe_freq_report freq_report = load_freq_report(params.moe_freq_report_in);
                        if (!freq_report.stats.empty()) {
                            // verify the report was generated for this model before applying it
                            const std::string model_fingerprint = llama_moe_model_fingerprint(*model);
                            const bool fingerprint_ok = !freq_report.model_fingerprint.empty() &&
                                                        freq_report.model_fingerprint == model_fingerprint;
                            if (!fingerprint_ok) {
                                LLAMA_LOG_WARN("%s: frequency report fingerprint mismatch (report: '%s', model: '%s'); ignoring report and falling back to full-slot mode\n",
                                        __func__, freq_report.model_fingerprint.c_str(), model_fingerprint.c_str());
                            } else {
                            const int32_t total_experts = freq_report.n_layers * freq_report.n_experts;
                            const int32_t gpu_count = std::max(1, static_cast<int32_t>(
                                total_experts * params.moe_gpu_expert_ratio));
                            // Build whitelist from sorted frequency list
                            std::vector<std::pair<int32_t, int32_t>> whitelist;
                            for (int32_t i = 0; i < gpu_count && i < (int32_t)freq_report.sorted_by_frequency.size(); i++) {
                                int32_t idx = freq_report.sorted_by_frequency[i];
                                whitelist.push_back({freq_report.stats[idx].layer_id,
                                                     freq_report.stats[idx].expert_id});
                            }
                            model->moe_gpu_expert_cache.set_frequency_whitelist(whitelist);
                            LLAMA_LOG_INFO("%s: frequency placement: %d/%d experts on GPU (ratio=%.2f)\n",
                                    __func__, (int32_t)whitelist.size(), total_experts, params.moe_gpu_expert_ratio);
                            }
                        } else {
                            LLAMA_LOG_WARN("%s: frequency report not found or empty at %s, falling back to full-slot mode\n",
                                    __func__, params.moe_freq_report_in);
                        }
                    }

                    // enable access tracking when an output freq report path is specified (Pass 1)
                    if (params.moe_freq_report_out && params.moe_freq_report_out[0]) {
                        model->moe_gpu_expert_cache.track_access = true;
                    }

                    llama_moe_gpu_expert_slot_preload(*model);
                }
            }
        }

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}

struct llama_model * llama_model_init_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params) {
    GGML_ASSERT(metadata != nullptr);
    std::string path_model;
    std::vector<std::string> splits = {};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, /*file*/ nullptr, params);
}
// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, splits.front(), splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_file_ptr(FILE * file, struct llama_model_params params) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }
    std::string path_model;
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, file, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

bool llama_save_moe_freq_report(llama_context * ctx, const char * path) {
    if (!ctx || !path || !path[0]) {
        fprintf(stderr, "[moe-stats] save called with null ctx or path\n");
        return false;
    }
    const auto & model = ctx->get_model();
    fprintf(stderr, "[moe-stats] track_access=%d, access_counts=%zu\n",
        model.moe_gpu_expert_cache.track_access,
        model.moe_gpu_expert_cache.access_counts.size());
    if (!model.moe_gpu_expert_cache.track_access) return false;
    auto report = model.moe_gpu_expert_cache.generate_access_report(llama_moe_model_fingerprint(model), 0);
    fprintf(stderr, "[moe-stats] report stats=%zu, sorted=%zu\n",
        report.stats.size(), report.sorted_by_frequency.size());
    bool ok = save_freq_report(report, path);
    fprintf(stderr, "[moe-stats] save_freq_report returned %d\n", ok);
    return ok;
}

