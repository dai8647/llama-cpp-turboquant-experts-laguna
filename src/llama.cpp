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
#include "llama-moe-placement.h"

#include <string>

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
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

static bool llama_moe_gpu_expert_slot_materialize(
        llama_model & model,
        int32_t slot_id,
        int32_t layer_id,
        int32_t expert_id,
        int32_t n_experts) {
    if (slot_id < 0 || slot_id >= model.moe_gpu_expert_cache.size()) {
        return false;
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
        for (const auto & bank_tensor : bank.tensors) {
            if (llama_moe_gpu_expert_bank_copy_tensor(bank_tensor, expert_id, slot_id)) {
                copied_bytes += bank_tensor.nbytes_per_expert;
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
                    llama_moe_gpu_expert_slot_materialize(model_mut, slot, layer_id, wl_expert, info.n_experts);
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
                llama_moe_gpu_expert_slot_materialize(model_mut, assigned >= 0 ? assigned : slot, layer_id, wl_expert, info.n_experts);
                ++n_preloaded;
            }

            // When the bank can hold the full layer, also materialize non-whitelisted
            // experts so the packed bank is complete and remap hits on all selections.
            if (cache.size() >= info.n_experts) {
                for (int32_t expert_id = 0; expert_id < info.n_experts; ++expert_id) {
                    if (cache.is_in_frequency_whitelist(layer_id, expert_id)) {
                        continue;
                    }
                    int32_t assigned = cache.preload_or_assign_slot(layer_id, expert_id, step++);
                    if (assigned >= 0) {
                        llama_moe_gpu_expert_slot_materialize(model_mut, assigned, layer_id, expert_id, info.n_experts);
                        ++n_preloaded;
                    }
                }
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
                    llama_moe_gpu_expert_slot_materialize(model_mut, slot, layer_id, expert_id, info.n_experts);
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
                llama_moe_gpu_expert_slot_materialize(model_mut, assigned >= 0 ? assigned : slot, layer_id, expert_id, info.n_experts);
                ++n_preloaded;
            }
        }
    }

    LLAMA_LOG_INFO("%s: MoE GPU expert slot preload complete: %d experts preloaded across %zu MoE layers\n", __func__, n_preloaded, moe_layers.size());
}

static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.use_mmap, params.use_direct_io,
            params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

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
            // enable access tracking when out path is specified (independent of slot cache)
            const char * freq_report_out = params.moe_freq_report_out_path && params.moe_freq_report_out_path[0]
                ? params.moe_freq_report_out_path
                : (params.moe_freq_report_path && params.moe_freq_report_path[0]
                    ? params.moe_freq_report_path : nullptr);
            if (freq_report_out) {
                model->moe_gpu_expert_cache.track_access = true;
            }
            model->moe_gpu_expert_cache.tracked_n_experts = (int32_t)model->hparams.n_expert;

            if (requested_slots < 0) {
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
                    LLAMA_LOG_INFO("%s: initialized MoE GPU expert slot cache with %d slots (requested %d), tracked_n_experts=%d\n", __func__, effective_slots, requested_slots, model->hparams.n_expert);

                    // frequency-based placement: set whitelist from freq report
                    const bool is_freq_mode = params.moe_expert_placement &&
                                              std::string(params.moe_expert_placement) == "frequency";
                    if (is_freq_mode) {
                        model->moe_gpu_expert_cache.frequency_mode = true;
                    }
                    const char * freq_report_in = params.moe_freq_report_in_path && params.moe_freq_report_in_path[0]
                        ? params.moe_freq_report_in_path
                        : (params.moe_freq_report_path && params.moe_freq_report_path[0]
                            ? params.moe_freq_report_path : nullptr);
                    if (is_freq_mode && freq_report_in) {
                        llama_moe_freq_report freq_report = load_freq_report(freq_report_in);
                        if (!freq_report.stats.empty()) {
                            const int32_t total_experts = freq_report.n_layers * freq_report.n_experts;
                            const int32_t gpu_count = std::max(1, static_cast<int32_t>(
                                total_experts * params.moe_gpu_expert_ratio));
                            // Build whitelist: layer-wise top experts (B-optimization)
                            std::vector<std::pair<int32_t, int32_t>> whitelist;
                            const int32_t per_layer_budget = std::max(1, static_cast<int32_t>(
                                std::round(static_cast<double>(freq_report.n_experts) * params.moe_gpu_expert_ratio)));
                            for (int32_t l = 0; l < freq_report.n_layers; l++) {
                                std::vector<int32_t> layer_expert_ids(freq_report.n_experts);
                                std::iota(layer_expert_ids.begin(), layer_expert_ids.end(), 0);
                                std::sort(layer_expert_ids.begin(), layer_expert_ids.end(),
                                          [&](int32_t a, int32_t b) {
                                              int32_t idx_a = l * freq_report.n_experts + a;
                                              int32_t idx_b = l * freq_report.n_experts + b;
                                              return freq_report.stats[idx_a].total_selections > freq_report.stats[idx_b].total_selections;
                                          });
                                int32_t take = std::min(per_layer_budget, (int32_t)freq_report.n_experts);
                                for (int32_t i = 0; i < take; i++) {
                                    int32_t expert_idx = layer_expert_ids[i];
                                    int32_t idx = l * freq_report.n_experts + expert_idx;
                                    whitelist.push_back({freq_report.stats[idx].layer_id, freq_report.stats[idx].expert_id});
                                }
                            }
                            model->moe_gpu_expert_cache.set_frequency_whitelist(whitelist);
                            LLAMA_LOG_INFO("%s: frequency placement (layer-wise): %d/%d experts on GPU (per-layer budget=%d, ratio=%.2f)\n",
                                    __func__, (int32_t)whitelist.size(), total_experts, per_layer_budget, params.moe_gpu_expert_ratio);
                        } else {
                            LLAMA_LOG_WARN("%s: frequency report not found or empty at %s, falling back to full-slot mode\n",
                                    __func__, freq_report_in);
                        }
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
    params.use_mmap = false;
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
        return false;
    }
    const auto & model = ctx->get_model();
    if (!model.moe_gpu_expert_cache.track_access) return false;
    auto report = model.moe_gpu_expert_cache.generate_access_report("moe-access-tracking", 0);
    if (report.n_experts <= 0) {
        return false;
    }
    return save_freq_report(report, path);
}

