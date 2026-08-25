#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-vocab.h"
#include "llama-moe-stats.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;

// available models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_14M,
    LLM_TYPE_17M,
    LLM_TYPE_22M,
    LLM_TYPE_33M,
    LLM_TYPE_47M,
    LLM_TYPE_60M,
    LLM_TYPE_70M,
    LLM_TYPE_80M,
    LLM_TYPE_109M,
    LLM_TYPE_137M,
    LLM_TYPE_140M,
    LLM_TYPE_149M,
    LLM_TYPE_160M,
    LLM_TYPE_190M,
    LLM_TYPE_220M,
    LLM_TYPE_230M,
    LLM_TYPE_250M,
    LLM_TYPE_256M,
    LLM_TYPE_270M,
    LLM_TYPE_335M,
    LLM_TYPE_350M,
    LLM_TYPE_360M,
    LLM_TYPE_395M,
    LLM_TYPE_410M,
    LLM_TYPE_450M,
    LLM_TYPE_475M,
    LLM_TYPE_558M,
    LLM_TYPE_700M,
    LLM_TYPE_770M,
    LLM_TYPE_780M,
    LLM_TYPE_950M,
    LLM_TYPE_0_3B,
    LLM_TYPE_0_5B,
    LLM_TYPE_0_6B,
    LLM_TYPE_0_8B,
    LLM_TYPE_1B,
    LLM_TYPE_1_2B,
    LLM_TYPE_1_3B,
    LLM_TYPE_1_4B,
    LLM_TYPE_1_5B,
    LLM_TYPE_1_6B,
    LLM_TYPE_1_7B,
    LLM_TYPE_1_8B,
    LLM_TYPE_2B,
    LLM_TYPE_2_6B,
    LLM_TYPE_2_8B,
    LLM_TYPE_2_9B,
    LLM_TYPE_3B,
    LLM_TYPE_4B,
    LLM_TYPE_6B,
    LLM_TYPE_6_9B,
    LLM_TYPE_7B,
    LLM_TYPE_8B,
    LLM_TYPE_9B,
    LLM_TYPE_11B,
    LLM_TYPE_12B,
    LLM_TYPE_13B,
    LLM_TYPE_14B,
    LLM_TYPE_15B,
    LLM_TYPE_16B,
    LLM_TYPE_20B,
    LLM_TYPE_26B,
    LLM_TYPE_27B,
    LLM_TYPE_30B,
    LLM_TYPE_31B,
    LLM_TYPE_32B,
    LLM_TYPE_34B,
    LLM_TYPE_35B,
    LLM_TYPE_36B,
    LLM_TYPE_40B,
    LLM_TYPE_65B,
    LLM_TYPE_70B,
    LLM_TYPE_120B,
    LLM_TYPE_142B,
    LLM_TYPE_236B,
    LLM_TYPE_290B,
    LLM_TYPE_314B,
    LLM_TYPE_405B,
    LLM_TYPE_671B,
    LLM_TYPE_SMALL,
    LLM_TYPE_MEDIUM,
    LLM_TYPE_LARGE,
    LLM_TYPE_XL,
    LLM_TYPE_A1_7B,
    LLM_TYPE_A2_7B,
    LLM_TYPE_8x7B,
    LLM_TYPE_8x22B,
    LLM_TYPE_16x12B,
    LLM_TYPE_16x3_8B,
    LLM_TYPE_10B_128x3_66B,
    LLM_TYPE_57B_A14B,
    LLM_TYPE_17B_16E, // llama4 Scout
    LLM_TYPE_17B_128E, // llama4 Maverick
    LLM_TYPE_A13B,
    LLM_TYPE_7B_A1B,
    LLM_TYPE_8B_A1B, // lfm2moe
    LLM_TYPE_12B_A2_5B,
    LLM_TYPE_16B_A1B,
    LLM_TYPE_21B_A3B, // Ernie MoE small
    LLM_TYPE_24B_A2B, // lfm2moe
    LLM_TYPE_26B_A4B, // Gemma4
    LLM_TYPE_30B_A3B,
    LLM_TYPE_31B_A3_5B,
    LLM_TYPE_35B_A3B, // Qwen3.5
    LLM_TYPE_48B_A3B, // Kimi Linear
    LLM_TYPE_80B_A3B, // Qwen3 Next
    LLM_TYPE_100B_A6B,
    LLM_TYPE_102B_A12B, // Solar-Open
    LLM_TYPE_106B_A12B, // GLM-4.5-Air
    LLM_TYPE_118B_A8B,  // Laguna-S-2
    LLM_TYPE_120B_A12B, // Nemotron 3 Super
    LLM_TYPE_122B_A10B, // Qwen3.5
    LLM_TYPE_196B_A11B, // Step3.5-Flash
    LLM_TYPE_230B_A10B, // Minimax M2
    LLM_TYPE_428B_A23B, // Minimax M3
    LLM_TYPE_235B_A22B,
    LLM_TYPE_300B_A47B, // Ernie MoE big
    LLM_TYPE_310B_A15B, // /MiMo-V2-Flash
    LLM_TYPE_355B_A32B, // GLM-4.5
    LLM_TYPE_397B_A17B, // Qwen3.5
    LLM_TYPE_685B_A37B, // DeepSeek V3.2
    LLM_TYPE_744B_A40B, // GLM-5
    LLM_TYPE_E2B,
    LLM_TYPE_E4B,
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);

// Map a GGUF activation-name string to llm_ffn_op_type. Returns `fallback` if
// the string is empty or not recognized.
llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback);

struct llama_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer_shortconv {
    struct ggml_tensor * in_proj  = nullptr;
    struct ggml_tensor * conv     = nullptr;
    struct ggml_tensor * out_proj = nullptr;
};

struct llama_layer_nextn {
    struct ggml_tensor * eh_proj               = nullptr;
    struct ggml_tensor * eh_proj_s             = nullptr;
    struct ggml_tensor * eh_proj_in_s          = nullptr;
    struct ggml_tensor * embed_tokens          = nullptr;
    struct ggml_tensor * enorm                 = nullptr;
    struct ggml_tensor * hnorm                 = nullptr;
    struct ggml_tensor * shared_head_head      = nullptr;
    struct ggml_tensor * shared_head_head_s    = nullptr;
    struct ggml_tensor * shared_head_head_in_s = nullptr;
    struct ggml_tensor * shared_head_norm      = nullptr;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_norm_b     = nullptr;
    struct ggml_tensor * attn_norm_2     = nullptr;
    struct ggml_tensor * attn_norm_2_b   = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_q_norm_b   = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_k_norm_b   = nullptr;
    struct ggml_tensor * attn_out_norm   = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm   = nullptr;
    struct ggml_tensor * attn_kv_a_norm  = nullptr;
    struct ggml_tensor * attn_sub_norm   = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_sub_norm    = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc   = nullptr;
    struct ggml_tensor * ssm_norm        = nullptr;
    struct ggml_tensor * ssm_dt_norm     = nullptr;
    struct ggml_tensor * ssm_b_norm      = nullptr;
    struct ggml_tensor * ssm_c_norm      = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wq_a      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkv_b     = nullptr;
    struct ggml_tensor * wkv       = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wqkv_b    = nullptr;
    struct ggml_tensor * wo_a      = nullptr;
    struct ggml_tensor * wo_b      = nullptr;
    struct ggml_tensor * wq_cross  = nullptr;
    struct ggml_tensor * wk_cross  = nullptr;
    struct ggml_tensor * wv_cross  = nullptr;
    struct ggml_tensor * wo_cross  = nullptr;
    struct ggml_tensor * wq_enc    = nullptr;
    struct ggml_tensor * wk_enc    = nullptr;
    struct ggml_tensor * wv_enc    = nullptr;
    struct ggml_tensor * wo_enc    = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;

    // relative position bias
    struct ggml_tensor * attn_rel_b       = nullptr;
    struct ggml_tensor * attn_rel_b_enc   = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_norm_b       = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;
    struct ggml_tensor * ffn_post_norm_1  = nullptr; // gemma4
    struct ggml_tensor * ffn_post_norm_2  = nullptr; // gemma4
    struct ggml_tensor * ffn_pre_norm_2   = nullptr; // gemma4
    struct ggml_tensor * layer_out_norm   = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps    = nullptr;
    struct ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp      = nullptr;
    struct ggml_tensor * ffn_gate_inp_s    = nullptr; // gemma4
    struct ggml_tensor * ffn_gate_exps     = nullptr;
    struct ggml_tensor * ffn_down_exps     = nullptr;
    struct ggml_tensor * ffn_up_exps       = nullptr;
    struct ggml_tensor * ffn_gate_up_exps  = nullptr;
    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
    struct ggml_tensor * ffn_gate_exps_b   = nullptr;
    struct ggml_tensor * ffn_down_exps_b   = nullptr;
    struct ggml_tensor * ffn_up_exps_b     = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_b = nullptr;

    // ff MoE per-expert scales (NVFP4 per-tensor scale2)
    struct ggml_tensor * ffn_gate_exps_s   = nullptr;
    struct ggml_tensor * ffn_down_exps_s   = nullptr;
    struct ggml_tensor * ffn_up_exps_s     = nullptr;

    // ff MoE latent proj
    struct ggml_tensor * ffn_latent_down = nullptr;
    struct ggml_tensor * ffn_latent_up   = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    // ff adjugate experts (chexps)
    struct ggml_tensor * ffn_gate_chexps     = nullptr;
    struct ggml_tensor * ffn_down_chexps     = nullptr;
    struct ggml_tensor * ffn_up_chexps       = nullptr;

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3
    struct ggml_tensor * ffn_act    = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;
    struct ggml_tensor * ffn_gate_tid2eid = nullptr;

    // mamba proj
    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    // qwen3next
    struct ggml_tensor * ssm_beta_alpha = nullptr;

    // qwen3.5
    struct ggml_tensor * ssm_alpha = nullptr;

    // rwkv
    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    // rwkv7
    struct ggml_tensor * time_mix_w0         = nullptr;
    struct ggml_tensor * time_mix_a0         = nullptr;
    struct ggml_tensor * time_mix_a1         = nullptr;
    struct ggml_tensor * time_mix_a2         = nullptr;
    struct ggml_tensor * time_mix_v0         = nullptr;
    struct ggml_tensor * time_mix_v1         = nullptr;
    struct ggml_tensor * time_mix_v2         = nullptr;
    struct ggml_tensor * time_mix_g1         = nullptr;
    struct ggml_tensor * time_mix_g2         = nullptr;
    struct ggml_tensor * time_mix_k_k        = nullptr;
    struct ggml_tensor * time_mix_k_a        = nullptr;
    struct ggml_tensor * time_mix_r_k        = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct ggml_tensor * wq_s       = nullptr;
    struct ggml_tensor * wk_s       = nullptr;
    struct ggml_tensor * wv_s       = nullptr;
    struct ggml_tensor * wo_s       = nullptr;
    struct ggml_tensor * wqkv_s     = nullptr;
    struct ggml_tensor * wqkv_gate_s = nullptr;
    struct ggml_tensor * ffn_gate_s = nullptr;
    struct ggml_tensor * ffn_up_s   = nullptr;
    struct ggml_tensor * ffn_down_s = nullptr;
    struct ggml_tensor * ffn_gate_shexp_s = nullptr;
    struct ggml_tensor * ffn_up_shexp_s   = nullptr;
    struct ggml_tensor * ffn_down_shexp_s = nullptr;
    struct ggml_tensor * ssm_in_s    = nullptr;
    struct ggml_tensor * ssm_out_s   = nullptr;
    struct ggml_tensor * ssm_alpha_s = nullptr;
    struct ggml_tensor * ssm_beta_s  = nullptr;

    // input scales
    struct ggml_tensor * wq_in_s            = nullptr;
    struct ggml_tensor * wk_in_s            = nullptr;
    struct ggml_tensor * wv_in_s            = nullptr;
    struct ggml_tensor * wo_in_s            = nullptr;
    struct ggml_tensor * wqkv_in_s          = nullptr;
    struct ggml_tensor * wqkv_gate_in_s     = nullptr;
    struct ggml_tensor * ffn_gate_in_s      = nullptr;
    struct ggml_tensor * ffn_up_in_s        = nullptr;
    struct ggml_tensor * ffn_down_in_s      = nullptr;
    struct ggml_tensor * ffn_gate_exps_in_s = nullptr;
    struct ggml_tensor * ffn_down_exps_in_s = nullptr;
    struct ggml_tensor * ffn_up_exps_in_s   = nullptr;
    struct ggml_tensor * ffn_gate_shexp_in_s= nullptr;
    struct ggml_tensor * ffn_up_shexp_in_s  = nullptr;
    struct ggml_tensor * ffn_down_shexp_in_s= nullptr;
    struct ggml_tensor * ssm_in_in_s        = nullptr;
    struct ggml_tensor * ssm_out_in_s       = nullptr;
    struct ggml_tensor * ssm_alpha_in_s     = nullptr;
    struct ggml_tensor * ssm_beta_in_s      = nullptr;

    // altup & laurel
    struct ggml_tensor * per_layer_inp_gate   = nullptr;
    struct ggml_tensor * per_layer_proj       = nullptr;
    struct ggml_tensor * per_layer_post_norm  = nullptr;
    struct ggml_tensor * altup_correct_coef   = nullptr;
    struct ggml_tensor * altup_correct_scale  = nullptr;
    struct ggml_tensor * altup_predict_coef   = nullptr;
    struct ggml_tensor * altup_router         = nullptr;
    struct ggml_tensor * altup_router_norm    = nullptr;
    struct ggml_tensor * laurel_l             = nullptr;
    struct ggml_tensor * laurel_r             = nullptr;
    struct ggml_tensor * laurel_post_norm     = nullptr;

    // openai-moe
    struct ggml_tensor * attn_sinks = nullptr;

    // DeepSeek-V4
    struct ggml_tensor * attn_kv_norm = nullptr;
    struct ggml_tensor * hc_attn_fn   = nullptr;
    struct ggml_tensor * hc_attn_base = nullptr;
    struct ggml_tensor * hc_attn_scale = nullptr;
    struct ggml_tensor * hc_ffn_fn    = nullptr;
    struct ggml_tensor * hc_ffn_base  = nullptr;
    struct ggml_tensor * hc_ffn_scale = nullptr;
    struct ggml_tensor * attn_comp_wkv   = nullptr;
    struct ggml_tensor * attn_comp_wgate = nullptr;
    struct ggml_tensor * attn_comp_ape   = nullptr;
    struct ggml_tensor * attn_comp_norm  = nullptr;
    struct ggml_tensor * indexer_comp_wkv   = nullptr;
    struct ggml_tensor * indexer_comp_wgate = nullptr;
    struct ggml_tensor * indexer_comp_ape   = nullptr;
    struct ggml_tensor * indexer_comp_norm  = nullptr;

    // cogvlm
    struct ggml_tensor * visexp_attn_wqkv = nullptr;
    struct ggml_tensor * visexp_attn_wo   = nullptr;
    struct ggml_tensor * visexp_ffn_gate  = nullptr;
    struct ggml_tensor * visexp_ffn_down  = nullptr;
    struct ggml_tensor * visexp_ffn_up    = nullptr;

    // xIELU activation parameters for Apertus
    struct ggml_tensor * ffn_act_alpha_n = nullptr;
    struct ggml_tensor * ffn_act_alpha_p = nullptr;
    struct ggml_tensor * ffn_act_beta    = nullptr;
    struct ggml_tensor * ffn_act_eps     = nullptr;

    // Kimi Linear KDA (using ssm_ prefix for consistency)
    // Note: ssm_dt_b already exists above (mamba bias), reused for Kimi dt_bias
    struct ggml_tensor * ssm_q_conv = nullptr;
    struct ggml_tensor * ssm_k_conv = nullptr;
    struct ggml_tensor * ssm_v_conv = nullptr;
    struct ggml_tensor * ssm_f_a    = nullptr;
    struct ggml_tensor * ssm_f_b    = nullptr;
    struct ggml_tensor * ssm_beta   = nullptr;
    struct ggml_tensor * ssm_g_a    = nullptr;
    struct ggml_tensor * ssm_g_b    = nullptr;
    struct ggml_tensor * ssm_o_norm = nullptr;

    // full-rank KDA forget/output gates (bailing-hybrid, no_kda_lora=true):
    // single matmuls that replace the ssm_{f,g}_{a,b} low-rank pairs above
    struct ggml_tensor * ssm_f      = nullptr;
    struct ggml_tensor * ssm_g      = nullptr;

    // DSA (deepseek sparse attention)
    struct ggml_tensor * indexer_k_norm   = nullptr;
    struct ggml_tensor * indexer_k_norm_b = nullptr;
    struct ggml_tensor * indexer_proj     = nullptr;
    struct ggml_tensor * indexer_attn_k   = nullptr;
    struct ggml_tensor * indexer_attn_q_b = nullptr; // note: for lora a/b, not bias

    // MSA
    struct ggml_tensor * index_q_proj = nullptr;
    struct ggml_tensor * index_k_proj = nullptr;
    struct ggml_tensor * index_q_norm = nullptr;
    struct ggml_tensor * index_k_norm = nullptr;

    // gemma4 layer output scale, reused for talkie embedding skip scale
    struct ggml_tensor * out_scale = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;

    struct llama_layer_shortconv shortconv;

    struct llama_layer_nextn nextn;
};

struct llama_device {
    bool is_meta;

    ggml_backend_dev_t dev;
};

struct llama_meta_device_get_split_state_userdata {
    size_t                     n_devices;
    const struct llama_model * model;
};

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata);

struct llama_moe_gpu_expert_slot_tensor {
    std::string name;
    struct ggml_tensor * src = nullptr;
    struct ggml_tensor * dev = nullptr;
    size_t nbytes = 0;
};

struct llama_moe_gpu_expert_slot {
    int32_t layer_id  = -1;
    int32_t expert_id = -1;
    int64_t last_used = 0;
    bool resident     = false;
    bool bank_backed  = false;

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    std::vector<llama_moe_gpu_expert_slot_tensor> tensors;

    void clear_storage() {
        tensors.clear();
        buf.reset();
        ctx.reset();
        bank_backed = false;
    }
};

struct llama_moe_gpu_expert_bank_tensor {
    std::string name;
    struct ggml_tensor * src = nullptr;
    struct ggml_tensor * dev = nullptr;
    int32_t expert_dim = -1;
    size_t nbytes_per_expert = 0;
};

struct llama_moe_gpu_expert_bank {
    int32_t layer_id  = -1;
    int32_t n_experts = 0;
    int32_t n_slots   = 0;

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    std::vector<llama_moe_gpu_expert_bank_tensor> tensors;

    void clear_storage() {
        tensors.clear();
        buf.reset();
        ctx.reset();
    }
};

// userdata for the eval-time expert-id -> slot-id remap op; one instance per
// MoE layer is pooled here (model lifetime) because built graphs are reused
// across steps and outlive the llm_graph_context that created the nodes.
// with the q* policy the same instance is shared by the two decode custom ops
// (planning op, then host-exec op); graph dependencies order them per step.
struct llm_moe_gpu_slot_remap_userdata {
    llama_moe_gpu_expert_cache * cache = nullptr;
    const struct llama_model * model = nullptr;
    int32_t layer_id = -1;
    int32_t n_experts = 0;

    // q* decode variant (filled at graph build)
    bool    qstar_active = false;
    int32_t n_used  = 0;   // selections per step (= n_expert_used)
    int32_t n_embd  = 0;

    // q* planning -> exec handoff, rewritten every step by the planning op
    std::vector<int32_t> qstar_deferred; // experts routed to host execution
};

// per-layer host-side expert execution engine for the q* policy: one tiny
// static ggml graph whose weight tensors alias the original host/mmap weight
// storage (zero copies; any expert id selectable at run time via mul_mat_id).
// all non-weight tensors live in one owned scratch block with data pointers
// assigned at build time.
struct llama_moe_qstar_layer_exec {
    bool ready     = false;
    bool supported = true; // false: layer falls back to transfer-always

    int32_t n_ff   = 0;
    int32_t n_embd = 0;
    size_t r_max   = 0;
    bool fused_gate_up = false;

    // aliased source weights and per-expert strides (bytes)
    const void * gate_data   = nullptr;
    const void * up_data     = nullptr;
    const void * down_data   = nullptr;
    const void * gateup_data = nullptr;
    size_t gate_stride = 0, up_stride = 0, down_stride = 0, gateup_stride = 0;
    enum ggml_type wtype = GGML_TYPE_F32;
    float scale_gate = 1.0f, scale_up = 1.0f, scale_down = 1.0f;

    ggml_context_ptr ctx;
    ggml_cgraph * gf = nullptr;          // owned by ctx
    ggml_tensor * t_ids = nullptr;       // [r,1] i32, data swapped per call
    ggml_tensor * t_ids_down = nullptr;  // [1,r] i32, same linear buffer
    ggml_tensor * t_x   = nullptr;       // [n_embd, 1, 1] f32 token vector
    ggml_tensor * t_y   = nullptr;       // [n_embd, r] f32 out
    std::vector<uint8_t> mem;            // activations/intermediates scratch
    std::vector<uint8_t> work;           // cplan work buffer

    void clear_storage() {
        ready = false;
        gf = nullptr;
        t_ids = t_ids_down = t_x = t_y = nullptr;
        mem.clear();
        work.clear();
        ctx.reset();
    }
};

struct llama_moe_gpu_expert_cache {
    using materialize_cb_t = bool (*)(void * userdata, int32_t slot_id, int32_t layer_id, int32_t expert_id, int32_t n_experts);

    int32_t n_slots = 0;

    std::unordered_map<int32_t, std::vector<llama_moe_gpu_expert_slot>> slots_by_layer;
    std::unordered_map<int32_t, llama_moe_gpu_expert_bank> banks_by_layer;
    std::unordered_map<uint64_t, int32_t> expert_to_slot;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> compute_tensor_by_src;

    materialize_cb_t materialize_cb = nullptr;
    void * materialize_userdata = nullptr;

    int64_t clock = 0;
    int64_t n_hit = 0;
    int64_t n_miss = 0;
    int64_t n_evict = 0;

    // global LRU pool (--moe-gpu-expert-global-lru / LLAMA_MOE_GLOBAL_LRU=1):
    // the n_slots budget is shared by all layers instead of each layer owning
    // its own full pool; hot layers take residency from cold ones. banks keep
    // per-layer capacity n_slots (lazy, stable pointers), only residency is
    // globally accounted. cache_mutex guarded.
    bool    global_lru_enabled = false;
    int64_t n_resident_global  = 0;
    int64_t n_evict_cross      = 0;

    // materialization telemetry (LLAMA_MOE_SLOT_STATS=1), cache_mutex guarded
    int64_t n_copy = 0;
    int64_t copy_bytes = 0;
    int64_t copy_ns = 0;

    // inter-step speculative prefetch (LLAMA_MOE_PREFETCH_MS), cache_mutex guarded
    double prefetch_budget_ms = 0.0;
    std::unordered_map<int32_t, std::vector<int32_t>> last_selections;

    // elastic VRAM sizing (--moe-gpu-expert-slot-num auto): set at model load,
    // resolved after KV allocation in the llama_context constructor
    bool auto_pending = false;

    // prefill double buffering (LLAMA_MOE_PREFILL_PF=1), cache_mutex guarded.
    // background H2D copies run on the backend's dedicated copy stream; the
    // target slot stays NON-resident until its event completes, so the eval
    // remap treats it as a miss and materialize() drains (waits for) the copy
    // before reusing the region - readers only ever see fully-written data.
    bool            prefill_pf_enabled      = false;
    int64_t         prefill_pf_budget_bytes = 256LL * 1024 * 1024; // per ubatch
    int64_t         prefill_pf_max_inflight = 64;                  // entries
    ggml_backend_t  prefill_pf_backend      = nullptr; // ext API backend

    struct prefill_pf_copy {
        int32_t layer_id  = -1;
        int32_t slot_id   = -1;
        int32_t expert_id = -1;
        void  * event     = nullptr; // copy-stream event (ext API)
    };
    std::vector<prefill_pf_copy> prefill_pf_inflight;

    // back-pointer to the owning model for the pooled remap userdata (the
    // graph context carries no model reference); set alongside init()
    const struct llama_model * owner_model = nullptr;

    // q* bandwidth-adaptive policy (--moe-qstar / LLAMA_MOE_QSTAR=1),
    // cache_mutex guarded
    bool    qstar_enabled   = false;
    int32_t qstar_threads   = 3;      // dedicated host GEMM workers
    double  qstar_budget_us = 300.0;  // per-layer per-step transfer budget
    bool    qstar_stats     = false;  // LLAMA_MOE_QSTAR_STATS per-step logging

    // calibration + runtime EMAs (us/byte units kept implicit: bytes and ns)
    double  qstar_h2d_bps     = 0.0;  // effective H2D bandwidth (bytes/s)
    double  qstar_h2d_fix_ns  = 0.0;  // fixed per-copy overhead (ns)
    double  qstar_cpu_bps     = 0.0;  // host GEMV streaming rate (bytes/s)
    int64_t qstar_expert_bytes = 0;   // weight bytes of one expert (one layer)
    int64_t n_qstar_xfer = 0;
    int64_t n_qstar_cpu  = 0;

    struct ggml_threadpool * qstar_tp = nullptr; // dedicated nested-compute pool
    std::map<int32_t, llama_moe_qstar_layer_exec> qstar_exec_by_layer;

    // hit/miss classification without assigning a slot: hits bump recency and
    // counters, misses only record the access and return -1 (used by the q*
    // planning pass, which decides per miss between transfer and CPU exec)
    int32_t find_touch(int32_t layer_id, int32_t expert_id, int64_t step) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        record_access(layer_id, expert_id);
        const int32_t slot = find(layer_id, expert_id);
        if (slot >= 0) {
            ++n_hit;
            if (auto * s = slot_at(layer_id, slot)) {
                s->last_used = ++clock;
            }
            return slot;
        }
        ++n_miss;
        return -1;
    }

    void record_selections(int32_t layer_id, const int32_t * ids, int64_t n) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        auto & v = last_selections[layer_id];
        for (int64_t i = 0; i < n; ++i) {
            if (ids[i] >= 0 && std::find(v.begin(), v.end(), ids[i]) == v.end()) {
                v.push_back(ids[i]);
            }
        }
    }

    std::vector<std::pair<int32_t, std::vector<int32_t>>> take_last_selections() {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        std::vector<std::pair<int32_t, std::vector<int32_t>>> out;
        out.reserve(last_selections.size());
        for (auto & kv : last_selections) {
            out.emplace_back(kv.first, std::move(kv.second));
        }
        last_selections.clear();
        return out;
    }

    int64_t next_clock() {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        return ++clock;
    }

    // guards all slot/bank state (expert_to_slot, slots_by_layer, banks_by_layer,
    // compute_tensor_by_src, clock, hit/miss counters). the eval-time slot remap
    // runs as one ggml_map_custom1 op per MoE layer on the CPU thread pool, so
    // different layers can mutate the cache concurrently. unsynchronized
    // unordered_map access corrupts the heap (observed crash inside the
    // access_mutex lock during record_access). recursive: helpers re-enter.
    // lock order is always cache_mutex -> access_mutex (one-way, no deadlock).
    mutable std::recursive_mutex cache_mutex;

    // access tracking for frequency stats
    // mutable: written from compute_tensor() (graph build) and read at shutdown; the
    // mutex keeps concurrent graph builds on a shared model from corrupting the bucket array
    mutable std::unordered_map<uint64_t, int64_t> access_counts; // key(layer_id, expert_id) -> count
    mutable std::mutex access_mutex;
    bool track_access = false;

    // frequency-based placement: preload experts in this list (frequency order)
    // empty means preload all (full-slot mode)
    std::vector<std::pair<int32_t, int32_t>> frequency_whitelist;

    // pooled remap userdata, one per MoE layer; pointers handed to ggml custom
    // ops must stay valid for as long as the built graph may be reused
    mutable std::map<int32_t, std::unique_ptr<llm_moe_gpu_slot_remap_userdata>> remap_userdata_by_layer;

    llm_moe_gpu_slot_remap_userdata * get_or_create_remap_userdata(int32_t layer_id, int32_t n_experts) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        auto it = remap_userdata_by_layer.find(layer_id);
        if (it == remap_userdata_by_layer.end()) {
            it = remap_userdata_by_layer.emplace(layer_id,
                    std::make_unique<llm_moe_gpu_slot_remap_userdata>()).first;
        }
        llm_moe_gpu_slot_remap_userdata * ud = it->second.get();
        ud->cache     = const_cast<llama_moe_gpu_expert_cache *>(this);
        ud->model     = owner_model;
        ud->layer_id  = layer_id;
        ud->n_experts = n_experts;
        return ud;
    }

    static uint64_t key(int32_t layer_id, int32_t expert_id) {
        return (uint64_t(uint32_t(layer_id)) << 32) | uint32_t(expert_id);
    }

    void init(int32_t n) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        n_slots = n > 0 ? n : 0;
        slots_by_layer.clear();
        banks_by_layer.clear();
        expert_to_slot.clear();
        compute_tensor_by_src.clear();
        expert_to_slot.reserve(n_slots);
        frequency_whitelist.clear();
        clock = 0;
        n_hit = 0;
        n_miss = 0;
        n_evict = 0;
        global_lru_enabled = false;
        n_resident_global = 0;
        n_evict_cross = 0;
        n_copy = 0;
        copy_bytes = 0;
        copy_ns = 0;
        last_selections.clear();
        prefill_pf_enabled      = false;
        prefill_pf_budget_bytes = 256LL * 1024 * 1024;
        prefill_pf_max_inflight = 64;
        prefill_pf_backend      = nullptr;
        prefill_pf_inflight.clear();
        n_qstar_xfer = 0;
        n_qstar_cpu = 0;
    }

    bool is_in_frequency_whitelist(int32_t layer_id, int32_t expert_id) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        if (frequency_whitelist.empty()) {
            return true; // no whitelist = full-slot mode
        }
        for (const auto& [lid, eid] : frequency_whitelist) {
            if (lid == layer_id && eid == expert_id) {
                return true;
            }
        }
        return false;
    }

    void set_frequency_whitelist(const std::vector<std::pair<int32_t, int32_t>>& experts) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        frequency_whitelist = experts;
    }

    void clear() {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        for (auto & layer_slots : slots_by_layer) {
            for (auto & slot : layer_slots.second) {
                slot.clear_storage();
            }
        }
        for (auto & layer_bank : banks_by_layer) {
            layer_bank.second.clear_storage();
        }
        n_slots = 0;
        slots_by_layer.clear();
        banks_by_layer.clear();
        expert_to_slot.clear();
        compute_tensor_by_src.clear();
        frequency_whitelist.clear();
        clock = 0;
        n_hit = 0;
        n_miss = 0;
        n_evict = 0;
        global_lru_enabled = false;
        n_resident_global = 0;
        n_evict_cross = 0;
        n_copy = 0;
        copy_bytes = 0;
        copy_ns = 0;
        last_selections.clear();
        prefill_pf_enabled      = false;
        prefill_pf_budget_bytes = 256LL * 1024 * 1024;
        prefill_pf_max_inflight = 64;
        prefill_pf_backend      = nullptr;
        prefill_pf_inflight.clear();
        n_qstar_xfer = 0;
        n_qstar_cpu = 0;
        for (auto & exec : qstar_exec_by_layer) {
            exec.second.clear_storage();
        }
        qstar_exec_by_layer.clear();
        if (qstar_tp != nullptr) {
            ggml_threadpool_free(qstar_tp);
            qstar_tp = nullptr;
        }
    }

    bool enabled() const {
        return n_slots > 0;
    }

    int32_t size() const {
        return n_slots;
    }

    bool empty() const {
        return n_slots == 0;
    }

    void register_compute_tensor(const ggml_tensor * src, ggml_tensor * compute) {
        if (src != nullptr && compute != nullptr) {
            std::lock_guard<std::recursive_mutex> lock(cache_mutex);
            compute_tensor_by_src[src] = compute;
        }
    }

    ggml_tensor * compute_tensor(const ggml_tensor * src) const {
        if (src == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        const auto it = compute_tensor_by_src.find(src);
        if (it == compute_tensor_by_src.end()) {
            return const_cast<ggml_tensor *>(src);
        }
        return it->second;
    }

    // record expert usage for the frequency report (Pass 1); called from the
    // eval-time slot remap so counts reflect real router selections
    void record_access(int32_t layer_id, int32_t expert_id) {
        if (!track_access || layer_id < 0 || expert_id < 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(access_mutex);
        access_counts[key(layer_id, expert_id)]++;
    }

    bool uses_compute_tensor(const ggml_tensor * src) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        return src != nullptr && compute_tensor_by_src.find(src) != compute_tensor_by_src.end();
    }

    // Generate frequency report from access counts
    llama_moe_freq_report generate_access_report(const std::string& fingerprint, int32_t n_active_experts) const {
        llama_moe_freq_report report;
        report.model_fingerprint = fingerprint;
        report.n_layers = 0;
        report.n_experts = 0;
        report.n_active_experts = n_active_experts;

        // snapshot under lock: access_counts can be written concurrently by graph builds
        std::unordered_map<uint64_t, int64_t> counts;
        {
            std::lock_guard<std::mutex> lock(access_mutex);
            counts = access_counts;
        }

        if (counts.empty()) {
            // No expert access was recorded. Do not fabricate a 1x1 all-zero
            // grid (the old behaviour): a zero-record report has empty stats,
            // so the load side refuses it exactly like it refuses a fingerprint
            // mismatch -- an accidental empty report can never be applied.
            fprintf(stderr, "[moe-stats] warning: no MoE access records collected - "
                    "writing zero-record frequency report (load side will refuse it)\n");
            return report;
        }

        // find max layer and expert IDs
        int32_t max_layer = 0, max_expert = 0;
        for (const auto& [k, v] : counts) {
            int32_t lid = int32_t(k >> 32);
            int32_t eid = int32_t(k & 0xFFFFFFFF);
            if (lid > max_layer) max_layer = lid;
            if (eid > max_expert) max_expert = eid;
        }
        report.n_layers = max_layer + 1;
        report.n_experts = max_expert + 1;

        report.stats.resize(report.n_layers * report.n_experts);
        for (int32_t l = 0; l < report.n_layers; l++) {
            for (int32_t e = 0; e < report.n_experts; e++) {
                auto& s = report.stats[l * report.n_experts + e];
                s.layer_id = l;
                s.expert_id = e;
                s.prompt_selections = 0;
                s.gen_selections = 0;
                s.total_selections = 0;
                s.estimated_weight_bytes = 0;
                auto it = counts.find(key(l, e));
                if (it != counts.end()) {
                    s.total_selections = it->second;
                    s.gen_selections = it->second;
                }
            }
        }

        report.sorted_by_frequency.resize(report.n_layers * report.n_experts);
        std::iota(report.sorted_by_frequency.begin(), report.sorted_by_frequency.end(), 0);
        std::sort(report.sorted_by_frequency.begin(), report.sorted_by_frequency.end(),
            [&](int32_t a, int32_t b) {
                return report.stats[a].total_selections > report.stats[b].total_selections;
            });

        return report;
    }

    std::vector<llama_moe_gpu_expert_slot> & slots_for_layer(int32_t layer_id) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        auto & layer_slots = slots_by_layer[layer_id];
        if ((int32_t) layer_slots.size() != n_slots) {
            layer_slots.resize(n_slots);
        }
        return layer_slots;
    }

    const std::vector<llama_moe_gpu_expert_slot> * slots_for_layer(int32_t layer_id) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        const auto it = slots_by_layer.find(layer_id);
        if (it == slots_by_layer.end()) {
            return nullptr;
        }
        return &it->second;
    }

    llama_moe_gpu_expert_bank & bank_for_layer(int32_t layer_id) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        return banks_by_layer[layer_id];
    }

    const llama_moe_gpu_expert_bank * bank_for_layer(int32_t layer_id) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        const auto it = banks_by_layer.find(layer_id);
        if (it == banks_by_layer.end()) {
            return nullptr;
        }
        return &it->second;
    }

    llama_moe_gpu_expert_slot * slot_at(int32_t layer_id, int32_t slot_id) {
        if (slot_id < 0 || slot_id >= n_slots) {
            return nullptr;
        }
        return &slots_for_layer(layer_id)[slot_id];
    }

    const llama_moe_gpu_expert_slot * slot_at(int32_t layer_id, int32_t slot_id) const {
        if (slot_id < 0 || slot_id >= n_slots) {
            return nullptr;
        }
        const auto * layer_slots = slots_for_layer(layer_id);
        if (layer_slots == nullptr || slot_id >= (int32_t) layer_slots->size()) {
            return nullptr;
        }
        return &(*layer_slots)[slot_id];
    }

    int32_t find(int32_t layer_id, int32_t expert_id) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        const auto it = expert_to_slot.find(key(layer_id, expert_id));
        if (it == expert_to_slot.end()) {
            return -1;
        }
        const int32_t slot_id = it->second;
        if (slot_id < 0 || slot_id >= n_slots) {
            return -1;
        }
        const auto * slot = slot_at(layer_id, slot_id);
        if (slot == nullptr || !slot->resident || slot->layer_id != layer_id || slot->expert_id != expert_id) {
            return -1;
        }
        return slot_id;
    }

    int32_t find_free(int32_t layer_id) const {
        const auto * layer_slots = slots_for_layer(layer_id);
        if (layer_slots == nullptr) {
            return 0;
        }
        for (int32_t i = 0; i < n_slots; ++i) {
            if (!(*layer_slots)[i].resident) {
                return i;
            }
        }
        return -1;
    }

    int32_t find_lru_victim(int32_t layer_id) const {
        if (n_slots <= 0) {
            return -1;
        }
        const auto * layer_slots = slots_for_layer(layer_id);
        if (layer_slots == nullptr) {
            return 0;
        }
        int32_t victim = 0;
        for (int32_t i = 1; i < n_slots; ++i) {
            if ((*layer_slots)[i].last_used < (*layer_slots)[victim].last_used) {
                victim = i;
            }
        }
        return victim;
    }

    // globally-oldest resident across all layers; false if nothing is resident.
    // used by the global LRU pool to let hot layers take budget from cold ones
    bool find_global_lru_victim(int32_t & victim_layer, int32_t & victim_slot) const {
        victim_layer = -1;
        victim_slot  = -1;
        for (const auto & layer_slots : slots_by_layer) {
            for (int32_t i = 0; i < (int32_t) layer_slots.second.size(); ++i) {
                const auto & s = layer_slots.second[i];
                if (!s.resident) {
                    continue;
                }
                if (victim_slot < 0 || s.last_used < slots_by_layer.at(victim_layer)[victim_slot].last_used) {
                    victim_layer = layer_slots.first;
                    victim_slot  = i;
                }
            }
        }
        return victim_slot >= 0;
    }

    // any resident slot of this layer, or -1; used to keep paging-regime
    // graphs from ever receiving an unbanked logical id on materialize failure
    int32_t find_any_resident(int32_t layer_id) const {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        const auto * layer_slots = slots_for_layer(layer_id);
        if (layer_slots == nullptr) {
            return -1;
        }
        for (int32_t i = 0; i < (int32_t) layer_slots->size(); ++i) {
            if ((*layer_slots)[i].resident) {
                return i;
            }
        }
        return -1;
    }

    void assign_slot(int32_t slot_id, int32_t layer_id, int32_t expert_id, int64_t step) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        if (slot_id < 0 || slot_id >= n_slots) {
            return;
        }
        auto & s = slots_for_layer(layer_id)[slot_id];
        if (s.resident) {
            expert_to_slot.erase(key(s.layer_id, s.expert_id));
            s.clear_storage();
        } else {
            ++n_resident_global;
        }
        s.layer_id = layer_id;
        s.expert_id = expert_id;
        s.last_used = step;
        s.resident = true;
        expert_to_slot[key(layer_id, expert_id)] = slot_id;
    }

    // undo a resident marking whose storage could not be materialized; keeps
    // expert_to_slot from ever pointing at an empty slot
    void release_slot(int32_t layer_id, int32_t slot_id) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        const auto it_layer = slots_by_layer.find(layer_id);
        if (it_layer == slots_by_layer.end() || slot_id < 0 || slot_id >= (int32_t) it_layer->second.size()) {
            return;
        }
        auto & s = it_layer->second[slot_id];
        if (s.resident) {
            expert_to_slot.erase(key(s.layer_id, s.expert_id));
            --n_resident_global;
        }
        s.clear_storage();
        s.resident  = false;
        s.bank_backed = false;
        s.layer_id  = -1;
        s.expert_id = -1;
    }

    int32_t get_or_assign_slot(int32_t layer_id, int32_t expert_id, int64_t step) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        clock = std::max(clock, step);

        const int32_t hit_slot = find(layer_id, expert_id);
        if (hit_slot >= 0) {
            ++n_hit;
            if (auto * slot = slot_at(layer_id, hit_slot)) {
                slot->last_used = step;
            }
            return hit_slot;
        }

        ++n_miss;
        int32_t slot = find_free(layer_id);
        if (slot < 0) {
            slot = find_lru_victim(layer_id);
            const auto * victim = slot_at(layer_id, slot);
            if (victim != nullptr && victim->resident) {
                ++n_evict;
            }
        }

        if (slot >= 0) {
            assign_slot(slot, layer_id, expert_id, step);
        }
        return slot;
    }

    int32_t preload_or_assign_slot(int32_t layer_id, int32_t expert_id, int64_t step) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        clock = std::max(clock, step);

        const int32_t hit_slot = find(layer_id, expert_id);
        if (hit_slot >= 0) {
            if (auto * slot = slot_at(layer_id, hit_slot)) {
                slot->last_used = step;
            }
            return hit_slot;
        }

        int32_t slot = find_free(layer_id);
        if (slot < 0) {
            slot = find_lru_victim(layer_id);
        }

        if (slot >= 0) {
            assign_slot(slot, layer_id, expert_id, step);
        }
        return slot;
    }

    int32_t ensure_resident(int32_t layer_id, int32_t expert_id, int32_t n_experts) {
        if (!enabled() || expert_id < 0 || expert_id >= n_experts) {
            return expert_id;
        }

        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        record_access(layer_id, expert_id);

        // Pass 1 collection (track_access) counts access only. Plain paging
        // runs (slots < experts, no whitelist) historically did the same
        // because graphs never redirect weights to banks in that regime; the
        // global LRU pool and the q* policy are what make dynamic banking
        // reachable end-to-end, so they lift the skip (graphs extend their
        // redirect accordingly).
        if (frequency_whitelist.empty() && n_slots < n_experts &&
                (track_access || !(global_lru_enabled || qstar_enabled))) {
            return expert_id;
        }

        int32_t slot = find(layer_id, expert_id);
        if (slot >= 0) {
            ++n_hit;
            if (auto * s = slot_at(layer_id, slot)) {
                s->last_used = ++clock;
            }
            return slot;
        }

        ++n_miss;

        // global LRU pool: the budget is shared across layers. when it is
        // exhausted, evict the globally-oldest resident wherever it lives so
        // hot layers can grow past their fair share. local-full implies
        // global-full-and-entirely-local (per-layer capacity == n_slots), in
        // which case the victim is local and a position frees up here too.
        const bool global_lru = global_lru_enabled && frequency_whitelist.empty();
        if (global_lru && n_resident_global >= n_slots) {
            int32_t victim_layer = -1;
            int32_t victim_slot  = -1;
            if (find_global_lru_victim(victim_layer, victim_slot)) {
                const auto * victim = slot_at(victim_layer, victim_slot);
                if (victim != nullptr && victim->resident) {
                    ++n_evict;
                    if (victim_layer != layer_id) {
                        ++n_evict_cross;
                    }
                }
                release_slot(victim_layer, victim_slot);
            }
        }

        slot = find_free(layer_id);
        if (slot < 0) {
            slot = find_lru_victim(layer_id);
            if (slot >= 0) {
                const auto * victim = slot_at(layer_id, slot);
                if (victim != nullptr && victim->resident) {
                    ++n_evict;
                }
            }
        }

        if (slot < 0) {
            return expert_id;
        }

        slot = preload_or_assign_slot(layer_id, expert_id, ++clock);
        if (slot < 0) {
            return safe_unbanked_fallback(layer_id, expert_id, n_experts);
        }

        if (materialize_cb != nullptr && !materialize_cb(materialize_userdata, slot, layer_id, expert_id, n_experts)) {
            // storage could not be allocated (e.g. VRAM exhausted); undo the
            // resident marking so the expert falls back to its CPU tensor
            release_slot(layer_id, slot);
            return safe_unbanked_fallback(layer_id, expert_id, n_experts);
        }

        return slot;
    }

    // when the graph indexes bank tensors with our return value (any paging
    // regime), an unbanked logical id would read past the bank; hand back a
    // resident slot instead. full-slot mode can return the logical id itself.
    int32_t safe_unbanked_fallback(int32_t layer_id, int32_t expert_id, int32_t n_experts) const {
        if (n_slots < n_experts) {
            const int32_t any = find_any_resident(layer_id);
            return any >= 0 ? any : 0;
        }
        return expert_id;
    }
};

// inter-step speculative expert prefetch; budget_ms <= 0 disables
void llama_moe_gpu_expert_slot_prefetch(struct llama_model & model, double budget_ms);

// elastic VRAM sizing: resolve cache.auto_pending into a concrete slot count
// from free VRAM after model weights + KV caches are resident; runs the
// standard preload. no-op unless auto_pending is set.
void llama_moe_gpu_expert_slot_auto_init(struct llama_model & model);

// prefill double buffering: poll completed background copies, then enqueue
// predicted-expert H2D copies for the next prefill chunk on backend's copy
// stream (opt-in via LLAMA_MOE_PREFILL_PF=1). backend may be null (falls back
// to a no-op).
void llama_moe_gpu_expert_slot_prefill_prefetch(struct llama_model & model, struct ggml_backend * backend);

// prefill double buffering teardown: synchronize and release any still-in-
// flight background copies (model unload path).
void llama_moe_gpu_expert_slot_prefill_shutdown(struct llama_model & model);

// q* bandwidth-adaptive policy: measure effective H2D and host-side GEMV
// rates once after model load (LLAMA_MOE_SLOT_STATS logs the results)
void llama_moe_gpu_qstar_calibrate(struct llama_model & model);

// host-side execution of the given experts' FFN against the token vector x
// (n_embd floats); accumulates UNWEIGHTED outputs into partials_out laid out
// as [i * n_embd + d]. returns false when the layer's layout is unsupported
// (caller then falls back to transferring that expert instead).
bool llama_moe_gpu_qstar_cpu_exec(struct llama_model & model, int32_t layer_id,
        const int32_t * experts, int32_t n, const float * x, float * partials_out);

// lazily build (or fetch) the layer's q* host-exec mini graph without running
// it; lets the planning op know up front whether CPU exec is available so
// unsupported layers transfer-always instead of deferring into a void
bool llama_moe_qstar_exec_prepare(struct llama_model & model, int32_t layer_id);

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    // for classifier models
    std::vector<std::string> classifier_labels;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    // longcat-flash-ngram: NgramEmbedding module
    // embedders/post_projs count = emb_split_num * (emb_neighbor_num - 1) = 4*3 = 12
    struct ggml_tensor * ngram_embd[12] = {};
    struct ggml_tensor * ngram_proj[12] = {};

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;


    // NVFP4 per-tensor scale2, input_scale for LM head
    struct ggml_tensor * output_s    = nullptr;
    struct ggml_tensor * output_in_s = nullptr;

    // NextN/MTP model-level projections
    struct ggml_tensor * nextn_proj_pre  = nullptr;
    struct ggml_tensor * nextn_proj_post = nullptr;

    // DeepSeek-V4
    struct ggml_tensor * hc_head_fn    = nullptr;
    struct ggml_tensor * hc_head_base  = nullptr;
    struct ggml_tensor * hc_head_scale = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;
    struct ggml_tensor * cls_norm  = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    // gemma3n altup
    struct ggml_tensor * altup_proj           = nullptr;
    struct ggml_tensor * altup_unembd_proj    = nullptr;
    struct ggml_tensor * per_layer_tok_embd   = nullptr;
    struct ggml_tensor * per_layer_model_proj = nullptr;
    struct ggml_tensor * per_layer_proj_norm  = nullptr;

    // eagle3
    struct ggml_tensor * fc  = nullptr;  // feature fusion layer
    struct ggml_tensor * d2t = nullptr;  // draft to target vocabulary mapping

    // dspark
    struct ggml_tensor * dspark_markov_w1   = nullptr;
    struct ggml_tensor * dspark_markov_w2   = nullptr;
    struct ggml_tensor * dspark_conf_proj   = nullptr;
    struct ggml_tensor * dspark_conf_proj_b = nullptr;

    // unified vector to store target-model extracted layer ids in eagle3, dflash, etc.
    std::vector<int32_t> target_layer_ids;

    std::vector<llama_layer> layers;

    //Dense linear projections for SentenceTransformers models like embeddinggemma
    // For Sentence Transformers models structure see
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    struct ggml_tensor * dense_2_out_layers   = nullptr;
    struct ggml_tensor * dense_2_out_layers_b = nullptr;
    struct ggml_tensor * dense_3_out_layers   = nullptr;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<llama_device> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    // for keeping track of associated LoRA adapters
    std::unordered_set<llama_adapter_lora *> loras;

    // statically allocated context for assigning
    struct llama_meta_device_get_split_state_userdata get_split_state_ud;

    llama_moe_gpu_expert_cache moe_gpu_expert_cache;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const llama_model_params & params);
    virtual ~llama_model();

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    llama_ftype ftype() const;

    size_t size() const; // file size
    size_t n_tensors() const;
    size_t n_devices() const;
    const float * tensor_split() const;

    uint32_t n_gpu_layers() const;
    llama_split_mode split_mode() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(int il) const;

    bool has_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;

    ggml_cgraph * build_graph(const llm_graph_params & params) const;

    virtual void load_stats  (llama_model_loader & ml) = 0;
    virtual void load_hparams(llama_model_loader & ml) = 0;
    virtual void load_vocab  (llama_model_loader & ml) = 0;
    virtual bool load_tensors(llama_model_loader & ml) = 0; // returns false if cancelled by progress_callback

    // model must define these
    virtual void load_arch_hparams(llama_model_loader & ml) = 0;
    virtual void load_arch_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const = 0;

protected:
    llama_model_params params;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params);
llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params);

// model must inherit from this
struct llama_model_base : public llama_model {
    friend struct llama_model;

    llama_model * model;
    llama_model_loader * ml = nullptr;
    const LLM_TN tn;

    // llama_model_loader is not yet defined at this point, so we will set it after construction
    const int TENSOR_DUPLICATED;
    const int TENSOR_NOT_REQUIRED;
    const int TENSOR_SKIP;
    const int TENSOR_SKIP_IF_VIRTUAL;
    const int TENSOR_ALLOW_RESHAPE;

    explicit llama_model_base(const llama_model_params & params);
    virtual ~llama_model_base() = default;

    ggml_tensor * create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    // convenience overload of create_tensor that doesn't require llama_model_loader
    ggml_tensor * create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    // helper: try merged gate_up_exps first, fall back to separate gate and up
    void create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_,
                int64_t n_ff_, int64_t n_expert_, int flags);

    // helper: try to load merged qkv first, fall back to separate q, k, v
    void create_tensor_qkv(llama_layer & layer, int bid,
                int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
                int flags);

    void load_stats  (llama_model_loader & ml) override;
    void load_hparams(llama_model_loader & ml) override;
    void load_vocab  (llama_model_loader & ml) override;
    bool load_tensors(llama_model_loader & ml) override;

    // model must define these
    void load_arch_hparams(llama_model_loader & ml) override = 0;
    void load_arch_tensors(llama_model_loader & ml) override = 0;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override = 0;
};

const char * llm_type_name(llm_type type);

// convenience macro for loading local variables for load_tensors() in llama_model_base
// note: cast to int64_t since we will use these for the tensor dimensions
#define LLAMA_LOAD_LOCALS \
    const int     n_layer        = hparams.n_layer();        GGML_UNUSED(n_layer); \
    const int     n_layer_all    = hparams.n_layer_all;      GGML_UNUSED(n_layer_all); \
    const int     n_layer_nextn  = hparams.n_layer_nextn;    GGML_UNUSED(n_layer_nextn); \
    const int64_t n_head         = hparams.n_head();         GGML_UNUSED(n_head); \
    const int64_t n_head_kv      = hparams.n_head_kv();      GGML_UNUSED(n_head_kv); \
    const int64_t n_embd         = hparams.n_embd;           GGML_UNUSED(n_embd); \
    const int64_t n_embd_k_gqa   = hparams.n_embd_k_gqa();   GGML_UNUSED(n_embd_k_gqa); \
    const int64_t n_embd_v_gqa   = hparams.n_embd_v_gqa();   GGML_UNUSED(n_embd_v_gqa); \
    const int64_t n_embd_head_k  = hparams.n_embd_head_k();  GGML_UNUSED(n_embd_head_k); \
    const int64_t n_embd_head_v  = hparams.n_embd_head_v();  GGML_UNUSED(n_embd_head_v); \
    const int64_t n_ff           = hparams.n_ff();           GGML_UNUSED(n_ff); \
    const int64_t n_embd_gqa     = n_embd_v_gqa;             GGML_UNUSED(n_embd_gqa); \
    const int64_t n_vocab        = vocab.n_tokens();         GGML_UNUSED(n_vocab); \
    const int64_t n_token_types  = vocab.n_token_types();    GGML_UNUSED(n_token_types); \
    const int64_t n_rot          = hparams.n_rot();          GGML_UNUSED(n_rot); \
    const int64_t n_expert       = hparams.n_expert;         GGML_UNUSED(n_expert); \
    const int64_t n_expert_used  = hparams.n_expert_used;    GGML_UNUSED(n_expert_used); \
    const int64_t n_ctx_train    = hparams.n_ctx_train;      GGML_UNUSED(n_ctx_train);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
