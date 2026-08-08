#pragma once
#include <cstdint>
#include <vector>
#include <string>

struct llama_moe_expert_stats {
    int32_t layer_id;
    int32_t expert_id;
    int64_t prompt_selections;
    int64_t gen_selections;
    int64_t total_selections;
    double  estimated_weight_bytes;
};

struct llama_moe_freq_report {
    std::string model_fingerprint;
    int32_t n_layers;
    int32_t n_experts;
    int32_t n_active_experts;
    std::vector<llama_moe_expert_stats> stats;
    std::vector<int32_t> sorted_by_frequency;
};

bool save_freq_report(const llama_moe_freq_report& report,
                      const std::string& path);
llama_moe_freq_report load_freq_report(const std::string& path);
