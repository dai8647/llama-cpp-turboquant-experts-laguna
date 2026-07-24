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

struct llama_moe_expert_placement {
    int32_t layer_id;
    int32_t expert_id;
    enum device_t { DEVICE_GPU, DEVICE_CPU } device;
    int32_t gpu_slot_id;
};

struct llama_moe_placement_plan {
    std::string model_fingerprint;
    int32_t n_layers;
    int32_t n_experts;
    int32_t n_gpu_experts;
    int32_t n_cpu_experts;
    int64_t estimated_gpu_bytes;
    std::vector<llama_moe_expert_placement> placements;
};

class llama_moe_expert_counter {
public:
    llama_moe_expert_counter(int32_t n_layers, int32_t n_experts);
    void record_selection(int32_t layer_id, int32_t expert_id);
    void reset();
    void swap_counts();
    llama_moe_freq_report generate_report(
        const std::string& fingerprint, int32_t n_active_experts);
private:
    std::vector<std::vector<int64_t>> counts_;
    std::vector<std::vector<int64_t>> counts_prompt_;
    int32_t n_layers_, n_experts_;
};

bool save_freq_report(const llama_moe_freq_report& report,
                      const std::string& path);
llama_moe_freq_report load_freq_report(const std::string& path);
bool save_placement_plan(const llama_moe_placement_plan& plan,
                         const std::string& path);
llama_moe_placement_plan load_placement_plan(const std::string& path);
