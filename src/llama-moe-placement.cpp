#include "llama-moe-placement.h"

llama_moe_placement_plan generate_frequency_plan(
    const llama_moe_freq_report& report,
    int64_t vram_budget_bytes,
    int64_t avg_expert_weight_bytes)
{
    llama_moe_placement_plan plan;
    plan.model_fingerprint = report.model_fingerprint;
    plan.n_layers = report.n_layers;
    plan.n_experts = report.n_experts;
    plan.n_gpu_experts = 0;
    plan.n_cpu_experts = 0;
    plan.estimated_gpu_bytes = 0;

    // total_experts = layers * experts_per_layer (all (layer, expert) pairs)
    const int32_t total_experts = report.n_layers * report.n_experts;

    int32_t gpu_budget_count = (avg_expert_weight_bytes > 0)
        ? static_cast<int32_t>(vram_budget_bytes / avg_expert_weight_bytes)
        : total_experts;

    if (gpu_budget_count > total_experts) {
        gpu_budget_count = total_experts;
    }

    plan.placements.resize(total_experts);

    for (int32_t i = 0; i < total_experts; i++) {
        int32_t idx = report.sorted_by_frequency[i];
        auto& p = plan.placements[idx];
        p.layer_id = report.stats[idx].layer_id;
        p.expert_id = report.stats[idx].expert_id;

        if (i < gpu_budget_count) {
            p.device = llama_moe_expert_placement::DEVICE_GPU;
            p.gpu_slot_id = plan.n_gpu_experts;
            plan.n_gpu_experts++;
            plan.estimated_gpu_bytes += avg_expert_weight_bytes;
        } else {
            p.device = llama_moe_expert_placement::DEVICE_CPU;
            p.gpu_slot_id = -1;
            plan.n_cpu_experts++;
        }
    }

    return plan;
}
