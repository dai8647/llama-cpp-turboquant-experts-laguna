#pragma once
#include "llama-moe-stats.h"
#include <vector>

// llama_moe_expert_placement and llama_moe_placement_plan are defined in llama-moe-stats.h

// Generate a frequency-based placement plan.
// report: frequency statistics (must have n_layers * n_experts entries)
// vram_budget_bytes: total VRAM budget for expert placement
// avg_expert_weight_bytes: average weight size per expert (for budget calculation)
// Returns a plan with DEVICE_GPU for top-frequency experts, DEVICE_CPU for the rest.
llama_moe_placement_plan generate_frequency_plan(
    const llama_moe_freq_report& report,
    int64_t vram_budget_bytes,
    int64_t avg_expert_weight_bytes);
