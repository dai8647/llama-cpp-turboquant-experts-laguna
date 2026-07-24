#include "llama-moe-stats.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>

llama_moe_expert_counter::llama_moe_expert_counter(int32_t n_layers, int32_t n_experts)
    : n_layers_(n_layers), n_experts_(n_experts)
{
    counts_.resize(n_layers, std::vector<int64_t>(n_experts, 0));
    counts_prompt_.resize(n_layers, std::vector<int64_t>(n_experts, 0));
}

void llama_moe_expert_counter::record_selection(int32_t layer_id, int32_t expert_id) {
    if (layer_id >= 0 && layer_id < n_layers_ &&
        expert_id >= 0 && expert_id < n_experts_) {
        counts_[layer_id][expert_id]++;
    }
}

void llama_moe_expert_counter::reset() {
    for (auto& layer : counts_) {
        std::fill(layer.begin(), layer.end(), 0);
    }
    for (auto& layer : counts_prompt_) {
        std::fill(layer.begin(), layer.end(), 0);
    }
}

void llama_moe_expert_counter::swap_counts() {
    counts_prompt_ = counts_;
    for (auto& layer : counts_) {
        std::fill(layer.begin(), layer.end(), 0);
    }
}

llama_moe_freq_report llama_moe_expert_counter::generate_report(
    const std::string& fingerprint, int32_t n_active_experts)
{
    llama_moe_freq_report report;
    report.model_fingerprint = fingerprint;
    report.n_layers = n_layers_;
    report.n_experts = n_experts_;
    report.n_active_experts = n_active_experts;

    report.stats.resize(n_layers_ * n_experts_);
    for (int32_t l = 0; l < n_layers_; l++) {
        for (int32_t e = 0; e < n_experts_; e++) {
            auto& s = report.stats[l * n_experts_ + e];
            s.layer_id = l;
            s.expert_id = e;
            s.prompt_selections = counts_prompt_[l][e];
            s.gen_selections = counts_[l][e];
            s.total_selections = counts_prompt_[l][e] + counts_[l][e];
            s.estimated_weight_bytes = 0;
        }
    }

    report.sorted_by_frequency.resize(n_layers_ * n_experts_);
    std::iota(report.sorted_by_frequency.begin(),
               report.sorted_by_frequency.end(), 0);
    std::sort(report.sorted_by_frequency.begin(),
              report.sorted_by_frequency.end(),
              [&](int32_t a, int32_t b) {
                  return report.stats[a].total_selections >
                         report.stats[b].total_selections;
              });

    return report;
}

bool save_freq_report(const llama_moe_freq_report& report,
                      const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) return false;

    f << "{\n";
    f << "  \"schema_version\": 1,\n";
    f << "  \"model_fingerprint\": \"" << report.model_fingerprint << "\",\n";
    f << "  \"n_layers\": " << report.n_layers << ",\n";
    f << "  \"n_experts\": " << report.n_experts << ",\n";
    f << "  \"n_active_experts\": " << report.n_active_experts << ",\n";
    f << "  \"layers\": [\n";

    for (int32_t l = 0; l < report.n_layers; l++) {
        f << "    {\n";
        f << "      \"layer\": " << l << ",\n";
        f << "      \"experts\": [\n";

        for (int32_t e = 0; e < report.n_experts; e++) {
            const auto& s = report.stats[l * report.n_experts + e];
            f << "        {";
            f << "\"expert\": " << e << ", ";
            f << "\"prompt_selections\": " << s.prompt_selections << ", ";
            f << "\"gen_selections\": " << s.gen_selections << ", ";
            f << "\"total_selections\": " << s.total_selections;
            f << "}";
            if (e < report.n_experts - 1) f << ",";
            f << "\n";
        }

        f << "      ]\n";
        f << "    }";
        if (l < report.n_layers - 1) f << ",";
        f << "\n";
    }

    f << "  ]\n";
    f << "}\n";
    return true;
}

static std::string json_get_string(const std::string& json,
                                   const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    return json.substr(pos + 1, end - pos - 1);
}

static int64_t json_get_int(const std::string& json,
                            const std::string& key, int64_t def = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    char* end = nullptr;
    long long val = std::strtoll(json.c_str() + pos, &end, 10);
    return (end != json.c_str() + pos) ? val : def;
}

static std::vector<std::string> json_split_objects(const std::string& arr) {
    std::vector<std::string> result;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == '{') depth++;
        else if (arr[i] == '}') {
            depth--;
            if (depth == 0) {
                result.push_back(arr.substr(start, i - start + 1));
                start = i + 1;
                while (start < arr.size() && (arr[start] == ',' || arr[start] == ' ' || arr[start] == '\n'))
                    start++;
            }
        }
    }
    return result;
}

llama_moe_freq_report load_freq_report(const std::string& path) {
    llama_moe_freq_report report = {};
    std::ifstream f(path);
    if (!f.is_open()) return report;

    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    report.model_fingerprint = json_get_string(json, "model_fingerprint");
    report.n_layers = (int32_t)json_get_int(json, "n_layers");
    report.n_experts = (int32_t)json_get_int(json, "n_experts");
    report.n_active_experts = (int32_t)json_get_int(json, "n_active_experts");

    auto layers_pos = json.find("\"layers\"");
    if (layers_pos == std::string::npos) return report;

    auto arr_start = json.find('[', layers_pos);
    auto arr_end = json.rfind(']');
    if (arr_start == std::string::npos || arr_end == std::string::npos) return report;
    std::string layers_arr = json.substr(arr_start + 1, arr_end - arr_start - 1);

    auto layer_objs = json_split_objects(layers_arr);
    for (auto& lo : layer_objs) {
        int32_t layer = (int32_t)json_get_int(lo, "layer");
        auto exp_pos = lo.find("\"experts\"");
        if (exp_pos == std::string::npos) continue;
        auto ea_start = lo.find('[', exp_pos);
        auto ea_end = lo.rfind(']');
        if (ea_start == std::string::npos || ea_end == std::string::npos) continue;
        std::string exp_arr = lo.substr(ea_start + 1, ea_end - ea_start - 1);
        auto expert_objs = json_split_objects(exp_arr);
        for (auto& eo : expert_objs) {
            int32_t expert = (int32_t)json_get_int(eo, "expert");
            llama_moe_expert_stats s = {};
            s.layer_id = layer;
            s.expert_id = expert;
            s.prompt_selections = json_get_int(eo, "prompt_selections");
            s.gen_selections = json_get_int(eo, "gen_selections");
            s.total_selections = json_get_int(eo, "total_selections");
            s.estimated_weight_bytes = 0;
            report.stats.push_back(s);
        }
    }

    report.sorted_by_frequency.resize(report.stats.size());
    std::iota(report.sorted_by_frequency.begin(),
              report.sorted_by_frequency.end(), 0);
    std::sort(report.sorted_by_frequency.begin(),
              report.sorted_by_frequency.end(),
              [&](int32_t a, int32_t b) {
                  return report.stats[a].total_selections >
                         report.stats[b].total_selections;
              });

    return report;
}

bool save_placement_plan(const llama_moe_placement_plan& plan,
                         const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) return false;

    f << "{\n";
    f << "  \"schema_version\": 1,\n";
    f << "  \"model_fingerprint\": \"" << plan.model_fingerprint << "\",\n";
    f << "  \"n_layers\": " << plan.n_layers << ",\n";
    f << "  \"n_experts\": " << plan.n_experts << ",\n";
    f << "  \"n_gpu_experts\": " << plan.n_gpu_experts << ",\n";
    f << "  \"n_cpu_experts\": " << plan.n_cpu_experts << ",\n";
    f << "  \"estimated_gpu_bytes\": " << plan.estimated_gpu_bytes << ",\n";
    f << "  \"placements\": [\n";

    for (size_t i = 0; i < plan.placements.size(); i++) {
        const auto& p = plan.placements[i];
        f << "    {\"layer\": " << p.layer_id
          << ", \"expert\": " << p.expert_id
          << ", \"device\": \"" << (p.device == llama_moe_expert_placement::DEVICE_GPU ? "gpu" : "cpu")
          << "\", \"gpu_slot_id\": " << p.gpu_slot_id << "}";
        if (i < plan.placements.size() - 1) f << ",";
        f << "\n";
    }

    f << "  ]\n";
    f << "}\n";
    return true;
}

llama_moe_placement_plan load_placement_plan(const std::string& path) {
    llama_moe_placement_plan plan = {};
    std::ifstream f(path);
    if (!f.is_open()) return plan;

    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    plan.model_fingerprint = json_get_string(json, "model_fingerprint");
    plan.n_layers = (int32_t)json_get_int(json, "n_layers");
    plan.n_experts = (int32_t)json_get_int(json, "n_experts");
    plan.n_gpu_experts = (int32_t)json_get_int(json, "n_gpu_experts");
    plan.n_cpu_experts = (int32_t)json_get_int(json, "n_cpu_experts");
    plan.estimated_gpu_bytes = json_get_int(json, "estimated_gpu_bytes");

    auto pp_pos = json.find("\"placements\"");
    if (pp_pos == std::string::npos) return plan;
    auto arr_start = json.find('[', pp_pos);
    auto arr_end = json.rfind(']');
    if (arr_start == std::string::npos || arr_end == std::string::npos) return plan;
    std::string arr = json.substr(arr_start + 1, arr_end - arr_start - 1);

    auto objs = json_split_objects(arr);
    for (auto& o : objs) {
        llama_moe_expert_placement p = {};
        p.layer_id = (int32_t)json_get_int(o, "layer");
        p.expert_id = (int32_t)json_get_int(o, "expert");
        std::string dev = json_get_string(o, "device");
        p.device = (dev == "gpu") ? llama_moe_expert_placement::DEVICE_GPU
                                  : llama_moe_expert_placement::DEVICE_CPU;
        p.gpu_slot_id = (int32_t)json_get_int(o, "gpu_slot_id", -1);
        plan.placements.push_back(p);
    }

    return plan;
}
