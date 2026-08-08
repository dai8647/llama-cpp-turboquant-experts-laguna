#include "llama-moe-stats.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>

using ordered_json = nlohmann::ordered_json;

bool save_freq_report(const llama_moe_freq_report& report,
                      const std::string& path) {
    // Defensive: refuse structurally inconsistent reports instead of reading
    // out of bounds. Callers build reports via generate_access_report, which
    // maintains stats.size() == n_layers * n_experts; a mismatch here means a
    // bug upstream and the file should not be written.
    if (report.n_layers < 0 || report.n_experts < 0 ||
        (int64_t) report.stats.size() != (int64_t) report.n_layers * report.n_experts) {
        fprintf(stderr, "[moe-stats] error: refusing to save structurally inconsistent "
                "frequency report (n_layers=%d, n_experts=%d, stats=%zu)\n",
                report.n_layers, report.n_experts, report.stats.size());
        return false;
    }

    ordered_json j;
    j["schema_version"] = 1;
    j["model_fingerprint"] = report.model_fingerprint;
    j["n_layers"] = report.n_layers;
    j["n_experts"] = report.n_experts;
    j["n_active_experts"] = report.n_active_experts;
    j["layers"] = ordered_json::array();

    for (int32_t l = 0; l < report.n_layers; l++) {
        ordered_json jl;
        jl["layer"] = l;
        jl["experts"] = ordered_json::array();

        for (int32_t e = 0; e < report.n_experts; e++) {
            const auto & s = report.stats[l * report.n_experts + e];
            jl["experts"].push_back(ordered_json{
                {"expert", e},
                {"prompt_selections", s.prompt_selections},
                {"gen_selections", s.gen_selections},
                {"total_selections", s.total_selections},
            });
        }

        j["layers"].push_back(std::move(jl));
    }

    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << j.dump(2) << "\n";
    return f.good();
}

static void report_parse_error(const char * path, const char * what) {
    fprintf(stderr, "[moe-stats] error: frequency report '%s': %s; refusing to apply the report\n",
            path, what);
}

llama_moe_freq_report load_freq_report(const std::string& path) {
    llama_moe_freq_report report = {};

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[moe-stats] error: cannot open frequency report '%s'\n", path.c_str());
        return report;
    }

    const std::string text((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());

    ordered_json j;
    try {
        j = ordered_json::parse(text);
    } catch (const nlohmann::json::parse_error & e) {
        report_parse_error(path.c_str(), e.what());
        return report;
    }

    // ---- header ----
    try {
        report.model_fingerprint = j.value("model_fingerprint", std::string());
        report.n_layers          = j.value("n_layers", 0);
        report.n_experts         = j.value("n_experts", 0);
        report.n_active_experts  = j.value("n_active_experts", 0);
    } catch (const nlohmann::json::exception & e) {
        report_parse_error(path.c_str(), e.what());
        return report;
    }

    if (report.n_layers < 0 || report.n_experts < 0) {
        report_parse_error(path.c_str(), "negative n_layers or n_experts");
        return report;
    }

    if (!j.contains("layers") || !j["layers"].is_array()) {
        report_parse_error(path.c_str(), "missing 'layers' array");
        return report;
    }

    const auto & layers = j["layers"];

    // The header must describe the actual array contents. A mismatch means the
    // report was truncated, hand-edited, or generated for a different model
    // shape -- applying it would misplace experts, so refuse it outright.
    if (layers.size() != (size_t) report.n_layers) {
        char msg[256];
        snprintf(msg, sizeof(msg), "header n_layers=%d but 'layers' contains %zu entries",
                 report.n_layers, layers.size());
        report_parse_error(path.c_str(), msg);
        return report;
    }

    try {
        std::vector<llama_moe_expert_stats> stats;
        stats.reserve((size_t) report.n_layers * report.n_experts);
        for (size_t l = 0; l < layers.size(); l++) {
            const auto & jl = layers[l];
            if (!jl.is_object() || !jl.contains("layer") || !jl["layer"].is_number_integer()) {
                report_parse_error(path.c_str(), "malformed layer entry (missing or non-integer 'layer')");
                return report;
            }
            const int32_t layer = jl["layer"].get<int32_t>();
            if (layer < 0) {
                report_parse_error(path.c_str(), "negative layer id");
                return report;
            }
            if (!jl.contains("experts") || !jl["experts"].is_array()) {
                report_parse_error(path.c_str(), "layer entry has no 'experts' array");
                return report;
            }

            const auto & experts = jl["experts"];
            if (experts.size() != (size_t) report.n_experts) {
                char msg[256];
                snprintf(msg, sizeof(msg), "header n_experts=%d but layer %d contains %zu experts",
                         report.n_experts, layer, experts.size());
                report_parse_error(path.c_str(), msg);
                return report;
            }

            for (size_t e = 0; e < experts.size(); e++) {
                const auto & je = experts[e];
                if (!je.is_object() || !je.contains("expert") || !je["expert"].is_number_integer()) {
                    report_parse_error(path.c_str(), "malformed expert entry (missing or non-integer 'expert')");
                    return report;
                }
                const int32_t expert = je["expert"].get<int32_t>();
                if (expert < 0) {
                    report_parse_error(path.c_str(), "negative expert id");
                    return report;
                }

                llama_moe_expert_stats s = {};
                s.layer_id = layer;
                s.expert_id = expert;
                s.prompt_selections = je.value("prompt_selections", (int64_t) 0);
                s.gen_selections    = je.value("gen_selections", (int64_t) 0);
                s.total_selections  = je.value("total_selections", (int64_t) 0);
                s.estimated_weight_bytes = 0;
                stats.push_back(s);
            }
        }
        // Only assign on full success: any error path above returns an empty
        // report so the caller can never apply a partially-parsed file.
        report.stats = std::move(stats);
    } catch (const nlohmann::json::exception & e) {
        report_parse_error(path.c_str(), e.what());
        return report;
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
