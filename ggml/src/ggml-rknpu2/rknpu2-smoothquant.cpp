#include "rknpu2-smoothquant.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {

// Mirrors Rknpu2DeviceConfig::resolve_op_support's tensor-name keying
// convention (rknpu2-configuration.cpp) so a tensor's smoothquant stats key
// always matches the key its pipeline resolution already uses -- including
// the ptr_<addr> fallback for an empty ggml_tensor::name.
std::string tensor_key(const struct ggml_tensor* t) {
    std::string name = t->name;
    if (name.empty()) {
        name = "ptr_" + std::to_string(reinterpret_cast<uintptr_t>(t));
    }
    return name;
}

constexpr float kSClampMin = 1.0f / 16.0f;
constexpr float kSClampMax = 16.0f;
constexpr float kWEps = 1e-6f;

float smooth_alpha() {
    static float a = -1.0f;
    if (a < 0.0f) {
        const char* e = std::getenv("GGML_RKNPU2_SMOOTH_ALPHA");
        a = (e != nullptr && e[0] != '\0') ? std::strtof(e, nullptr) : 0.5f;
        // Guard against a garbage/out-of-range env value rather than let a
        // bad alpha silently produce degenerate (all-0 or all-inf-before-
        // clamp) s_k for every tensor.
        if (!(a > 0.0f) || !(a < 1.0f)) a = 0.5f;
    }
    return a;
}

std::mutex g_calib_mutex;
std::unordered_map<std::string, std::vector<float>> g_calib_max_a; // name -> running max|A_k|

std::mutex g_smooth_mutex;
bool g_smooth_loaded = false;
std::unordered_map<std::string, std::vector<float>> g_smooth_max_a; // loaded from GGML_RKNPU2_SMOOTH's file
std::unordered_map<std::string, std::vector<float>> g_smooth_s;     // name -> cached, finalized s_k

// GGML_RKNPU2_CALIB's dump and GGML_RKNPU2_SMOOTH's load share ONE file
// format: one line per tensor, "<name> <K> <v0> <v1> ... <v(K-1)>",
// whitespace-separated. Must hold g_smooth_mutex.
void load_smooth_stats_locked(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr,
            "[RKNPU2_SMOOTH] warning: could not open stats file '%s' -- "
            "s_k=1.0 (no-op) for every tensor\n", path.c_str());
        return;
    }
    char name_buf[512];
    int k = 0;
    while (fscanf(f, "%511s %d", name_buf, &k) == 2 && k >= 0) {
        std::vector<float> vals((size_t)k);
        bool ok = true;
        for (int i = 0; i < k; ++i) {
            if (fscanf(f, "%f", &vals[i]) != 1) { ok = false; break; }
        }
        if (ok) {
            g_smooth_max_a[name_buf] = std::move(vals);
        } else {
            fprintf(stderr,
                "[RKNPU2_SMOOTH] warning: truncated stats line for '%s' "
                "(expected %d values) -- stopping load early, remaining "
                "tensors in '%s' get s_k=1.0 (no-op)\n", name_buf, k, path.c_str());
            break;
        }
    }
    fclose(f);
}

} // namespace

namespace rknpu2_smoothquant {

bool calib_enabled() {
    static int v = -1;
    if (v == -1) {
        const char* e = std::getenv("GGML_RKNPU2_CALIB");
        v = (e != nullptr && e[0] != '\0') ? 1 : 0;
    }
    return v == 1;
}

bool smooth_enabled() {
    static int v = -1;
    if (v == -1) {
        const char* e = std::getenv("GGML_RKNPU2_SMOOTH");
        v = (e != nullptr && e[0] != '\0') ? 1 : 0;
    }
    return v == 1;
}

void record_activation(const struct ggml_tensor* weight_tensor, int k_offset, const float* row, int n_elements) {
    if (!calib_enabled() || weight_tensor == nullptr || row == nullptr || n_elements <= 0) return;
    const std::string key = tensor_key(weight_tensor);

    std::lock_guard<std::mutex> lock(g_calib_mutex);
    std::vector<float>& acc = g_calib_max_a[key];
    const size_t needed = (size_t)k_offset + (size_t)n_elements;
    if (acc.size() < needed) acc.resize(needed, 0.0f);
    for (int i = 0; i < n_elements; ++i) {
        const float v = std::fabs(row[i]);
        const size_t k = (size_t)k_offset + (size_t)i;
        if (v > acc[k]) acc[k] = v;
    }
}

void dump_calibration() {
    if (!calib_enabled()) return;

    std::lock_guard<std::mutex> lock(g_calib_mutex);
    if (g_calib_max_a.empty()) return;

    const char* path = std::getenv("GGML_RKNPU2_CALIB");
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr,
            "[RKNPU2_CALIB] warning: could not open '%s' for write -- "
            "%zu tensors' recorded stats LOST\n", path, g_calib_max_a.size());
        return;
    }
    for (const auto& kv : g_calib_max_a) {
        fprintf(f, "%s %zu", kv.first.c_str(), kv.second.size());
        for (float v : kv.second) fprintf(f, " %.9g", v);
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "[RKNPU2_CALIB] wrote max|A_k| stats for %zu tensors to '%s'\n",
        g_calib_max_a.size(), path);
}

const std::vector<float>* compute_and_cache_s(const struct ggml_tensor* weight_tensor, const float* weight_col_absmax, int K) {
    if (!smooth_enabled() || weight_tensor == nullptr || weight_col_absmax == nullptr || K <= 0) return nullptr;
    const std::string key = tensor_key(weight_tensor);

    std::lock_guard<std::mutex> lock(g_smooth_mutex);
    if (!g_smooth_loaded) {
        const char* path = std::getenv("GGML_RKNPU2_SMOOTH");
        if (path != nullptr && path[0] != '\0') load_smooth_stats_locked(path);
        g_smooth_loaded = true;
    }

    auto cached = g_smooth_s.find(key);
    if (cached != g_smooth_s.end()) return &cached->second;

    std::vector<float> s((size_t)K, 1.0f);
    auto stats_it = g_smooth_max_a.find(key);
    if (stats_it != g_smooth_max_a.end()) {
        const std::vector<float>& max_a = stats_it->second;
        const float alpha = smooth_alpha();
        for (int k = 0; k < K; ++k) {
            const float a_k = ((size_t)k < max_a.size()) ? max_a[k] : 0.0f;
            const float w_k = std::max(weight_col_absmax[k], kWEps);
            // a_k==0 (dead activation channel) -> sk==0 pre-clamp, raised to
            // kSClampMin below. w_k is bounded away from 0 by kWEps above,
            // so this never divides by zero even for a dead weight column.
            const float sk = std::pow(a_k, alpha) / std::pow(w_k, 1.0f - alpha);
            s[k] = std::min(std::max(sk, kSClampMin), kSClampMax);
        }
    }
    // else: tensor missing from the recorded stats entirely -- leave s at
    // the all-ones no-op default (still cached, so a later set_tensor call
    // on the same tensor, e.g. a re-quantize, doesn't redo the miss lookup).

    auto& slot = g_smooth_s[key];
    slot = std::move(s);
    return &slot;
}

const std::vector<float>* lookup_s(const struct ggml_tensor* weight_tensor) {
    if (!smooth_enabled() || weight_tensor == nullptr) return nullptr;
    const std::string key = tensor_key(weight_tensor);

    std::lock_guard<std::mutex> lock(g_smooth_mutex);
    auto it = g_smooth_s.find(key);
    return (it != g_smooth_s.end()) ? &it->second : nullptr;
}

} // namespace rknpu2_smoothquant
