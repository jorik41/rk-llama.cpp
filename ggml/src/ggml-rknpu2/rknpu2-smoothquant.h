#pragma once

#include <string>
#include <vector>

struct ggml_tensor;

/**
 * @brief npu_fix12_smoothquant_20260902: SmoothQuant-style per-input-channel
 * activation/weight smoothing for the RKNPU2 W8A8 (NPU_TYPE_INT8 activation)
 * pipeline. See npu_fix12_smoothquant_plan_20260902.md for the numerical
 * design and npu_fix8_int8_scales_20260902.md / npu_fix9's outlier
 * diagnostic for why per-row weight scales alone (fix8) still leave some
 * MUL_MAT tiles NMSE-failing when a handful of input channels dominate the
 * activation range: this fix reshapes both sides of the quantization ahead
 * of fix8's existing per-row scale so no single channel dominates either
 * operand any more.
 *
 * Two independent, env-gated, opt-in modes -- both no-ops (byte-identical
 * to pre-fix12 behavior) unless their env var is set:
 *
 *  - GGML_RKNPU2_CALIB=<path>: recording mode. graph_compute's A-prep step
 *    accumulates a running per-input-channel max|A_k| for every MUL_MAT on
 *    an INT8-activation, non-Hadamard pipeline, keyed by the weight
 *    tensor's name (matching Rknpu2DeviceConfig::resolve_op_support's own
 *    name-with-ptr_<addr>-fallback keying convention, so the same tensor
 *    always resolves to the same key both places). ggml_backend_rknpu_free()
 *    dumps the accumulated stats to <path> on process exit as a trivially
 *    parseable text file: one line per tensor, `<name> <K> <v0> <v1> ...
 *    <v(K-1)>`, whitespace-separated (ggml_tensor names never contain
 *    whitespace).
 *
 *  - GGML_RKNPU2_SMOOTH=<path> (+ optional GGML_RKNPU2_SMOOTH_ALPHA, default
 *    0.5): apply mode. Lazily loads the max|A_k| stats file the first time
 *    any tensor's s_k is requested. ggml_backend_rknpu_buffer_set_tensor()
 *    computes and caches
 *        s_k = clamp(max|A_k|^alpha / max(max|W_k|, 1e-6)^(1-alpha), 1/16, 16)
 *    once per weight tensor, from the loaded max|A_k| stats and a
 *    max|W_k| it derives itself (a lightweight extra dequant pass over the
 *    tensor's segments -- see set_tensor's pre-pass, before it re-uses the
 *    exact same dequantize_tensor_segment() call in its real pass). The
 *    stats file this mode loads therefore holds raw max|A_k|, never
 *    precomputed s_k -- no offline "combine with weights" step is needed;
 *    the C++ side always has both operands' stats in hand at the point it
 *    needs them. s_k = 1.0 (no-op) for every k of a tensor missing from the
 *    stats file. graph_compute's A-prep divides the activation row by the
 *    same cached s_k (via lookup_s()) before its existing amax/INT8
 *    quantization -- set_tensor always runs first, at model-load time, well
 *    before any graph_compute call, so the cache is always populated by
 *    the time graph_compute looks it up.
 *
 * Both modes skip Hadamard and non-INT8-activation (FP16/INT4) pipelines
 * entirely (npu_type_a != NPU_TYPE_INT8, or use_hadamard) -- see the plan
 * doc sec 3 for why composing SmoothQuant with the Hadamard transform would
 * fight rather than compound, and fix8's own precedent for INT4 being out
 * of scope. Zero cost when neither env var is set: every call site gates on
 * a single cached-bool branch (calib_enabled()/smooth_enabled()) before
 * touching the stats maps.
 */
namespace rknpu2_smoothquant {

// True if GGML_RKNPU2_CALIB is set to a non-empty value -- recording mode
// active. Cached after the first call (env vars don't change mid-process).
bool calib_enabled();

// True if GGML_RKNPU2_SMOOTH is set to a non-empty value -- apply mode
// active. Cached after the first call.
bool smooth_enabled();

// Recording mode (GGML_RKNPU2_CALIB): accumulate this row's contribution to
// weight_tensor's running per-input-channel max|A_k|. k_offset is the
// global input-channel index of row[0]; row holds n_elements consecutive
// channels starting there (a MUL_MAT tile's K_seg_op-wide slice -- the
// accumulator grows lazily to the widest index seen, so callers never need
// to know the tensor's full K up front). Thread-safe (internal mutex). A
// no-op if calib_enabled() is false or weight_tensor/row is null.
void record_activation(const struct ggml_tensor* weight_tensor, int k_offset, const float* row, int n_elements);

// Recording mode: writes every tensor's accumulated max|A_k| stats to
// GGML_RKNPU2_CALIB's path. Call exactly once, from
// ggml_backend_rknpu_free(). A no-op if calib_enabled() is false or nothing
// was recorded.
void dump_calibration();

// Apply mode (GGML_RKNPU2_SMOOTH): first call for a given weight_tensor
// computes and caches its finalized s_k vector (length K) from the loaded
// max|A_k| stats (falling back to 1.0 per channel missing from the stats
// file, or for the whole tensor if it isn't in the file at all) and the
// caller-supplied weight_col_absmax (max|W_k| per column, length K, from a
// pre-pass over the tensor's own dequantized segments -- see set_tensor).
// A later call for the same tensor returns the cached vector unchanged
// (weight_col_absmax is only read on the first call). Returns nullptr if
// smooth_enabled() is false or weight_tensor/weight_col_absmax is null.
const std::vector<float>* compute_and_cache_s(const struct ggml_tensor* weight_tensor, const float* weight_col_absmax, int K);

// Apply mode: pure lookup of an already-cached s_k vector -- graph_compute
// never has weight_col_absmax in hand, so it can only ever look up what
// set_tensor already computed for this tensor at model-load time. Returns
// nullptr if smooth_enabled() is false, or the tensor has no cached s_k yet
// (fall back to s_k=1 for every k, i.e. do nothing).
const std::vector<float>* lookup_s(const struct ggml_tensor* weight_tensor);

} // namespace rknpu2_smoothquant
