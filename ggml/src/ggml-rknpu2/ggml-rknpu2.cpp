#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include <atomic>

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"
#include "rknpu2-smoothquant.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <omp.h>

#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <sys/mman.h>
#include <sys/resource.h>
#include <cerrno>
#include <sstream>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdint>

#define UNUSED(x) (void)(x)

// --- RKNPU2 debug instrumentation (Stage-4 glue hunt, env-gated) ---
// Enable with GGML_RKNPU2_DEBUG=1. Traces the buffer-integration glue:
// tensor_allocs (per-buffer, offset-keyed) and matmul_ctx_cache (per-backend,
// process-lifetime, keyed by addr+offset+shape) -- the two caches suspected
// of stale-binding on address reuse (see rk3576-stage4-glue-hunt-20260828).
static bool rknpu2_debug_enabled() {
    static int v = -1;
    if (v == -1) {
        const char* e = std::getenv("GGML_RKNPU2_DEBUG");
        v = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}
#define RKNPU2_DBG(...) do { if (rknpu2_debug_enabled()) { fprintf(stderr, "[RKNPU2_DBG] " __VA_ARGS__); fflush(stderr); } } while (0)

// npu_fix5b_keep_host_weights_20260902: test/diagnosis-only opt-in. When
// set, set_tensor() additionally stashes a verbatim copy of the host-side
// bytes it was called with for every NPU-quantized weight tensor, and
// get_tensor() returns that verbatim copy instead of reconstructing the
// weight from the chip-native quantized/Hadamard-transformed layout. Unset
// by default -- real serving never reads weights back, so the stash is
// simply never populated and this costs nothing.
static bool rknpu2_keep_host_weights_enabled() {
    static int v = -1;
    if (v == -1) {
        const char* e = std::getenv("GGML_RKNPU_KEEP_HOST_WEIGHTS");
        v = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}

static bool rknpu2_device_disabled() {
    static int v = -1;
    if (v == -1) {
        const char* dis = std::getenv("GGML_RKNPU_DISABLE");
        bool disabled = (dis != nullptr && dis[0] != '\0' && dis[0] != '0');
        if (!disabled) {
            const char* dev = std::getenv("RKNPU_DEVICE");
            if (dev != nullptr) {
                std::string s(dev);
                for (auto& c : s) c = (char)std::tolower((unsigned char)c);
                disabled = (s == "none" || s == "off" || s == "disable" || s == "disabled");
            }
        }
        v = disabled ? 1 : 0;
    }
    return v == 1;
}

// fix10-profile-20260902: per-stage host-side timing instrumentation for
// ggml_backend_rknpu_graph_compute(), gated behind GGML_RKNPU2_PROFILE=1.
// Unset (the default): rknpu2_profile_enabled() is one cached branch, and
// every timed block in graph_compute() is wrapped in `if (prof)`, so the
// only unconditional cost when disabled is that one already-cached bool
// check per block -- no std::chrono::steady_clock::now() call, no counter
// update, no allocation. See npu_fix10_nanopi_port_20260902.md sec on
// GGML_RKNPU2_PROFILE for the five stages measured (B-bind, A-prep, A
// set_io_mem+sync, rknn_matmul_run, C sync+descale) and the summary-table
// print points (one line per token to stderr, plus a lifetime summary at
// backend free).
static bool rknpu2_profile_enabled() {
    static int v = -1;
    if (v == -1) {
        const char* e = std::getenv("GGML_RKNPU2_PROFILE");
        v = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}

// rknpu2-broadcast-mulmat-20260902 (fix #6): minimum per-slice M (== the
// MUL_MAT's src1->ne[1], i.e. token/row count) below which we decline a
// *broadcast* MUL_MAT (GQA-style K/V-repeat: src0's ne[2]/ne[3] < src1's)
// rather than looping the NPU over each of the r2*r3 head/batch slices.
// Each slice is a separate rknn_matmul_run()+rknn_mem_sync() round trip;
// the B-matrix (src0 slice) bind is cached and reused across the r2/r3
// repeats of the same slice (see matmul_ctx_cache, keyed by
// address+offset -- broadcast reuse is "free" there), but the A-matrix
// upload, the run itself, and the C readback still happen once per
// slice. At decode time (M==1) that per-call dispatch overhead dominates
// the tiny actual compute and is expected to lose to ggml-cpu's fused
// attention kernel; at prefill-sized M the per-call NPU compute is large
// enough to amortize it. 32 is a first-draft threshold, not yet
// board-measured -- see npu_fix6_broadcast_20260902.md secs 4-5 for the
// reasoning and the bench-driven follow-up to tune (or make config-driven).
static constexpr int64_t RKNPU2_BROADCAST_MIN_M = 32;

// --- IOMMU Domain Manager ---

// Helper function for parsing complex integer lists
static std::vector<int32_t> parse_domain_list(const std::string& str) {
    std::vector<int32_t> result;
    if (str.empty()) return result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        auto dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
            int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
            for (int i = start; i <= end; ++i) result.push_back(i);
        } else {
            result.push_back(std::strtol(token.c_str(), nullptr, 10));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct IOMMUDomainManager {
    std::mutex mutex;

    // Max domain size for assigning
    const size_t max_domain_size = ((size_t) std::numeric_limits<int32_t>::max() - 65536);

    // Storage for domains and their sizes
    std::unordered_map<int32_t, size_t> domain_sizes;
    std::unordered_map<int32_t, rknn_matmul_ctx> allocator_contexts;

    // Allowed domain IDs defined by the user
    std::vector<int32_t> allowed_domains;

    IOMMUDomainManager() {
        // Read restricted domains from ENV variable
        const char* env_domains = std::getenv("RKNPU_DOMAINS");
        if (env_domains != nullptr) {
            allowed_domains = parse_domain_list(env_domains);

            if (!allowed_domains.empty()) {
                fprintf(stderr, "\n"
                    "RKNPU WARNING: Custom IOMMU domains detected via RKNPU_DOMAINS.\n"
                    "Due to Rockchip library limitations, concurrent execution of\n"
                    "multiple processes accessing the NPU simultaneously WILL LEAD\n"
                    "to a SYSTEM KERNEL PANIC and WILL FREEZE YOUR OPERATING SYSTEM.\n"
                    "Execute models SEQUENTIALLY if using multiple independent processes.\n");
            }
        }
    }

    // Function for assigning the domain for the tensor of given size
    int32_t assign_domain_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Allocate strictly within the allowed domains
        if (!allowed_domains.empty()) {
            for (int32_t d : allowed_domains) {
                if (domain_sizes[d] + size <= max_domain_size) {
                    // Check the allocator context BEFORE charging the domain
                    // size, so a failed context creation (see
                    // ensure_allocator_context) never hands back a domain id
                    // whose accounting says it holds memory it doesn't.
                    if (!ensure_allocator_context(d)) {
                        return -1;
                    }
                    domain_sizes[d] += size;
                    return d;
                }
            }

            fprintf(stderr, "RKNPU ERROR: Out of memory in allowed IOMMU domains!\n");
            return -1;
        // Allocate dynamically
        } else {
            for (int32_t i = 0; i <= 15; ++i) {
                if (domain_sizes[i] + size <= max_domain_size) {
                    if (!ensure_allocator_context(i)) {
                        return -1;
                    }
                    domain_sizes[i] += size;
                    return i;
                }
            }
            fprintf(stderr, "RKNPU ERROR: Out of memory in all IOMMU domains!\n");
            return -1;
        }
    }

    // Function for releasing the given size of the domain memory
    void release_domain_memory(int32_t domain_id, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = domain_sizes.find(domain_id);
        if (it != domain_sizes.end()) {
            if (it->second >= size) {
                it->second -= size;
            } else {
                it->second = 0;
            }
        }
    }

    // Function for getting a new dummy context in the required domain.
    // Returns 0 (an invalid rknn_context handle, matching this file's
    // existing "0 == unset/invalid" convention -- see rknpu_matmul_context's
    // ctx field) if the context could not be created -- see
    // ensure_allocator_context. Callers MUST check for 0 before passing the
    // result to any rknn_create_mem*/rknn_destroy_mem*/rknn_matmul_* call.
    rknn_matmul_ctx get_allocator_context(int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ensure_allocator_context(domain_id)) {
            return 0;
        }
        return allocator_contexts[domain_id];
    }

private:
    // Function for ensuring a dummy context existence in the required
    // domain. Returns true if a valid (or already-cached) allocator context
    // exists for domain_id, false if creation failed. On failure we
    // deliberately do NOT insert into allocator_contexts, so a later call
    // can retry -- the fd pressure causing the failure may be transient
    // (other allocations freeing fds in the meantime).
    bool ensure_allocator_context(int32_t domain_id) {
        if (allocator_contexts.find(domain_id) != allocator_contexts.end()) {
            return true;
        }

        rknn_matmul_info info;
        memset(&info, 0, sizeof(info));
        info.M = 32; info.K = 32; info.N = 32;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.iommu_domain_id = domain_id;

        rknn_matmul_io_attr io_attr;
        rknn_matmul_ctx ctx = 0;
        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret != RKNN_SUCC || ctx == 0) {
            fprintf(stderr,
                "RKNPU ERROR: rknn_matmul_create failed (ret=%d) creating the "
                "dummy allocator context for IOMMU domain %d -- a likely "
                "cause is RKNPU2 per-tensor dma-buf handle allocation "
                "exhausting the process file-descriptor limit (librknnrt "
                "reports 'failed to convert handle to fd, errno 24' in that "
                "case); raise it, e.g. `ulimit -n 65536`\n",
                ret, domain_id);
            return false;
        }
        allocator_contexts[domain_id] = ctx;
        return true;
    }
};
static IOMMUDomainManager g_domain_manager;

// --- Stage-5 A/C confession instrumentation (rk3576-stage5-20260828, env-gated) ---
// Captures the full dequantized FP32 weight matrix (N x K, row-major by n)
// for each B-matrix tensor as set_tensor() sees it, so graph_compute can
// independently recompute the reference C on the CPU and diff it against
// what the NPU actually produced -- isolating whether the bug is in the
// A-matrix prep, the C-matrix collection math, or upstream of both.
static std::mutex g_debug_weight_mutex;
static std::unordered_map<const struct ggml_tensor*, std::vector<float>> g_debug_weight_fp32;

// Macros for RKNN API calls. NOTE (rk3576-emfile-fix-window-20260828): this
// previously logged and then called assert(false) -- but this codebase's
// CMake default (Release, which sets -DNDEBUG) is what actually ships, and
// plain <cassert> assert() compiles to a no-op under NDEBUG. That made every
// one of these checks a SILENT NO-OP in the real build: on failure it
// printed one line and then FELL THROUGH, letting the caller run further
// RKNN calls (including rknn_matmul_run) against a context/buffer that was
// never actually bound -- a documented path to librknnrt crashing
// downstream. These now log (including the fd-limit hint, since dma-buf
// handle exhaustion is the most common cause of an RKNN call failing here)
// and actually fail the operation instead of silently continuing.
#define RKNN_LOG_FAILURE(ret, msg)                                            \
    fprintf(stderr,                                                          \
        "RKNN error %d at %s:%d: %s (a likely cause is RKNPU2 per-tensor "   \
        "dma-buf handle allocation exhausting the process file-descriptor "  \
        "limit -- librknnrt reports 'failed to convert handle to fd, errno " \
        "24' in that case; raise it, e.g. `ulimit -n 65536`)\n",             \
        ret, __FILE__, __LINE__, msg)

// For use inside ggml_backend_rknpu_graph_compute() (returns enum ggml_status).
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            RKNN_LOG_FAILURE(ret, msg);                                 \
            return GGML_STATUS_FAILED;                                  \
        }                                                               \
    } while (0)

// For use inside void-returning backend buffer callbacks (e.g. set_tensor),
// where there is no status to return -- the caller must still bail out
// cleanly rather than proceed with a call that failed.
#define RKNN_CHECK_VOID(stmt, msg)                                      \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            RKNN_LOG_FAILURE(ret, msg);                                 \
            return;                                                     \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information for N dimension
struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

// Matrix segment information for K dimension
struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

// Split B-matrix into N-segments for cores
static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();

    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);

    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

// Split B-matrix into K-segments for hardware limit
static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;

    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }

    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    void* virtual_base;
    size_t total_size;
    std::string name;

    // RKNN buffers allocations for each tensor
    struct TensorAllocation {
        rknn_tensor_mem* mem = nullptr;
        size_t size = 0;
        int32_t iommu_domain_id = 0;
    };
    std::unordered_map<size_t, TensorAllocation> tensor_allocs;

    // Per-block scaling factors for quantized weights
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    // npu_fix5b_keep_host_weights_20260902: verbatim copy of the host-side
    // bytes set_tensor() was called with for a "pipeline" (NPU-quantized)
    // weight tensor, keyed by tensor pointer. Only populated when
    // GGML_RKNPU_KEEP_HOST_WEIGHTS=1 (see rknpu2_keep_host_weights_enabled()
    // below) -- unset by default, so this costs real serving nothing.
    // Exists so a readback-sensitive caller (test-backend-ops MODE_TEST
    // building its CPU reference via get_tensor()) can compare the NPU's
    // compute output against the ORIGINAL un-requantized weights instead of
    // a dequant->inverse-Hadamard->ggml_quantize_chunk() reconstruction
    // (see npu_fix5_q4_0_nan_20260902.md / npu_fix5b_plan_20260902.md for
    // why that reconstruction is itself a second, independent quantization
    // step and inflates the measured error beyond genuine NPU compute
    // error).
    std::unordered_map<const struct ggml_tensor *, std::vector<uint8_t>> host_weight_bytes;

    std::mutex mutex;

    // Function for the allocation of a RKNN buffer for the individual tensor
    TensorAllocation get_tensor_allocation(size_t tensor_offset, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Trying to find an existing buffer
        auto it = tensor_allocs.find(tensor_offset);
        if (it != tensor_allocs.end()) {
            if (it->second.size < size) {
                RKNPU2_DBG("alloc RESIZE buf=%s offset=%zu old_size=%zu new_size=%zu old_fd=%d old_ptr=%p\n",
                    name.c_str(), tensor_offset, it->second.size, size, it->second.mem->fd, it->second.mem->virt_addr);
                rknn_matmul_ctx old_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                if (old_ctx != 0) {
                    rknn_destroy_mem(old_ctx, it->second.mem);
                } else {
                    fprintf(stderr,
                        "RKNPU2 WARNING: get_tensor_allocation RESIZE buf=%s "
                        "offset=%zu -- could not re-acquire allocator context "
                        "for old domain %d to free the old buffer (leaking it "
                        "instead of calling rknn_destroy_mem with an invalid "
                        "context) -- see prior RKNPU ERROR above\n",
                        name.c_str(), tensor_offset, it->second.iommu_domain_id);
                }
                g_domain_manager.release_domain_memory(it->second.iommu_domain_id, it->second.size);

                // rk3576-emfile-fix-window-20260828: this resize path used to
                // call rknn_create_mem() and dereference the result
                // unconditionally (it->second.mem->fd / ->virt_addr a few
                // lines below). Under fd-limit exhaustion (RKNPU2 allocates
                // one dma-buf handle/fd per tensor buffer; librknnrt reports
                // "failed to convert handle to fd, errno 24"), rknn_create_mem
                // returns NULL and that dereference SIGSEGVs. The old buffer
                // is already destroyed by this point (above), so on failure
                // here we cannot fall back to it -- drop this tensor's
                // allocation entirely and fail cleanly instead of crashing.
                int32_t new_domain_id = g_domain_manager.assign_domain_memory(size);
                rknn_matmul_ctx new_ctx = (new_domain_id >= 0) ? g_domain_manager.get_allocator_context(new_domain_id) : 0;
                rknn_tensor_mem* new_mem = (new_ctx != 0) ? rknn_create_mem(new_ctx, size) : nullptr;

                if (new_mem == nullptr) {
                    if (new_domain_id >= 0) {
                        g_domain_manager.release_domain_memory(new_domain_id, size);
                    }
                    fprintf(stderr,
                        "RKNPU2 ERROR: get_tensor_allocation RESIZE buf=%s "
                        "offset=%zu old_size=%zu new_size=%zu -- rknn_create_mem "
                        "failed for the resized buffer; a likely cause is "
                        "RKNPU2 per-tensor dma-buf handle allocation exhausting "
                        "the process file-descriptor limit (librknnrt reports "
                        "'failed to convert handle to fd, errno 24' in that "
                        "case); raise it, e.g. `ulimit -n 65536`. The old "
                        "buffer was already freed, so this tensor's allocation "
                        "is now dropped -- failing cleanly instead of "
                        "dereferencing a null buffer.\n",
                        name.c_str(), tensor_offset, it->second.size, size);
                    tensor_allocs.erase(it);
                    return TensorAllocation{};
                }

                it->second.iommu_domain_id = new_domain_id;
                it->second.mem = new_mem;
                it->second.size = size;
                RKNPU2_DBG("alloc RESIZE-DONE buf=%s offset=%zu new_fd=%d new_ptr=%p align64=%zu\n",
                    name.c_str(), tensor_offset, it->second.mem->fd, it->second.mem->virt_addr,
                    (size_t)((uintptr_t)it->second.mem->virt_addr % 64));
            } else {
                RKNPU2_DBG("alloc HIT buf=%s offset=%zu req_size=%zu cached_size=%zu fd=%d ptr=%p\n",
                    name.c_str(), tensor_offset, size, it->second.size, it->second.mem->fd, it->second.mem->virt_addr);
            }
            return it->second;
        }

        // Acquiring a domain for allocation. rk3576-emfile-fix-window-20260828:
        // this new-allocation path used to call rknn_create_mem() and only
        // guard the result with GGML_ASSERT (which aborts the whole process --
        // not the graceful "fail this op, keep the process alive" behavior we
        // want) and the domain/context acquisition above it was entirely
        // unchecked. Both are now checked and fail cleanly, returning a
        // sentinel TensorAllocation{} (mem=nullptr) that every caller must
        // (and now does) check before dereferencing.
        int32_t domain_id = g_domain_manager.assign_domain_memory(size);
        if (domain_id < 0) {
            fprintf(stderr,
                "RKNPU2 ERROR: get_tensor_allocation buf=%s offset=%zu "
                "size=%zu -- no IOMMU domain available (see prior RKNPU "
                "ERROR above); failing this tensor allocation cleanly "
                "instead of dereferencing a null buffer\n",
                name.c_str(), tensor_offset, size);
            return TensorAllocation{};
        }

        rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(domain_id);
        if (alloc_ctx == 0) {
            g_domain_manager.release_domain_memory(domain_id, size);
            fprintf(stderr,
                "RKNPU2 ERROR: get_tensor_allocation buf=%s offset=%zu "
                "size=%zu domain=%d -- no allocator context available (see "
                "prior RKNPU ERROR above); failing this tensor allocation "
                "cleanly instead of dereferencing a null buffer\n",
                name.c_str(), tensor_offset, size, domain_id);
            return TensorAllocation{};
        }

        // Allocating a new buffer for the tensor
        rknn_tensor_mem* mem = rknn_create_mem(alloc_ctx, size);
        if (mem == nullptr) {
            g_domain_manager.release_domain_memory(domain_id, size);
            fprintf(stderr,
                "RKNPU2 ERROR: rknn_create_mem failed buf=%s offset=%zu "
                "size=%zu domain=%d -- a likely cause is RKNPU2 per-tensor "
                "dma-buf handle allocation exhausting the process "
                "file-descriptor limit (librknnrt reports 'failed to convert "
                "handle to fd, errno 24' in that case); raise it, e.g. "
                "`ulimit -n 65536`. Failing this tensor allocation cleanly "
                "instead of dereferencing a null buffer.\n",
                name.c_str(), tensor_offset, size, domain_id);
            return TensorAllocation{};
        }

        TensorAllocation alloc;
        alloc.mem = mem;
        alloc.size = size;
        alloc.iommu_domain_id = domain_id;

        RKNPU2_DBG("alloc MISS-NEW buf=%s offset=%zu size=%zu domain=%d fd=%d ptr=%p align64=%zu\n",
            name.c_str(), tensor_offset, size, domain_id, alloc.mem->fd, alloc.mem->virt_addr,
            (size_t)((uintptr_t)alloc.mem->virt_addr % 64));
        tensor_allocs[tensor_offset] = alloc;

        return alloc;
    }
};


// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    bool b_bound = false;
    std::shared_ptr<rknn_tensor_mem> mem_B;

    // fix10-host-roundtrip-20260902: identity of the last rknn_tensor_mem
    // bound to this context's A io slot via rknn_matmul_set_io_mem. The
    // A-buffer handed in each graph_compute() call comes from
    // backend_ctx->a_buffer_cache, keyed by (M_op, K_seg_op, npu_type_a,
    // domain) -- for a steady decode loop (stable M/K/type/domain across
    // tokens) get_tensor_buffer() returns the SAME rknn_tensor_mem object
    // every token, yet the call site unconditionally re-issued
    // rknn_matmul_set_io_mem(A) every token for every active N segment.
    // mem_B already avoids this exact redundant rebind via b_bound; mirror
    // it here for A. Raw pointer only (identity check) -- ownership stays
    // with the shared_ptr in a_buffer_cache / mem_A_shared.
    rknn_tensor_mem* a_bound_mem = nullptr;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type, int32_t domain_id) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        info.iommu_domain_id = domain_id;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        mem_B.reset();

        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};

// Global pointer to the single (process-lifetime) RKNPU backend instance,
// set by ggml_backend_rknpu_device_init_backend. Used by
// ggml_backend_rknpu_buffer_free_buffer to invalidate matmul_ctx_cache
// entries when their backing DMA allocation is freed -- see
// invalidate_matmul_ctx_for_address (Stage-4 fix,
// rk3576-stage4-glue-hunt-20260828).
static struct ggml_backend_rknpu_context* g_active_rknpu_backend_ctx = nullptr;

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (tensor_fd, offset, M, K, N, core_id, type, domain_id)
    std::unordered_map<std::tuple<uintptr_t, size_t, int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // A-matrices cache (M, K, npu_type_a, domain_id)
    std::unordered_map<std::tuple<int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;

    // C-matrices cache (M, N, core_id, npu_type_c, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    // fix10-profile-20260902: lifetime accumulators for GGML_RKNPU2_PROFILE=1
    // (stage names match the per-token stderr line graph_compute() prints).
    // graph_compute() is called serially (never re-entered concurrently for
    // one backend instance), so plain counters are sufficient -- no atomics,
    // no locking. Only ever written/read when rknpu2_profile_enabled().
    uint64_t prof_b_bind_ns = 0;
    uint64_t prof_a_prep_ns = 0;
    uint64_t prof_a_bind_ns = 0;
    uint64_t prof_run_ns = 0;
    uint64_t prof_c_ns = 0;
    uint64_t prof_n_matmul = 0;
    uint64_t prof_n_tokens = 0;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(uintptr_t tensor_id, size_t offset, int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(tensor_id, offset, M, K, N, core_id, (int)type, (int)domain_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            RKNPU2_DBG("matmul_ctx HIT addr=0x%lx off=%zu M=%d K=%d N=%d core=%d type=%d dom=%d b_bound=%d mem_B_ptr=%p ctx_obj=%p\n",
                (unsigned long)tensor_id, offset, M, K, N, core_id, (int)type, domain_id,
                it->second->b_bound ? 1 : 0,
                (void*)(it->second->mem_B ? it->second->mem_B->virt_addr : nullptr),
                (void*)it->second.get());
            return it->second;
        }
        RKNPU2_DBG("matmul_ctx MISS-NEW addr=0x%lx off=%zu M=%d K=%d N=%d core=%d type=%d dom=%d\n",
            (unsigned long)tensor_id, offset, M, K, N, core_id, (int)type, domain_id);

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        matmul_ctx_cache[key] = ctx;
        return ctx;
    }

    // Stage-4 fix (rk3576-stage4-glue-hunt-20260828): matmul_ctx_cache is
    // keyed by (address, offset, shape...) with no tensor identity. When the
    // tensor_allocs slot backing that address is freed (buffer teardown) and
    // the address is later reused for a DIFFERENT logical tensor with the
    // same shape/type/domain, a stale cache entry -- possibly already
    // b_bound=true -- would silently serve the new tensor's matmul with the
    // OLD tensor's binding (confirmed on hardware: window-2 trace, M=3's
    // bind reused for M=4 at the same reallocated DMA address). Evicting
    // every entry whose key address matches a just-freed allocation restores
    // the invariant that a cache hit only ever returns a binding made for
    // memory that is still validly backing the same logical tensor.
    void invalidate_matmul_ctx_for_address(void* addr) {
        std::lock_guard<std::mutex> lock(mutex);
        uintptr_t target = (uintptr_t)addr;
        for (auto it = matmul_ctx_cache.begin(); it != matmul_ctx_cache.end(); ) {
            if (std::get<0>(it->first) == target) {
                RKNPU2_DBG("matmul_ctx INVALIDATE addr=%p (backing buffer freed) evicting ctx=%p b_bound_was=%d\n",
                    addr, (void*)it->second.get(), it->second->b_bound ? 1 : 0);
                it = matmul_ctx_cache.erase(it);
            } else {
                ++it;
            }
        }
    }
};


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

// fix10-profile-20260902: prints the lifetime GGML_RKNPU2_PROFILE summary
// table for one backend instance to stderr. Called from
// ggml_backend_rknpu_free() (process/backend teardown) so a normal run
// always gets one final table even if nothing else polls it; graph_compute()
// separately prints one raw per-token line every call (see its own comment)
// for finer-grained inspection without waiting for teardown.
static void rknpu2_print_profile_summary(const ggml_backend_rknpu_context* ctx) {
    if (ctx->prof_n_tokens == 0) return;
    const double to_us = 1e-3;
    fprintf(stderr,
        "[RKNPU2_PROFILE] ==== summary: tokens=%llu n_matmul=%llu ====\n"
        "[RKNPU2_PROFILE] stage        total_us      avg_us/token\n"
        "[RKNPU2_PROFILE] b_bind    %14.1f  %14.3f\n"
        "[RKNPU2_PROFILE] a_prep    %14.1f  %14.3f\n"
        "[RKNPU2_PROFILE] a_bind    %14.1f  %14.3f\n"
        "[RKNPU2_PROFILE] run       %14.1f  %14.3f\n"
        "[RKNPU2_PROFILE] c_stage   %14.1f  %14.3f\n",
        (unsigned long long)ctx->prof_n_tokens, (unsigned long long)ctx->prof_n_matmul,
        ctx->prof_b_bind_ns * to_us, ctx->prof_b_bind_ns * to_us / (double)ctx->prof_n_tokens,
        ctx->prof_a_prep_ns * to_us, ctx->prof_a_prep_ns * to_us / (double)ctx->prof_n_tokens,
        ctx->prof_a_bind_ns * to_us, ctx->prof_a_bind_ns * to_us / (double)ctx->prof_n_tokens,
        ctx->prof_run_ns    * to_us, ctx->prof_run_ns    * to_us / (double)ctx->prof_n_tokens,
        ctx->prof_c_ns      * to_us, ctx->prof_c_ns      * to_us / (double)ctx->prof_n_tokens);
    fflush(stderr);
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    // npu_fix12_smoothquant_20260902: flush any GGML_RKNPU2_CALIB-recorded
    // per-input-channel max|A_k| stats to disk exactly once, at backend
    // teardown. A no-op (single cached-bool branch) when GGML_RKNPU2_CALIB
    // is unset or nothing was recorded.
    rknpu2_smoothquant::dump_calibration();

    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    if (rknpu2_profile_enabled()) {
        rknpu2_print_profile_summary(ctx);
    }
    // Stage-2.5 hardening (defensive, separate hazard from the reorder
    // above): clear the dangling process-global before delete so no later
    // reader (e.g. a -j>1 / multi-init caller) can dereference a freed ctx.
    if (g_active_rknpu_backend_ctx == ctx) {
        g_active_rknpu_backend_ctx = nullptr;
    }
    delete ctx;
    delete backend;
}

// Function for acquiring a pointer for tensor data
static void* get_tensor_real_ptr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nullptr;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        auto* ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_allocs.find(offset);
        // it->second.mem can be nullptr if a prior allocation for this
        // tensor failed (rk3576-emfile-fix-window-20260828, e.g. fd-limit
        // exhaustion) -- fall through to the raw tensor->data pointer below
        // rather than dereferencing a null buffer.
        if (it != ctx->tensor_allocs.end() && it->second.mem != nullptr) {
            return it->second.mem->virt_addr;
        }
    }

    return tensor->data;
}

// Function for getting buffer from cache or creating new one
//
// a_buffer_cache / c_buffer_cache entries outlive any single matmul_ctx:
// they are keyed by (M, K/N, type, domain) -- not by which rknpu_matmul_context
// created them -- and matmul_ctx_cache entries can be erased mid-run by
// invalidate_matmul_ctx_for_address() (buffer teardown, Stage-4 fix), which
// drops the owning rknpu_matmul_context and, once its refcount hits zero,
// calls rknn_matmul_destroy(ctx) in ~rknpu_matmul_context. Previously this
// function captured only the raw rknn_matmul_ctx handle (by value) in the
// mem's deleter; if the owning rknpu_matmul_context was destroyed while a
// cached buffer created against it was still live, that captured handle
// went stale, and calling rknn_destroy_mem() on it later (a subsequent
// eviction, or backend teardown -- see ~ggml_backend_rknpu_context) read
// through the already-freed librknnrt context and crashed with a SEGV deep
// inside rknn_destroy_mem (rk3588-fix2-teardown-destroy-order-20260902,
// ASAN: SEGV on unknown address 0x1783 in rknn_destroy_mem, called from the
// shared_ptr<_rknn_tensor_memory> deleter tearing down a_buffer_cache /
// c_buffer_cache). Taking (and capturing) the owning rknpu_matmul_context by
// shared_ptr instead of its raw ctx handle keeps that context alive for as
// long as any buffer created against it is cached -- so rknn_destroy_mem
// always runs against a still-live ctx, and only after it returns can the
// captured shared_ptr's refcount drop, letting rknn_matmul_destroy() run.
// This fixes both the mid-run-invalidation case and backend teardown,
// regardless of unordered_map/member destruction order.
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    const std::shared_ptr<rknpu_matmul_context>& matmul_ctx_owner,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->size >= size) {
            return it->second;
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx_owner->ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx_owner](rknn_tensor_mem* m) {
        if (m != 0) {
            rknn_destroy_mem(matmul_ctx_owner->ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    // fix10-profile-20260902: GGML_RKNPU2_PROFILE=1 per-stage timing, zero
    // cost when unset -- `prof` is a single cached-bool check, and every
    // timed block below is wrapped `if (prof) { ...steady_clock::now()... }`,
    // so with the env var unset no clock is ever read and no counter is
    // ever touched. Accumulated locally across this one graph_compute() call
    // (one token's worth of MUL_MAT nodes x K-segments), folded into
    // backend_ctx's lifetime totals and printed as one raw line at the end
    // of this call; ggml_backend_rknpu_free() prints the lifetime summary
    // table built from those same backend_ctx accumulators.
    const bool prof = rknpu2_profile_enabled();
    uint64_t prof_b_bind_ns = 0, prof_a_prep_ns = 0, prof_a_bind_ns = 0, prof_run_ns = 0, prof_c_ns = 0;
    uint64_t prof_n_matmul_this_call = 0;

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        // rknpu2-broadcast-mulmat-20260902 (fix #6, mirrors the check added
        // to ggml_backend_rknpu_device_supports_op() below -- see its
        // comment for the full rationale). This re-derives the same
        // has_batch / integer-ratio / M-threshold decision supports_op
        // already made, in case the two gates disagree (see
        // MASTERPORT_FIX_REDIAGNOSIS_20260828.md sec 0/2, where exactly
        // that divergence was the root cause of a prior regression) --
        // trust nothing from the caller. `continue` leaves this node
        // unexecuted, matching every other decline path in this loop
        // (zero-dimension, missing pipeline, empty segments); unlike
        // those, a decline HERE after supports_op already accepted the
        // node would silently leave dst un-written (ggml already routed
        // this node to this backend), so this check's accept/decline set
        // must stay in lockstep with supports_op's -- both are re-checked
        // against the same RKNPU2_BROADCAST_MIN_M and pipeline->npu_type_b
        // conditions below, after `pipeline` is resolved.
        const bool has_batch = (src0->ne[2] != 1 || src0->ne[3] != 1 ||
                                 src1->ne[2] != 1 || src1->ne[3] != 1 ||
                                 dst->ne[2]  != 1 || dst->ne[3]  != 1);
        if (has_batch) {
            if (dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3] ||
                src0->ne[2] > src1->ne[2] || src0->ne[3] > src1->ne[3] ||
                src1->ne[2] % src0->ne[2] != 0 ||
                src1->ne[3] % src0->ne[3] != 0 ||
                src1->ne[1] < RKNPU2_BROADCAST_MIN_M) {
                continue;
            }
        }
        // Broadcast loop bounds: for a plain 2D MUL_MAT these are all 1,
        // so the (i3,i2) loop added below runs its body exactly once --
        // byte-for-byte the pre-fix-6 control flow. r2/r3 are GGML's own
        // MUL_MAT broadcast ratios (mirrors ggml-cpu's
        // ggml_compute_forward_mul_mat: r2 = ne12/ne02, r3 = ne13/ne03);
        // i02 = i2/r2, i03 = i3/r3 pick which src0 slice a given
        // src1/dst slice broadcasts against.
        const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
        const int64_t r2 = ne12 / src0->ne[2], r3 = ne13 / src0->ne[3];

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        // Skipping zero-dimension matmuls
        if (M == 0 || K == 0 || N == 0) {
            continue;
        }

        // Using next power of two for M for efficient caching
        int M_op = M;
        if (M > 1) {
            M_op = rknpu2_calibration::next_power_of_two(M);
        }

        const auto* pipeline = config.resolve_op_support(src0);
        if (!pipeline) continue;

        // rknpu2-broadcast-mulmat-20260902 (fix #6): mirrors the identical
        // FP16-only / non-Hadamard scope-limit gate in
        // ggml_backend_rknpu_device_supports_op() below -- see its comment
        // for the full rationale (INT8/INT4 per-slice dequant-scale
        // indexing is not yet extended for a batched src0). Must stay in
        // lockstep with that gate's decision (see the note on has_batch
        // above) -- if this ever disagreed with supports_op, a node
        // supports_op accepted would silently reach here and get declined,
        // leaving dst un-written instead of computed.
        if (has_batch && (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16 || pipeline->use_hadamard)) {
            continue;
        }

        // Initializing Hadamard Transform Logic
        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;

        // Computing specific hardware segments
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto all_k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);

        std::vector<MatrixSegmentN> active_n_segments;
        for (const auto& seg : all_n_segments) {
            if (seg.size_n > 0) active_n_segments.push_back(seg);
        }

        if (active_n_segments.empty()) continue;

        // rknpu2-broadcast-mulmat-20260902 (fix #6): the per-(K,N)-slice
        // packed size, i.e. what get_tensor_packed_size() (this file)
        // computes for ONE slice before its own ne[2]*ne[3] multiply.
        // FP16-only per the scope gate above (type_size_packed==2); needed
        // to locate each src0 slice's own region within the buffer
        // ggml_backend_rknpu_buffer_set_tensor()'s matching per-slice loop
        // packed back-to-back (i3-major, i2-minor over src0's OWN
        // ne[2]/ne[3] -- see slice_linear_index below).
        size_t per_slice_packed_size = 0;
        for (const auto& k_seg : all_k_segments) {
            for (const auto& n_seg : active_n_segments) {
                per_slice_packed_size += (size_t)n_seg.size_n * k_seg.size_k * 2;
            }
        }

        // Cleaning the whole C-matrix (dst) buffer once, up front -- the
        // (i3,i2) loop below writes disjoint M*N slices of it, each zeroed
        // exactly once here rather than per-slice.
        uint8_t* dst_data_base = (uint8_t*)get_tensor_real_ptr(dst);
        memset(dst_data_base, 0, ggml_nbytes(dst));

        // rknpu2-broadcast-mulmat-20260902 (fix #6): one NPU matmul per
        // (i3,i2) slice of src1/dst's batch shape. For a plain 2D MUL_MAT
        // ne12==ne13==1 so this runs its body exactly once with i2=i3=0,
        // identical to the pre-fix-6 control flow. i02/i03 pick which
        // (smaller, or equal) src0 slice this src1/dst slice broadcasts
        // against; slice_linear_index addresses that slice's packed
        // region within the shared B-matrix buffer (see
        // per_slice_packed_size above).
        for (int64_t i3 = 0; i3 < ne13; ++i3) {
        for (int64_t i2 = 0; i2 < ne12; ++i2) {
        const int64_t i02 = i2 / r2;
        const int64_t i03 = i3 / r3;
        const int64_t slice_linear_index = i03 * src0->ne[2] + i02;

        // Initializing variables
        const size_t num_active_segments = active_n_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // Acquiring the B-matrix buffer. NOTE: this tensor_allocs lookup is
        // keyed by src0's own (whole-tensor) offset -- unaffected by which
        // (i02,i03) slice we're on -- so it is invariant across the i3/i2
        // loop and, strictly, could be hoisted above it; left inside for a
        // smaller first-draft diff (one extra mutex-guarded map lookup per
        // slice, cheap relative to the matmul itself).
        ggml_backend_buffer_t src0_buffer = src0->buffer;
        auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
        size_t tensor_offset_in_virtual = (uintptr_t)src0->data - (uintptr_t)src0_buf_ctx->virtual_base;

        int32_t b_domain_id = 0;
        int tensor_fd = -1;
        void* tensor_virt_addr = nullptr;
        {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->tensor_allocs.find(tensor_offset_in_virtual);
            // rk3576-emfile-fix-window-20260828: this used to be a
            // GGML_ASSERT that only checked presence in the map, then
            // unconditionally dereferenced it->second.mem below. A tensor
            // whose allocation failed (e.g. fd-limit exhaustion) is either
            // absent from the map or present with mem==nullptr depending on
            // which allocation path failed (see get_tensor_allocation) --
            // check both and fail this op with an error status instead of
            // aborting the whole process or dereferencing a null buffer.
            if (it == src0_buf_ctx->tensor_allocs.end() || it->second.mem == nullptr) {
                fprintf(stderr,
                    "RKNPU2 ERROR: graph_compute tensor=%s -- B-matrix RKNN "
                    "buffer missing or failed to allocate (a likely cause is "
                    "the process file-descriptor limit; see prior RKNPU2 "
                    "ERROR, and try `ulimit -n 65536`); failing this op "
                    "cleanly instead of dereferencing a null buffer\n",
                    src0->name);
                return GGML_STATUS_FAILED;
            }

            tensor_fd = it->second.mem->fd;
            tensor_virt_addr = it->second.mem->virt_addr;
            b_domain_id = it->second.iommu_domain_id;
        }
        if (rknpu2_debug_enabled()) {
            uint8_t* b0 = (uint8_t*)tensor_virt_addr;
            RKNPU2_DBG("graph_compute B-src tensor=%p name=%s buf=%p offset=%zu fd=%d ptr=%p M=%d M_op=%d K=%d N=%d i2=%lld i3=%lld i02=%lld i03=%lld first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                (const void*)src0, src0->name, (void*)src0_buffer, tensor_offset_in_virtual, tensor_fd, tensor_virt_addr, M, M_op, K, N,
                (long long)i2, (long long)i3, (long long)i02, (long long)i03,
                b0[0], b0[1], b0[2], b0[3], b0[4], b0[5], b0[6], b0[7]);
        }

        // This slice's region of the C-matrix (dst) buffer -- already
        // zeroed in bulk above.
        float* dst_data = (float*)(dst_data_base + i2 * dst->nb[2] + i3 * dst->nb[3]);

        // Acquiring the Hadamard vector
        std::vector<float> s_vec;
        if (is_hadamard) {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->hadamard_s_vectors.find(src0);
            GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
            s_vec = it->second;
        }

        // Calculating the B-matrix scale
        std::vector<float> scales_B_grid;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->quantized_tensor_scales.find(src0);
            GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
            scales_B_grid = it->second;
        }

        // Calculating tensor packed size
        size_t type_size_packed = 0;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;

        // fix10-profile-20260902: this node has cleared every decline gate
        // above and is genuinely dispatched to the NPU -- count it once per
        // node (not per K-segment) as this call's matmul-node count.
        if (prof) ++prof_n_matmul_this_call;

        // Computing K dimensions segments. rknpu2-broadcast-mulmat-20260902
        // (fix #6): starts at this slice's own base offset within the
        // shared packed buffer (0 for the common non-batch case, where
        // slice_linear_index is always 0) instead of always 0.
        size_t current_offset_in_tensor = slice_linear_index * per_slice_packed_size;
        for (size_t k_idx = 0; k_idx < all_k_segments.size(); ++k_idx) {
            const auto& k_seg = all_k_segments[k_idx];
            const int K_seg_op = k_seg.size_k;

            // ===========================================
            // ========== 1. Preparing Contexts ==========
            // ===========================================
            std::chrono::steady_clock::time_point prof_t_bbind_start;
            if (prof) prof_t_bbind_start = std::chrono::steady_clock::now();
            for (const auto& n_seg : all_n_segments) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    if (active_n_segments[idx].offset_n == n_seg.offset_n) {
                        size_t offset_in_dma = current_offset_in_tensor;

                        // Getting matmul context from cache
                        matmul_ctxs[idx] = backend_ctx->get_matmul_ctx(
                            (uintptr_t)tensor_virt_addr, offset_in_dma, M_op, K_seg_op, n_seg.size_n,
                            n_seg.core_id, matmul_type, b_domain_id
                        );
                        if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;

                        auto& matmul_ctx = matmul_ctxs[idx];

                        // Assigning B-matrix only once to reduce computation overhead
                        if (!matmul_ctx->b_bound) {
                            size_t segment_size_bytes = matmul_ctx->io_attr.B.size;
                            RKNPU2_DBG("B-BIND tensor=%s addr=%p off_in_dma=%zu seg_bytes=%zu matmul_ctx=%p\n",
                                src0->name, tensor_virt_addr, offset_in_dma, segment_size_bytes, (void*)matmul_ctx.get());

                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(
                                matmul_ctx->ctx,
                                tensor_fd,
                                tensor_virt_addr,
                                segment_size_bytes,
                                offset_in_dma
                            );
                            if (!mem) return GGML_STATUS_FAILED;
                            if (rknpu2_debug_enabled()) {
                                uint8_t* srcb = (uint8_t*)tensor_virt_addr + offset_in_dma;
                                uint8_t* impb = (uint8_t*)mem->virt_addr;
                                RKNPU2_DBG("B-IMPORTED mem=%p imp_fd=%d imp_ptr=%p imp_size=%u SRC(passed-in) src_fd=%d src_ptr=%p src_off=%zu SAME_PTR=%d src_first8=%02x%02x%02x%02x%02x%02x%02x%02x imp_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                                    (void*)mem, mem->fd, mem->virt_addr, mem->size,
                                    tensor_fd, tensor_virt_addr, offset_in_dma,
                                    (mem->virt_addr == (void*)((uint8_t*)tensor_virt_addr + offset_in_dma)) ? 1 : 0,
                                    srcb[0], srcb[1], srcb[2], srcb[3], srcb[4], srcb[5], srcb[6], srcb[7],
                                    impb[0], impb[1], impb[2], impb[3], impb[4], impb[5], impb[6], impb[7]);
                            }

                            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            matmul_ctx->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);

                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");

                            // Stage-4 fix (rk3576-stage4-glue-hunt-20260828): the A-matrix
                            // path a few lines below explicitly calls rknn_mem_sync(...,
                            // TO_DEVICE) after writing+binding; this B-matrix path never did,
                            // even though it imports a SEPARATE rknn_tensor_mem handle (via
                            // rknn_create_mem_from_fd, bound to this specific matmul_ctx->ctx)
                            // distinct from the handle buffer_set_tensor already synced
                            // (alloc.mem, under a dummy allocator context). Confirmed on
                            // hardware (window-2 trace): even a completely fresh, first-ever
                            // bind (no cache staleness possible) produced garbage output;
                            // adding the matching sync call here fixes it.
                            RKNN_CHECK(rknn_mem_sync(matmul_ctx->ctx, matmul_ctx->mem_B.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE (matmul_ctx)");

                            matmul_ctx->b_bound = true;
                        } else if (rknpu2_debug_enabled()) {
                            uint8_t* cb = (uint8_t*)tensor_virt_addr + offset_in_dma;
                            RKNPU2_DBG("B-SKIP(cached-bind-reused) tensor=%s addr=%p off_in_dma=%zu matmul_ctx=%p live_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                                src0->name, tensor_virt_addr, offset_in_dma, (void*)matmul_ctx.get(),
                                cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]);
                        }
                        break;
                    }
                }

                if (n_seg.size_n > 0) {
                    current_offset_in_tensor += type_size_packed > 0 ? (size_t)n_seg.size_n * K_seg_op * type_size_packed : (size_t)n_seg.size_n * K_seg_op / 2;
                }
            }
            if (prof) prof_b_bind_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prof_t_bbind_start).count();

            // ===========================================
            // ========== 2. Preparing A-matrix ==========
            // ===========================================
            std::vector<float> scales_A(M, 1.0f);
            {
                auto cache_key = std::make_tuple(M_op, K_seg_op, (int)pipeline->npu_type_a, b_domain_id);
                auto& matmul_ctx_0 = matmul_ctxs[0];

                // Getting A-buffer from cache
                mem_A_shared = get_tensor_buffer(backend_ctx, matmul_ctx_0, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
                if (!mem_A_shared) return GGML_STATUS_FAILED;

                // rknpu2-broadcast-mulmat-20260902 (fix #6): this slice's
                // region of src1 -- get_tensor_real_ptr(src1) is the base
                // (src1 is never a "pipeline" tensor here -- see the
                // is_contiguous() check in supports_op -- so this is a
                // plain nb[2]/nb[3]-strided byte offset into src1's own
                // ggml-format buffer, not a packed-buffer lookup).
                const float* x = (const float*)((const uint8_t*)get_tensor_real_ptr(src1) + i2 * src1->nb[2] + i3 * src1->nb[3]);
                const int row_stride = (int)(src1->nb[1] / sizeof(float));
                void* dst_base = mem_A_shared->virt_addr;

                std::chrono::steady_clock::time_point prof_t_aprep_start;
                if (prof) prof_t_aprep_start = std::chrono::steady_clock::now();
                #pragma omp parallel for
                for (int m = 0; m < M; ++m) {
                    const float* src_row = x + (size_t)m * row_stride;

                    // fix10-host-roundtrip-20260902: the non-Hadamard path
                    // (the common case -- only pipelines with use_hadamard
                    // set take the other branch) used to memcpy K_seg_op
                    // floats into a freshly heap-allocated `ready_row`
                    // per m-iteration purely so the conversion calls below
                    // had a contiguous pointer -- but src_row + offset_k is
                    // already contiguous (supports_op requires
                    // ggml_is_contiguous(src1)), so the copy and the
                    // allocation both existed only to feed a pointer the
                    // source already provided. Point `ready_ptr` straight at
                    // it and skip both. The Hadamard path still needs a real
                    // transform output buffer; give it thread-local storage
                    // (thread_local, not a fresh heap vector every m) so it
                    // is reused across iterations on each OpenMP thread
                    // instead of allocating/freeing per row.
                    const float* ready_ptr;
                    if (is_hadamard) {
                        static thread_local std::vector<float> tls_signed_row;
                        static thread_local std::vector<float> tls_full_hadamard_row;
                        tls_signed_row.resize(K);
                        tls_full_hadamard_row.resize(K_op);
                        for(int k=0; k<K; ++k) tls_signed_row[k] = src_row[k] * s_vec[k];
                        rknpu2_calibration::hadamard_transform(tls_full_hadamard_row.data(), tls_signed_row.data(), K, K_op);

                        ready_ptr = tls_full_hadamard_row.data() + k_seg.offset_k;
                    } else {
                        ready_ptr = src_row + k_seg.offset_k;
                    }

                    // npu_fix12_smoothquant_20260902: two independent,
                    // env-gated, opt-in hooks around the INT8-activation
                    // amax/quantize step below -- zero cost (single
                    // cached-bool branch each) unless GGML_RKNPU2_CALIB or
                    // GGML_RKNPU2_SMOOTH is set. Both only apply to the
                    // INT8-activation, non-Hadamard pipelines this fix
                    // targets (see rknpu2-smoothquant.h).
                    //
                    // fix10-rebase-repair-20260903: fix10's copy-elision
                    // above deleted the `ready_row` buffer this block was
                    // originally written against (see fix6/fix12) and left
                    // only the read-only `ready_ptr` alias into src1's own
                    // storage (or the shared thread_local Hadamard buffer).
                    // record_activation() only reads the row, so it takes
                    // `ready_ptr` directly (it already accepts a bare
                    // `const float*`). The smoothing divide below mutates
                    // the row in place, which cannot go through `ready_ptr`
                    // (const, and would corrupt src1 / the shared Hadamard
                    // buffer) -- so it needs a real owned copy. `ready_row`
                    // is declared here, empty, and only actually allocated
                    // and populated on the one path that mutates
                    // (smooth_enabled() with a cached s_k for src0);
                    // `ready_ptr` is then repointed at it so the
                    // FP16/INT8/INT4 conversion below picks up the smoothed
                    // values. The common case (SmoothQuant off) still pays
                    // no copy, preserving fix10's optimization.
                    std::vector<float> ready_row;
                    if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 && !is_hadamard) {
                        // Recording mode: accumulate this row's contribution
                        // to src0's per-input-channel max|A_k| BEFORE any
                        // smoothing divide below, so a GGML_RKNPU2_CALIB run
                        // (GGML_RKNPU2_SMOOTH unset) measures the true
                        // unsmoothed activation distribution.
                        if (rknpu2_smoothquant::calib_enabled()) {
                            rknpu2_smoothquant::record_activation(src0, k_seg.offset_k, ready_ptr, K_seg_op);
                        }
                        // Apply mode: divide by the same s_k the weight side
                        // (ggml_backend_rknpu_buffer_set_tensor) already
                        // folded into src0's weight tile at load time, so
                        // the amax_m/scales_A[m] computation right below
                        // (unchanged) runs on the smoothed row.
                        if (rknpu2_smoothquant::smooth_enabled()) {
                            const std::vector<float>* s = rknpu2_smoothquant::lookup_s(src0);
                            if (s != nullptr) {
                                ready_row.assign(ready_ptr, ready_ptr + K_seg_op);
                                for (int kk = 0; kk < K_seg_op; ++kk) {
                                    const int k = k_seg.offset_k + kk;
                                    if ((size_t)k < s->size()) ready_row[kk] /= (*s)[k];
                                }
                                ready_ptr = ready_row.data();
                            }
                        }
                    }

                    // Handling types and quantizations
                    if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                        uint16_t* dst_ptr = (uint16_t*)dst_base;
                        uint16_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                        rknpu2_quantization::convert_fp32_to_fp16(ready_ptr, dst_row, K_seg_op);
                        if (rknpu2_debug_enabled()) {
                            double dbg_sum = 0; for (int dk = 0; dk < K_seg_op; ++dk) dbg_sum += ready_ptr[dk];
                            fprintf(stderr, "[RKNPU2_DBG] A-CONV m=%d K_seg_op=%d k_off=%d src_first=%.6f src_last=%.6f src_sum=%.6f conv_first4=%04x,%04x,%04x,%04x conv_last=%04x\n",
                                m, K_seg_op, k_seg.offset_k, ready_ptr[0], ready_ptr[K_seg_op-1], dbg_sum,
                                dst_row[0], dst_row[1], dst_row[2], dst_row[3], dst_row[K_seg_op-1]);
                            fflush(stderr);
                        }
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                        float amax_m = 0.0f;
                        for (int k = 0; k < K_seg_op; ++k) amax_m = std::max(amax_m, std::abs(ready_ptr[k]));
                        scales_A[m] = amax_m / 127.0f;

                        int8_t* dst_ptr = (int8_t*)dst_base;
                        int8_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                        rknpu2_quantization::quantize_fp32_to_int8(ready_ptr, dst_row, K_seg_op, scales_A[m]);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                        float amax_m = 0.0f;
                        for (int k = 0; k < K_seg_op; ++k) amax_m = std::max(amax_m, std::abs(ready_ptr[k]));
                        scales_A[m] = amax_m / 7.0f;

                        uint8_t* dst_ptr = (uint8_t*)dst_base;
                        uint8_t* dst_row = dst_ptr + (size_t)m * (K_seg_op / 2);
                        rknpu2_quantization::quantize_fp32_to_int4_packed(ready_ptr, dst_row, K_seg_op, scales_A[m]);
                    }
                }
                if (prof) prof_a_prep_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prof_t_aprep_start).count();

                // Assigning A-matrix to all contexts for the parallel execution.
                // fix10-host-roundtrip-20260902: skip the rknn_matmul_set_io_mem
                // call when this context's A slot is already bound to the same
                // rknn_tensor_mem object (mirrors the existing b_bound skip for
                // B below) -- avoids a per-token, per-active-segment driver call
                // that does not change any binding in the common (stable
                // M_op/K_seg_op/type/domain) decode-loop case.
                std::chrono::steady_clock::time_point prof_t_abind_start;
                if (prof) prof_t_abind_start = std::chrono::steady_clock::now();
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    if (matmul_ctxs[idx]->a_bound_mem != mem_A_shared.get()) {
                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[idx]->ctx, mem_A_shared.get(), &matmul_ctxs[idx]->io_attr.A), "set_io_mem A for core");
                        matmul_ctxs[idx]->a_bound_mem = mem_A_shared.get();
                    }
                }

                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[0]->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
                if (prof) prof_a_bind_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prof_t_abind_start).count();
            }

            // ===========================================
            // ========== 3. Preparing C-matrix ==========
            // ===========================================
            {
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    auto& matmul_ctx = matmul_ctxs[idx];
                    auto cache_key = std::make_tuple(M_op, active_n_segments[idx].size_n, active_n_segments[idx].core_id, (int)pipeline->npu_type_c, b_domain_id);

                    // Getting C-buffer from cache
                    mem_C_segments[idx] = get_tensor_buffer(backend_ctx, matmul_ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                    if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;

                    // Assigning C-matrix to current context for the parallel execution
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
                }
            }

            // ==========================================
            // ========== 4. Running operation ==========
            // ==========================================
            {
                if (rknpu2_debug_enabled()) {
                    uint16_t* ar = (uint16_t*)mem_A_shared->virt_addr;
                    fprintf(stderr, "[RKNPU2_DBG] PRE-RUN-A mem_A=%p ptr=%p M=%d M_op=%d K_seg_op=%d row0_first4=%04x,%04x,%04x,%04x\n",
                        (void*)mem_A_shared.get(), mem_A_shared->virt_addr, M, M_op, K_seg_op, ar[0], ar[1], ar[2], ar[3]);
                    fflush(stderr);
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        auto& mc = matmul_ctxs[idx];
                        uint8_t* rb = mc->mem_B ? (uint8_t*)mc->mem_B->virt_addr : nullptr;
                        RKNPU2_DBG("PRE-RUN idx=%zu matmul_ctx=%p mem_B=%p mem_B_ptr=%p first8=%s\n",
                            idx, (void*)mc.get(), mc->mem_B ? (void*)mc->mem_B.get() : nullptr, (void*)rb,
                            rb ? [&]{ static char buf[32]; snprintf(buf, sizeof(buf), "%02x%02x%02x%02x%02x%02x%02x%02x", rb[0],rb[1],rb[2],rb[3],rb[4],rb[5],rb[6],rb[7]); return (const char*)buf; }() : "(null)");
                    }
                }
                std::chrono::steady_clock::time_point prof_t_run_start;
                if (prof) prof_t_run_start = std::chrono::steady_clock::now();
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    int ret = rknn_matmul_run(matmul_ctxs[idx]->ctx);
                    if (ret != RKNN_SUCC) {
                        RKNPU2_DBG("rknn_matmul_run FAILED idx=%zu ret=%d\n", idx, ret);
                    }
                }
                if (prof) prof_run_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prof_t_run_start).count();
            }

            // ===========================================
            // ========== 5. Collecting results ==========
            // ===========================================
            {
                std::chrono::steady_clock::time_point prof_t_c_start;
                if (prof) prof_t_c_start = std::chrono::steady_clock::now();
                std::atomic<int> c_sync_error{0};
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    int ret = rknn_mem_sync(
                        matmul_ctxs[idx]->ctx,
                        mem_C_segments[idx].get(),
                        RKNN_MEMORY_SYNC_FROM_DEVICE);
                    if (ret < 0) {
                        int expected = 0;
                        c_sync_error.compare_exchange_strong(expected, ret);
                    }
                }
                if (c_sync_error.load() < 0) {
                    RKNN_LOG_FAILURE(c_sync_error.load(), "sync C FROM_DEVICE (parallel)");
                    return GGML_STATUS_FAILED;
                }

                if (rknpu2_debug_enabled() && pipeline->npu_type_c == rknpu2_configuration::NPU_TYPE_FP32) {
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        float* craw = (float*)mem_C_segments[idx]->virt_addr;
                        int nseg = active_n_segments[idx].size_n;
                        fprintf(stderr, "[RKNPU2_DBG] C-RAW idx=%zu M_op=%d N_seg=%d row0_first4=%.6f,%.6f,%.6f,%.6f row0_last=%.6f\n",
                            idx, M_op, nseg, craw[0], craw[1], craw[2], craw[3], craw[nseg-1]);
                    }
                    fflush(stderr);
                }

                const float hadamard_divisor = pipeline->use_hadamard ? (float)K_op : 1.0f;

                #pragma omp parallel for
                for (int m = 0; m < M; m++) {
                    // Handling types and quantizations
                    switch (pipeline->npu_type_c) {
                        case rknpu2_configuration::NPU_TYPE_FP32: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                const float* wscale = scales_B_grid.empty() ? nullptr : (scales_B_grid.data() + k_idx * N + N_offset);
                                float* src_segment_base = (float*)mem_C_segments[idx]->virt_addr;
                                float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                                float* src_ptr = src_segment_base + (size_t)m * N_segment;

                                for(int n=0; n<N_segment; ++n) {
                                    float scale_B = wscale ? wscale[n] : 1.0f;
                                    float dequant_scale = (scales_A[m] * scale_B) / hadamard_divisor;
                                    dst_ptr[n] += src_ptr[n] * dequant_scale;
                                }
                            }
                            if (rknpu2_debug_enabled() && !pipeline->use_hadamard && all_k_segments.size() == 1) {
                                std::lock_guard<std::mutex> dbg_lock(g_debug_weight_mutex);
                                auto dbg_it = g_debug_weight_fp32.find(src0);
                                if (dbg_it != g_debug_weight_fp32.end() && (int)dbg_it->second.size() == N * K) {
                                    const float* w = dbg_it->second.data();
                                    const float* xr_base = (const float*)get_tensor_real_ptr(src1);
                                    const int dbg_row_stride = (int)(src1->nb[1] / sizeof(float));
                                    const float* xr = xr_base + (size_t)m * dbg_row_stride;
                                    double max_abs_err = 0, max_rel_err = 0; int worst_n = -1;
                                    for (int n = 0; n < N; ++n) {
                                        double acc = 0;
                                        for (int k = 0; k < K; ++k) acc += (double)xr[k] * (double)w[(size_t)n * K + k];
                                        double got = dst_data[(size_t)m * N + n];
                                        double err = fabs(got - acc);
                                        double rel = err / (fabs(acc) + 1e-6);
                                        if (err > max_abs_err) { max_abs_err = err; worst_n = n; }
                                        if (rel > max_rel_err) max_rel_err = rel;
                                        if (m == 0) fprintf(stderr, "[RKNPU2_DBG] C-CHECK m=%d n=%d got=%.6f ref=%.6f diff=%.6f ratio=%.6f\n",
                                            m, n, got, acc, got - acc, (acc != 0.0) ? got/acc : 0.0);
                                    }
                                    fprintf(stderr, "[RKNPU2_DBG] C-CHECK-SUMMARY m=%d M=%d N=%d K=%d maxAbsErr=%.6f maxRelErr=%.6f worst_n=%d\n",
                                        m, M, N, K, max_abs_err, max_rel_err, worst_n);
                                    fflush(stderr);
                                }
                            }
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT32: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                const float* wscale = scales_B_grid.empty() ? nullptr : (scales_B_grid.data() + k_idx * N + N_offset);
                                float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                                int32_t* src_ptr = (int32_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;

                                for(int n=0; n<N_segment; ++n) {
                                    float scale_B = wscale ? wscale[n] : 1.0f;
                                    float dequant_scale = (scales_A[m] * scale_B) / hadamard_divisor;
                                    dst_ptr[n] += (float)src_ptr[n] * dequant_scale;
                                }
                            }
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT16: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                const float* wscale = scales_B_grid.empty() ? nullptr : (scales_B_grid.data() + k_idx * N + N_offset);
                                float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                                int16_t* src_ptr = (int16_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;

                                for(int n=0; n<N_segment; ++n) {
                                    float scale_B = wscale ? wscale[n] : 1.0f;
                                    float dequant_scale = (scales_A[m] * scale_B) / hadamard_divisor;
                                    dst_ptr[n] += (float)src_ptr[n] * dequant_scale;
                                }
                            }
                            break;
                        }

                        default:
                            // This should not be reached if config is correct
                            break;
                    }
                }
                if (prof) prof_c_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prof_t_c_start).count();
            }
        }
        } // i2 (rknpu2-broadcast-mulmat-20260902, fix #6)
        } // i3 (rknpu2-broadcast-mulmat-20260902, fix #6)
    }

    // fix10-profile-20260902: fold this call's accumulators into the
    // backend's lifetime totals and print one raw per-token line. Both
    // gated behind the same cached `prof` bool checked at function entry --
    // with GGML_RKNPU2_PROFILE unset this whole block is skipped.
    if (prof) {
        backend_ctx->prof_b_bind_ns += prof_b_bind_ns;
        backend_ctx->prof_a_prep_ns += prof_a_prep_ns;
        backend_ctx->prof_a_bind_ns += prof_a_bind_ns;
        backend_ctx->prof_run_ns    += prof_run_ns;
        backend_ctx->prof_c_ns      += prof_c_ns;
        backend_ctx->prof_n_matmul  += prof_n_matmul_this_call;
        backend_ctx->prof_n_tokens  += 1;
        fprintf(stderr,
            "[RKNPU2_PROFILE] tok=%llu n_matmul=%llu b_bind_us=%.1f a_prep_us=%.1f a_bind_us=%.1f run_us=%.1f c_us=%.1f\n",
            (unsigned long long)backend_ctx->prof_n_tokens, (unsigned long long)prof_n_matmul_this_call,
            prof_b_bind_ns / 1e3, prof_a_prep_ns / 1e3, prof_a_bind_ns / 1e3, prof_run_ns / 1e3, prof_c_ns / 1e3);
        fflush(stderr);
    }

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

// Function for calculating a real tensor size for the NPU
static size_t get_tensor_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        size_t total_size = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& seg : n_segments) {
                if (seg.size_n > 0) {
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        total_size += (size_t)seg.size_n * k_seg.size_k / 2;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
                        total_size += (size_t)seg.size_n * k_seg.size_k;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
                        total_size += (size_t)seg.size_n * k_seg.size_k * 2;
                    }
                }
            }
        }

        // rknpu2-broadcast-mulmat-20260902 (fix #6): total_size above is the
        // packed size of ONE logical (K,N) 2D slice. A tensor that can be the
        // src0 (weight/"B") side of a GQA-broadcast MUL_MAT (ne[2]/ne[3] > 1,
        // e.g. a per-head-kv-group K/V cache slab) needs ne[2]*ne[3] such
        // slices packed back-to-back -- see the matching per-slice loop this
        // fix adds to ggml_backend_rknpu_buffer_set_tensor() below. Ordinary
        // 2D weight tensors have ne[2]==ne[3]==1, so this is a no-op for them.
        return total_size * (size_t)tensor->ne[2] * (size_t)tensor->ne[3];
    }
    return ggml_nbytes(tensor);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    // Freeing an every individual RKNN buffer using the allocator context
    for (auto& pair : ctx->tensor_allocs) {
        if (pair.second.mem) {
            RKNPU2_DBG("free_buffer buf=%s offset=%zu fd=%d ptr=%p size=%zu (invalidating matmul_ctx_cache for this addr)\n",
                ctx->name.c_str(), pair.first, pair.second.mem->fd, pair.second.mem->virt_addr, pair.second.size);
            if (g_active_rknpu_backend_ctx) {
                g_active_rknpu_backend_ctx->invalidate_matmul_ctx_for_address(pair.second.mem->virt_addr);
            }
            rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(pair.second.iommu_domain_id);
            if (alloc_ctx != 0) {
                rknn_destroy_mem(alloc_ctx, pair.second.mem);
            } else {
                fprintf(stderr,
                    "RKNPU2 WARNING: free_buffer buf=%s offset=%zu -- could "
                    "not re-acquire allocator context for domain %d to free "
                    "this buffer (leaking it instead of calling "
                    "rknn_destroy_mem with an invalid context) -- see prior "
                    "RKNPU ERROR above\n",
                    ctx->name.c_str(), pair.first, pair.second.iommu_domain_id);
            }
            g_domain_manager.release_domain_memory(pair.second.iommu_domain_id, pair.second.size);
        }
    }

    // Freeing the virtual memory block
    munmap(ctx->virtual_base, ctx->total_size);

    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->virtual_base;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    // Initialize tensor only if it is supported by the pipeline
    if (pipeline) {
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
        size_t size = get_tensor_packed_size(tensor);
        RKNPU2_DBG("init_tensor tensor=%p name=%s buf=%s offset=%zu size=%zu\n",
            (const void*)tensor, tensor->name, ctx->name.c_str(), offset, size);
        auto alloc = ctx->get_tensor_allocation(offset, size);
        if (alloc.mem == nullptr) {
            fprintf(stderr,
                "RKNPU2 ERROR: buffer_init_tensor tensor=%s buf=%s offset=%zu "
                "size=%zu -- RKNN buffer allocation failed (see prior "
                "RKNPU2 ERROR above, likely fd-limit exhaustion; try "
                "`ulimit -n 65536`); failing tensor init cleanly instead of "
                "leaving a null buffer for later ops to dereference\n",
                tensor->name, ctx->name.c_str(), offset, size);
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

// Function for dequantizing a single row from GGUF format to FP32
static void dequantize_row(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int n, int K,
    float * row_out)
{
    if (tensor->type == GGML_TYPE_F32) {
        const float* src = (const float*)raw_data;
        memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
        const ggml_fp16_t* src_row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        const block_q8_0* src = (const block_q8_0*)raw_data;
        dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        const block_q6_K* src = (const block_q6_K*)raw_data;
        dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        const block_q4_0* src = (const block_q4_0*)raw_data;
        dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
    } else {
        GGML_ASSERT(false && "Unsupported weight type for NPU pipeline");
    }
}

// Function for extracting a specific tensor segment and converting it to FP32
static void dequantize_tensor_segment(
    std::vector<float>& out_segment,
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    bool use_hadamard)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    out_segment.resize(seg_elements);

    std::vector<float> s_vec;
    if (use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        s_vec = ctx->hadamard_s_vectors[tensor];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        int global_n = n_seg.offset_n + i;

        if (global_n < N) {
            std::vector<float> row_raw(K);
            std::vector<float> row_processed(K_op, 0.0f);

            dequantize_row(tensor, raw_data, global_n, K, row_raw.data());

            if (use_hadamard) {
                std::vector<float> signed_row(K);
                for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec[k];
                rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
            } else {
                memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
            }

            memcpy(&out_segment[i * k_seg.size_k], &row_processed[k_seg.offset_k], k_seg.size_k * sizeof(float));
        } else {
            memset(&out_segment[i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
        }
    }
}

// Function for quantizing the FP32 segment to the target NPU format
static void quantize_tensor_segment(
    const std::vector<float>& fp32_segment,
    std::vector<uint8_t>& out_quantized,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float>& row_scales,
    rknpu2_configuration::Rknpu2NpuType npu_type)
{
    const int K_seg = k_seg.size_k;
    const int N_seg = n_seg.size_n;
    const size_t seg_elements = (size_t)N_seg * K_seg;

    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        out_quantized.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(
            fp32_segment.data(),
            (uint16_t*)out_quantized.data(),
            seg_elements);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        out_quantized.resize(seg_elements);
        int8_t* dst = (int8_t*)out_quantized.data();
        for (int i = 0; i < N_seg; ++i) {
            const float* src_row = fp32_segment.data() + (size_t)i * K_seg;
            int8_t* dst_row = dst + (size_t)i * K_seg;
            rknpu2_quantization::quantize_fp32_to_int8(src_row, dst_row, K_seg, row_scales[i]);
        }
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT4) {
        out_quantized.resize(seg_elements / 2);
        uint8_t* dst = out_quantized.data();
        for (int i = 0; i < N_seg; ++i) {
            const float* src_row = fp32_segment.data() + (size_t)i * K_seg;
            uint8_t* dst_row = dst + (size_t)i * (K_seg / 2);
            rknpu2_quantization::quantize_fp32_to_int4_packed(src_row, dst_row, K_seg, row_scales[i]);
        }
    }
}

// Function for packing
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits)
{
    UNUSED(N_total);

    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes     = (size_t)k_align * element_bits / 8;
    const size_t src_row_bytes  = (size_t)K_total * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            uint8_t* dst_tile = dst + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_global = (size_t)n_offset + ni * n_align + nn;
                const size_t k_start  = (size_t)k_offset + ki * k_align;

                const uint8_t* src_ptr = src + n_global * src_row_bytes
                                             + k_start * element_bits / 8;
                uint8_t* dst_ptr = dst_tile + nn * k_sub_bytes;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for unpacking: the exact inverse of pack_native() above. Reads a
// segment from the chip native tiled layout (src) and reconstructs the
// compact [n_segment x k_segment] row-major buffer it was packed from
// (dst), local to this segment (k_offset=0/n_offset=0 convention, matching
// how pack_tensor_segment() calls pack_native() -- see Stage-5 fix in
// ggml_backend_rknpu_buffer_get_tensor below).
static void unpack_native(
    uint8_t* dst, const uint8_t* src,
    int k_segment, int k_align,
    int n_segment, int n_align,
    int element_bits)
{
    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes    = (size_t)k_align * element_bits / 8;
    const size_t dst_row_bytes  = (size_t)k_segment * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            const uint8_t* src_tile = src + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_local  = ni * n_align + nn;
                const size_t k_start  = ki * k_align;
                const uint8_t* src_ptr = src_tile + nn * k_sub_bytes;
                uint8_t* dst_ptr = dst + n_local * dst_row_bytes + k_start * element_bits / 8;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for packing the quantized segment into the native NPU layout and writing to DMA
static size_t pack_tensor_segment(
    const std::vector<uint8_t>& quantized_segment,
    uint8_t * dst_dma_ptr,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    int element_bits = 0;
    size_t segment_packed_size = 0;

    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k * 2;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k / 2;
    }

    pack_native(dst_dma_ptr, quantized_segment.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return segment_packed_size;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
    RKNPU2_DBG("set_tensor ENTER tensor=%p name=%s buf=%s off_virtual=%zu call_offset=%zu call_size=%zu pipeline=%s\n",
        (const void*)tensor, tensor->name, ctx->name.c_str(), tensor_offset_in_virtual, offset, size, pipeline ? "yes" : "no");

    if (pipeline) {
        // npu_fix5b_keep_host_weights_20260902: stash the verbatim host
        // bytes (this tensor's real GGUF block-format encoding, e.g. actual
        // per-32-element q4_0 blocks) before anything below quantizes/packs
        // them into the chip-native layout. Test/diagnosis-only, env-gated,
        // no cost when unset. May be called multiple times for the same
        // tensor at different `offset` (chunked upload) -- accumulate into
        // one ggml_nbytes(tensor)-sized buffer keyed by tensor pointer.
        if (rknpu2_keep_host_weights_enabled()) {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            auto& stash = ctx->host_weight_bytes[tensor];
            const size_t total = ggml_nbytes(tensor);
            if (stash.size() != total) stash.assign(total, 0);
            if (offset + size <= stash.size()) {
                memcpy(stash.data() + offset, data, size);
            } else {
                fprintf(stderr,
                    "RKNPU2 WARNING: set_tensor tensor=%s -- fix5b host-weight "
                    "stash write out of range (offset=%zu size=%zu total=%zu); "
                    "skipping stash for this chunk, get_tensor's verbatim "
                    "readback for this tensor will be incomplete/stale\n",
                    tensor->name, offset, size, total);
            }
        }

        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];
        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        // Initializing Hadamard Transform Logic
        if (pipeline->use_hadamard) {
            std::vector<float> s_vec(K_op, 1.0f);
            std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor));
            std::uniform_int_distribution<int> distrib(0, 1);

            for(int k = 0; k < K_op; ++k) {
                s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
            }

            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->hadamard_s_vectors[tensor] = s_vec;
        }

        // Computing global scale
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        // Allocating a new buffer for a tensor. rk3576-emfile-fix-window-
        // 20260828: this used to dereference alloc.mem->virt_addr
        // unconditionally -- get_tensor_allocation() returns mem==nullptr on
        // allocation failure (e.g. fd-limit exhaustion / errno 24), which
        // made this THE primary SIGSEGV site during model load. Check first
        // and drop this set_tensor cleanly instead of crashing.
        size_t required_size = get_tensor_packed_size(tensor);
        auto alloc = ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
        if (alloc.mem == nullptr) {
            fprintf(stderr,
                "RKNPU2 ERROR: set_tensor tensor=%s buf=%s offset=%zu "
                "required_size=%zu -- RKNN buffer allocation failed (see "
                "prior RKNPU2 ERROR above, likely fd-limit exhaustion; try "
                "`ulimit -n 65536`); dropping this set_tensor cleanly instead "
                "of dereferencing a null buffer. This tensor's weights will "
                "be missing/stale -- expect graph_compute to fail cleanly "
                "for ops that read it.\n",
                tensor->name, ctx->name.c_str(), tensor_offset_in_virtual, required_size);
            return;
        }
        uint8_t* tensor_dma_ptr = (uint8_t*)alloc.mem->virt_addr;

        // Computing specific hardware segments
        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        std::vector<float> seg_fp32;
        std::vector<uint8_t> seg_npu;
        uint8_t* current_write_ptr = tensor_dma_ptr + offset;

        // Per-channel weight scales storage
        std::vector<float> per_channel_scales;
        if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
            per_channel_scales.resize(k_segments.size() * N, 1.0f);
        }

        std::vector<float> row_scales;

        // rknpu2-broadcast-mulmat-20260902 (fix #6): a tensor eligible to be
        // the src0/"B" side of a GQA-broadcast MUL_MAT carries ne[2]/ne[3] >
        // 1 (e.g. one packed slice per K/V head-kv-group). Loop the existing
        // per-2D-slice segment-packing logic below over every (i3,i2) slice,
        // packing them back-to-back into the buffer sized by the matching
        // ne[2]*ne[3] multiply this fix adds to get_tensor_packed_size()
        // above. i3-major, i2-minor order to match the slice_linear_index =
        // i03*ne[2]+i02 addressing ggml_backend_rknpu_graph_compute() (this
        // file) uses to find a given slice's B-matrix offset. Ordinary 2D
        // weight tensors (ne[2]==ne[3]==1) run this loop exactly once, byte
        // for byte identical to before this fix -- `slice_raw_data == data`.
        // NOTE (scope): this only re-derives each slice's *row data*
        // (dequantize_row indexes `raw_data` as a flat contiguous (N,K)
        // block per tensor->nb[2]/nb[3] strides, which requires `tensor` to
        // be contiguous -- already required by supports_op's
        // ggml_is_contiguous() check on the graph_compute side). It does
        // NOT extend `offset`/`size` partial-write handling: like the
        // pre-fix-6 code, this assumes a single full-tensor write
        // (offset==0, size==ggml_nbytes(tensor)), true for how llama.cpp
        // populates NPU-resident weights and (per current evidence) also
        // true for how K/V-cache tensors reach this backend; a future
        // incremental (row-at-a-time) KV-cache write path through this
        // buffer type would need separate handling, out of scope here.

        // npu_fix12_smoothquant_20260902: GGML_RKNPU2_SMOOTH pre-pass.
        // Finds this tensor's per-input-channel max|W_k| by dequantizing
        // every segment the exact same way the real pass below does (same
        // dequantize_tensor_segment() call), just discarding each tile
        // instead of continuing on to fix8's scale/quantize/pack steps, so
        // rknpu2_smoothquant::compute_and_cache_s() can fold it against the
        // loaded max|A_k| stats into a finalized s_k BEFORE the real pass's
        // fold (see the insertion right before "Calculating local scale of
        // the block." below, which is a pure lookup of what this pre-pass
        // caches here). Only for the INT8-activation, non-Hadamard
        // pipelines this fix targets -- zero cost otherwise (single
        // cached-bool branch via smooth_enabled()). Runs once per
        // set_tensor call (model load), never in the per-token decode
        // path. K_op==K always here: use_hadamard is false whenever this
        // branch is taken.
        if (rknpu2_smoothquant::smooth_enabled() &&
            pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 &&
            !pipeline->use_hadamard) {
            std::vector<float> w_col_absmax((size_t)K, 0.0f);
            std::vector<float> stats_seg_fp32;
            for (int64_t si3 = 0; si3 < tensor->ne[3]; ++si3) {
            for (int64_t si2 = 0; si2 < tensor->ne[2]; ++si2) {
                const void* stats_slice_raw_data = (const uint8_t*)data + si3 * tensor->nb[3] + si2 * tensor->nb[2];
                for (const auto& sk_seg : k_segments) {
                    for (const auto& sn_seg : n_segments) {
                        if (sn_seg.size_n == 0) continue;
                        dequantize_tensor_segment(stats_seg_fp32, tensor, ctx, stats_slice_raw_data, K, N, K_op, sk_seg, sn_seg, pipeline->use_hadamard);
                        for (int i = 0; i < sn_seg.size_n; ++i) {
                            const float* row = stats_seg_fp32.data() + (size_t)i * sk_seg.size_k;
                            for (int kk = 0; kk < sk_seg.size_k; ++kk) {
                                const int k = sk_seg.offset_k + kk;
                                if (k < K) w_col_absmax[k] = std::max(w_col_absmax[k], std::fabs(row[kk]));
                            }
                        }
                    }
                }
            }
            }
            rknpu2_smoothquant::compute_and_cache_s(tensor, w_col_absmax.data(), K);
        }

        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
        const void* slice_raw_data = (const uint8_t*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2];

        // Processing individual segments block-by-block
        for (size_t k_idx = 0; k_idx < k_segments.size(); ++k_idx) {
            const auto& k_seg = k_segments[k_idx];
            for (const auto& n_seg : n_segments) {
                if (n_seg.size_n == 0) continue;

                // Dequantizing the block
                dequantize_tensor_segment(seg_fp32, tensor, ctx, slice_raw_data, K, N, K_op, k_seg, n_seg, pipeline->use_hadamard);

                // Stage-5 debug: stash this block's FP32 values into a full
                // N x K side-buffer keyed by tensor pointer, for graph_compute's
                // independent CPU cross-check (see g_debug_weight_fp32 above).
                // Not extended per-slice (diagnostic-only): for a batched
                // tensor this ends up holding only the LAST (i3,i2) slice
                // visited.
                if (rknpu2_debug_enabled() && !pipeline->use_hadamard) {
                    std::lock_guard<std::mutex> dbg_lock(g_debug_weight_mutex);
                    auto& dbuf = g_debug_weight_fp32[tensor];
                    if ((int)dbuf.size() != N * K) dbuf.assign((size_t)N * K, 0.0f);
                    for (int i = 0; i < n_seg.size_n; ++i) {
                        int global_n = n_seg.offset_n + i;
                        if (global_n < N) {
                            int ncols = std::min((int)k_seg.size_k, K - k_seg.offset_k);
                            if (ncols > 0) {
                                memcpy(&dbuf[(size_t)global_n * K + k_seg.offset_k],
                                       &seg_fp32[(size_t)i * k_seg.size_k], ncols * sizeof(float));
                            }
                        }
                    }
                }

                // npu_fix12_smoothquant_20260902: fold per-input-channel
                // smoothing into the weight tile before the per-channel
                // absmax below is computed, so that amax already reflects
                // post-smoothing magnitudes. seg_fp32 layout: row i (local
                // output channel within this n_seg) * k_seg.size_k + kk
                // (local column); global input channel k = k_seg.offset_k +
                // kk, matching the s_k vector's indexing (size K,
                // K == tensor->ne[0]). s_k was computed once above by the
                // GGML_RKNPU2_SMOOTH pre-pass -- this is a pure lookup.
                if (rknpu2_smoothquant::smooth_enabled() &&
                    pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 &&
                    !pipeline->use_hadamard) {
                    const std::vector<float>* s = rknpu2_smoothquant::lookup_s(tensor);
                    if (s != nullptr) {
                        #pragma omp parallel for
                        for (int i = 0; i < n_seg.size_n; ++i) {
                            float* row = seg_fp32.data() + (size_t)i * k_seg.size_k;
                            for (int kk = 0; kk < k_seg.size_k; ++kk) {
                                const int k = k_seg.offset_k + kk;
                                if ((size_t)k < s->size()) row[kk] *= (*s)[k];
                            }
                        }
                    }
                }

                // Calculating per-channel scales of the segment
                if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
                    const float quant_divisor = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) ? 7.0f : 127.0f;
                    row_scales.resize(n_seg.size_n);

                    #pragma omp parallel for
                    for (int i = 0; i < n_seg.size_n; ++i) {
                        const float* row_fp32 = seg_fp32.data() + (size_t)i * k_seg.size_k;
                        float amax = 0.0f;
                        for (int j = 0; j < k_seg.size_k; ++j) {
                            amax = std::max(amax, std::abs(row_fp32[j]));
                        }
                        float sw = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
                        row_scales[i] = sw;

                        int global_n = n_seg.offset_n + i;
                        per_channel_scales[k_idx * N + global_n] = sw;
                    }
                } else {
                    row_scales.assign(n_seg.size_n, 1.0f);
                }

                // Quantizing
                quantize_tensor_segment(seg_fp32, seg_npu, k_seg, n_seg, row_scales, pipeline->npu_type_b);

                // Packing into chip native layout
                size_t bytes_written = pack_tensor_segment(seg_npu, current_write_ptr, k_seg, n_seg, pipeline);

                current_write_ptr += bytes_written;
            }
        }
        } // i2
        } // i3

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor] = std::move(per_channel_scales);
        }

        rknn_matmul_ctx sync_ctx = g_domain_manager.get_allocator_context(alloc.iommu_domain_id);
        if (sync_ctx == 0) {
            fprintf(stderr,
                "RKNPU2 ERROR: set_tensor tensor=%s buf=%s -- could not "
                "re-acquire allocator context for domain %d to sync the "
                "buffer to device (see prior RKNPU ERROR above); dropping "
                "this set_tensor cleanly instead of calling into RKNN with "
                "an invalid context\n",
                tensor->name, ctx->name.c_str(), alloc.iommu_domain_id);
            return;
        }
        RKNN_CHECK_VOID(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");
        if (rknpu2_debug_enabled()) {
            uint8_t* pb = (uint8_t*)alloc.mem->virt_addr;
            RKNPU2_DBG("set_tensor DONE tensor=%p name=%s fd=%d ptr=%p first8_packed=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                (const void*)tensor, tensor->name, alloc.mem->fd, alloc.mem->virt_addr,
                pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6], pb[7]);
        }
    } else {
        memcpy((uint8_t*)tensor->data + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context*)buffer->context;
    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto it = ctx->tensor_allocs.find(tensor_offset_in_virtual);
    // it->second.mem can be nullptr if a prior allocation for this tensor
    // failed (rk3576-emfile-fix-window-20260828, e.g. fd-limit exhaustion) --
    // treat that the same as "not found" and fall back to the raw backing
    // bytes instead of dereferencing a null buffer below.
    if (it == ctx->tensor_allocs.end() || it->second.mem == nullptr) {
        if (it != ctx->tensor_allocs.end()) {
            fprintf(stderr,
                "RKNPU2 WARNING: get_tensor tensor=%s buf=%s offset=%zu -- "
                "RKNN buffer allocation previously failed for this tensor "
                "(see prior RKNPU2 ERROR above); returning raw backing bytes "
                "instead of dereferencing a null buffer\n",
                tensor->name, ctx->name.c_str(), tensor_offset_in_virtual);
        }
        memcpy(data, (uint8_t*)tensor->data + offset, size);
        return;
    }

    // npu_fix5b_keep_host_weights_20260902: if this tensor's original
    // host-side bytes were stashed by set_tensor() (GGML_RKNPU_KEEP_HOST_WEIGHTS
    // enabled), return them verbatim instead of reconstructing the weight
    // from the chip-native quantized/Hadamard layout below. This makes a
    // readback-sensitive caller's CPU reference exact (identical to what
    // set_tensor() was actually called with), so any remaining NPU-vs-CPU
    // delta is genuine NPU compute error (weight-quant + activation-quant +
    // hardware accumulation), not an artifact of re-deriving and
    // re-quantizing the weight for the comparison. Takes priority over the
    // FP16 and fix5 INT8/INT4 reconstruction paths below; falls through to
    // them (then to the raw-bytes memcpy) if the stash is missing/stale for
    // this tensor, e.g. env was enabled after this tensor's set_tensor().
    if (rknpu2_keep_host_weights_enabled()) {
        auto hit = ctx->host_weight_bytes.find(tensor);
        if (hit != ctx->host_weight_bytes.end() && offset + size <= hit->second.size()) {
            memcpy(data, hit->second.data() + offset, size);
            return;
        }
    }

    // Stage-5 fix (rk3576-stage5-20260828): for a "pipeline" tensor (a
    // weight matmul routed through the NPU), the backing allocation holds
    // the chip-native packed/quantized layout, NOT the tensor's declared
    // row-major ggml format. Blindly memcpy-ing those bytes back (the old
    // behavior, still used below for pipelines this fix doesn't cover) is
    // harmless for real serving -- weights are write-once, never read back
    // -- but ggml_backend_compare_graph_backend (test-backend-ops MODE_TEST)
    // DOES read input tensors back to build its CPU reference. That silently
    // fed the CPU reference scrambled native-tiled bytes reinterpreted as
    // row-major FP16, producing the "FAIL ERR=1.6-4.2" results Stages 2-4
    // chased -- even though the NPU's actual matmul was numerically correct
    // the whole time (proven via independent CPU cross-check computed from
    // the pre-pack FP32 values, see rk3576-stage5-window-1-20260828
    // evidence: NPU output matched a from-scratch reference to ~0.1-0.7%,
    // consistent with FP16 rounding, not the reported 110-420% error).
    // Fix: unpack the native tiling and dequantize back to FP32 in the
    // tensor's original (n, k) order, then re-encode to FP16 row-major, so
    // a caller reading this tensor back gets a faithful round-trip.
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline && !pipeline->use_hadamard && pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16
        && tensor->type == GGML_TYPE_F16) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto k_segments = compute_k_segments(K, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        // rk3588-get-tensor-readback-size-20260902 + rknpu2-broadcast-
        // mulmat-20260902 (fix #6): the unpack loop below now walks every
        // (i3,i2) slice of a batched (ne[2]/ne[3] > 1) tensor -- e.g. a
        // per-head-kv-group K/V cache slab -- in the same i3-major,
        // i2-minor order ggml_backend_rknpu_buffer_set_tensor() packed them
        // in (see its matching fix-6 loop above), instead of only ever
        // filling the first N*K elements and leaving the rest zeroed. The
        // caller-supplied `size` (== ggml_nbytes(tensor) for the common
        // offset=0 full-tensor read, ggml-backend.cpp:401/408) already
        // counts every GGML_MAX_DIMS via tensor->nb[] strides, so sizing
        // the staging allocation from the same expression the final memcpy
        // uses (as before this fix) is already big enough for the full
        // ne[2]*ne[3] batch -- this fix only changes whether it gets
        // filled correctly instead of zero-padded past slice 0.
        const size_t full_elems = std::max<size_t>((size_t)N * K, (offset + size + sizeof(uint16_t) - 1) / sizeof(uint16_t));
        std::vector<uint16_t> full_f16(full_elems, 0);
        const uint8_t* read_ptr = (const uint8_t*)it->second.mem->virt_addr;
        const int64_t ne2 = tensor->ne[2], ne3 = tensor->ne[3];

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
        const size_t slice_elem_base = ((size_t)i3 * ne2 + (size_t)i2) * (size_t)N * K;
        if (slice_elem_base >= full_elems) continue; // defensive; should not happen, full_elems covers ne2*ne3*N*K

        for (const auto& k_seg : k_segments) {
            for (const auto& n_seg : n_segments) {
                if (n_seg.size_n == 0) continue;
                size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;

                std::vector<uint16_t> unpacked(seg_elements);
                unpack_native((uint8_t*)unpacked.data(), read_ptr,
                               k_seg.size_k, pipeline->k_align, n_seg.size_n, pipeline->n_align, 16);
                read_ptr += seg_elements * 2;

                for (int i = 0; i < n_seg.size_n; ++i) {
                    int global_n = n_seg.offset_n + i;
                    if (global_n >= N) continue;
                    memcpy(&full_f16[slice_elem_base + (size_t)global_n * K + k_seg.offset_k],
                           &unpacked[(size_t)i * k_seg.size_k], k_seg.size_k * sizeof(uint16_t));
                }
            }
        }
        } // i2
        } // i3

        memcpy(data, (const uint8_t*)full_f16.data() + offset, size);
        return;
    }

    // npu_fix5_q4_0_nan_20260902: generalizes the fast readback path above
    // to the INT8/INT4-weight pipelines (Q8_0/Q6_K/Q4_0 -> W8A8*/W4A4* on
    // RK3588, W8A16*/W4A16* on RK3576) the TODO below still leaves
    // uncovered -- this includes every pipeline GGML_TYPE_Q4_0 can resolve
    // to today (W4A4_HADAMARD / W4A16_HADAMARD are this backend's only
    // registered Q4_0 patterns, both use_hadamard=true, npu_type_b=INT4).
    // Root cause of npu_fix5_q4_0_nan_20260902.md's
    // MUL_MAT(q4_0,m=576,n=512,k=576) NaN: test-backend-ops' CPU reference
    // is built by reading this weight tensor BACK through this function
    // (see the Stage-5 comment above); for exactly this npu_type_b/
    // use_hadamard combination the old code fell straight through to the
    // raw-bytes memcpy at the bottom of this function, handing the CPU
    // reference builder the chip-native INT4-packed, Hadamard-transformed
    // bytes reinterpreted as a block_q4_0 array. A stray reinterpreted
    // fp16 "d" scale half-word from that mismatched byte layout can decode
    // to NaN/Inf and poison every weight dequantized from that block --
    // corrupting only the CPU reference, never the NPU's own output, which
    // matches the observed evidence exactly (RKNPU=-10.100720, CPU=-nan:
    // see npu_fix3_plan_20260902.md sec 1). Same unpack_native() primitive
    // as the block above, extended to: (1) dequantize with this segment's
    // real per-block scale (ctx->quantized_tensor_scales, unconditionally
    // populated by set_tensor() for every pipeline, scale==1.0f only for
    // the FP16 case handled above) instead of a raw bit-reinterpret;
    // (2) invert the forward Hadamard transform + random sign vector
    // set_tensor() applied for Hadamard pipelines (fwht_iterative() is its
    // own inverse up to a factor of K_op when invoked with K==padded_size
    // -- rknpu2-calibration.cpp -- matching the K_op divisor
    // graph_compute() already applies to MUL_MAT output for the identical
    // reason); (3) re-encode the recovered FP32 matrix into the tensor's
    // own declared block format via ggml_quantize_chunk() (ggml.h, already
    // visible here via the ggml-quants.h include above) instead of a
    // type-specific hand-rolled encoder.
    if (pipeline &&
        (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 ||
         pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) &&
        (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K)) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];
        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        // Already under ctx->mutex (locked at function entry above) --
        // must not re-lock (std::mutex is non-recursive).
        std::vector<float> block_scales;
        {
            auto sit = ctx->quantized_tensor_scales.find(tensor);
            if (sit != ctx->quantized_tensor_scales.end()) block_scales = sit->second;
        }
        std::vector<float> s_vec;
        if (pipeline->use_hadamard) {
            auto hit = ctx->hadamard_s_vectors.find(tensor);
            if (hit != ctx->hadamard_s_vectors.end()) s_vec = hit->second;
        }
        const bool undo_transform = pipeline->use_hadamard && !s_vec.empty();

        std::vector<float> full_fp32((size_t)N * K, 0.0f);
        std::vector<float> transformed;
        float* stage = full_fp32.data();
        size_t stage_stride = (size_t)K;
        if (undo_transform) {
            transformed.assign((size_t)N * K_op, 0.0f);
            stage = transformed.data();
            stage_stride = (size_t)K_op;
        }

        const uint8_t* read_ptr = (const uint8_t*)it->second.mem->virt_addr;
        size_t scale_idx = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& n_seg : n_segments) {
                if (n_seg.size_n == 0) continue;
                const size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
                const float scale = (scale_idx < block_scales.size()) ? block_scales[scale_idx] : 1.0f;
                ++scale_idx;

                const int element_bits = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) ? 8 : 4;
                const size_t packed_bytes = (element_bits == 4) ? (seg_elements / 2) : seg_elements;

                std::vector<uint8_t> packed(packed_bytes);
                unpack_native(packed.data(), read_ptr, k_seg.size_k, pipeline->k_align,
                               n_seg.size_n, pipeline->n_align, element_bits);
                read_ptr += packed_bytes;

                for (int i = 0; i < n_seg.size_n; ++i) {
                    const int global_n = n_seg.offset_n + i;
                    if (global_n >= N) continue;
                    float* dst_row = stage + (size_t)global_n * stage_stride + k_seg.offset_k;

                    if (element_bits == 8) {
                        const int8_t* src = (const int8_t*)packed.data() + (size_t)i * k_seg.size_k;
                        for (int k = 0; k < k_seg.size_k; ++k) dst_row[k] = (float)src[k] * scale;
                    } else {
                        const uint8_t* src = packed.data() + ((size_t)i * k_seg.size_k) / 2;
                        for (int k = 0; k < k_seg.size_k; k += 2) {
                            uint8_t b = src[k / 2];
                            int8_t lo = (int8_t)(b & 0x0F); if (lo >= 8) lo -= 16;
                            int8_t hi = (int8_t)((b >> 4) & 0x0F); if (hi >= 8) hi -= 16;
                            dst_row[k] = (float)lo * scale;
                            if (k + 1 < k_seg.size_k) dst_row[k + 1] = (float)hi * scale;
                        }
                    }
                }
            }
        }

        if (undo_transform) {
            std::vector<float> tmp(K_op);
            const float inv_K_op = 1.0f / (float)K_op;
            for (int n = 0; n < N; ++n) {
                float* row = &transformed[(size_t)n * K_op];
                rknpu2_calibration::hadamard_transform(tmp.data(), row, K_op, K_op);
                for (int k = 0; k < K; ++k) {
                    const float sign = (k < (int)s_vec.size()) ? s_vec[k] : 1.0f;
                    full_fp32[(size_t)n * K + k] = tmp[k] * inv_K_op * sign;
                }
            }
        }

        std::vector<uint8_t> reencoded(ggml_nbytes(tensor));
        const size_t written = ggml_quantize_chunk(tensor->type, full_fp32.data(), reencoded.data(), 0, N, K, nullptr);
        if (written == reencoded.size() && offset + size <= reencoded.size()) {
            memcpy(data, reencoded.data() + offset, size);
            return;
        }
        fprintf(stderr,
            "RKNPU2 WARNING: get_tensor tensor=%s -- fix5 reconstruction path "
            "produced an unexpected size (written=%zu expected=%zu) or an "
            "out-of-range request (offset=%zu size=%zu); falling back to raw "
            "native bytes (same limitation as before this fix)\n",
            tensor->name, written, reencoded.size(), offset, size);
    }

    // TODO(rk3576-stage5): the same class of bug applies to INT8/INT4
    // (Q8_0/Q6_K/Q4_0 weight) pipelines and Hadamard pipelines -- their
    // native packed bytes are handed back unchanged below, which is only
    // safe because real serving never reads weights back. Fix + hardware-
    // verify when those pipelines are next exercised by a readback-
    // sensitive caller (e.g. test-backend-ops coverage expands past
    // type_a=f16, or a future goal needs weight readback for those types).
    memcpy(data, (uint8_t*)it->second.mem->virt_addr + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);

    for (auto& pair : ctx->tensor_allocs) {
        // Skip entries whose allocation previously failed
        // (rk3576-emfile-fix-window-20260828, e.g. fd-limit exhaustion) --
        // mem==nullptr there, and there is nothing to clear.
        if (pair.second.mem == nullptr) continue;
        memset((uint8_t*)pair.second.mem->virt_addr, value, pair.second.size);
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    // Reserving virtual memory block
    void* virtual_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (virtual_base == MAP_FAILED) {
        return NULL;
    }

    // Initializing buffer context
    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->virtual_base = virtual_base;
    ctx->total_size = size;
    ctx->name = "rknpu_virtual_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return get_tensor_packed_size(tensor);
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // rknpu2-broadcast-mulmat-20260902 (fix #6, see companion loop
            // in ggml_backend_rknpu_graph_compute() above/below). Prior to
            // this, any operand carrying a batch dim (ne[2]/ne[3] != 1)
            // was unconditionally declined here (rknpu2-broadcast-decline-
            // 20260902, fix #3) because graph_compute() had no i2/i3 loop
            // and could only execute a logically-2D MUL_MAT. graph_compute
            // now loops src0's ne[2]/ne[3]-broadcast form (the GQA K/V-
            // repeat pattern: src0 has fewer head/batch slices than
            // src1/op, each src0 slice reused r2=ne12/ne02,
            // r3=ne13/ne03 times) one NPU matmul per (i3,i2) slice. GGML's
            // own MUL_MAT broadcast contract (ggml_can_mul_mat, asserted
            // at graph-build time) guarantees src1/op's batch dims are
            // exact multiples of src0's whenever this op reached us with
            // src0->ne[2]<src1->ne[2] or src0->ne[3]<src1->ne[3] -- but we
            // still only accept the exact shape graph_compute's loop
            // handles (src0 <= src1/op batch dims, integer ratio, op
            // matches src1) and additionally gate on M (src1->ne[1]) so
            // small-M (decode) broadcasts still decline to CPU -- see
            // RKNPU2_BROADCAST_MIN_M above and the design doc's overhead
            // analysis (per-slice dispatch cost is not worth it at M==1).
            // A *non*-broadcast batch mismatch (src0 batch dims larger
            // than src1's, or a non-integer ratio, or op not matching
            // src1) is not a shape GGML's own MUL_MAT contract produces,
            // but we still decline it defensively rather than assume.
            const bool has_batch = (src0->ne[2] != 1 || src0->ne[3] != 1 ||
                                     src1->ne[2] != 1 || src1->ne[3] != 1 ||
                                     op->ne[2]   != 1 || op->ne[3]   != 1);
            if (has_batch) {
                if (op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3] ||
                    src0->ne[2] > src1->ne[2] || src0->ne[3] > src1->ne[3] ||
                    src1->ne[2] % src0->ne[2] != 0 ||
                    src1->ne[3] % src0->ne[3] != 0 ||
                    src1->ne[1] < RKNPU2_BROADCAST_MIN_M) {
                    return false;
                }
            }

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                return false;
            }

            // rknpu2-broadcast-mulmat-20260902 (fix #6, scope limit): the
            // per-slice buffer-layer fix (get_tensor_packed_size /
            // buffer_set_tensor / buffer_get_tensor, this file) packs each
            // (i3,i2) slice back-to-back but does NOT extend the INT8/INT4
            // per-block `quantized_tensor_scales` indexing (graph_compute's
            // `scales_B_grid[k_idx * num_active_segments + idx]`) to add a
            // per-slice offset, so an INT8/INT4-weight broadcast MUL_MAT
            // would read another slice's dequant scale -- wrong numbers,
            // not a crash. Restrict this fix's broadcast support to the
            // FP16 weight pipeline (the one covering all 24 target FAIL
            // cases in npu_fix3_plan_20260902.md sec 1 / npu_fix6_
            // broadcast_20260902.md) until INT8/INT4 broadcast scale
            // indexing is extended and separately validated. Hadamard
            // pipelines are excluded on the same "not yet extended,
            // decline rather than assume" basis, though in practice this
            // backend only uses Hadamard for INT4/INT8, never FP16.
            if (has_batch && (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16 || pipeline->use_hadamard)) {
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }
        default:
            return false;
    }
}

// Self-healing fd-limit raise (rk3576-emfile-fix-window-20260828). RKNPU2
// allocates one dma-buf handle (and therefore one fd) per distinct tensor
// buffer / matmul-context bind. On the common default 1024 soft
// RLIMIT_NOFILE this is exhausted loading a mid-sized model, producing
// librknnrt's "failed to convert handle to fd, errno 24" -- which, before
// the checks added throughout this file, caused a SIGSEGV rather than a
// clean failure. Try to raise the soft limit here so the backend is
// self-healing without the operator needing to know to run
// `ulimit -n 65536` first. This only ever WIDENS the soft limit up to the
// process's own hard limit (setrlimit without CAP_SYS_RESOURCE cannot
// exceed rlim_max), never lowers it, and any failure here is logged and
// treated as non-fatal: the allocation-failure checks elsewhere in this
// file are the real safety net, this is just a best-effort convenience.
static void rknpu2_maybe_raise_fd_limit() {
    const rlim_t kDesiredSoft = 65536;
    const rlim_t kLowWatermark = 8192;

    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) {
        fprintf(stderr,
            "RKNPU2: getrlimit(RLIMIT_NOFILE) failed (errno=%d: %s) -- "
            "skipping self-raise; if you hit 'failed to convert handle to "
            "fd, errno 24' during inference, run `ulimit -n 65536` before "
            "starting this process\n", errno, strerror(errno));
        return;
    }

    if (rl.rlim_cur == RLIM_INFINITY || rl.rlim_cur >= kLowWatermark) {
        RKNPU2_DBG("fd_limit already sufficient: soft=%lu hard=%lu\n",
            (unsigned long)rl.rlim_cur, (unsigned long)rl.rlim_max);
        return;
    }

    rlim_t target = (rl.rlim_max == RLIM_INFINITY) ? kDesiredSoft
                   : std::min(kDesiredSoft, rl.rlim_max);
    if (target <= rl.rlim_cur) {
        fprintf(stderr,
            "RKNPU2 WARNING: RLIMIT_NOFILE soft=%lu is low and the hard "
            "limit (%lu) does not allow raising it further -- RKNPU2 "
            "allocates one fd per tensor buffer and may hit 'failed to "
            "convert handle to fd, errno 24' on larger models. Ask an "
            "admin to raise the hard limit (e.g. /etc/security/limits.conf "
            "nofile), then `ulimit -n 65536` before starting this process.\n",
            (unsigned long)rl.rlim_cur, (unsigned long)rl.rlim_max);
        return;
    }

    rlim_t old_soft = rl.rlim_cur;
    rl.rlim_cur = target;
    if (setrlimit(RLIMIT_NOFILE, &rl) != 0) {
        fprintf(stderr,
            "RKNPU2 WARNING: setrlimit(RLIMIT_NOFILE, %lu) failed "
            "(errno=%d: %s) -- continuing with the existing soft limit=%lu. "
            "RKNPU2 allocates one fd per tensor buffer and may hit 'failed "
            "to convert handle to fd, errno 24' on larger models; run "
            "`ulimit -n 65536` before starting this process to avoid it.\n",
            (unsigned long)target, errno, strerror(errno), (unsigned long)old_soft);
        return;
    }

    fprintf(stderr,
        "RKNPU2: raised RLIMIT_NOFILE soft limit %lu -> %lu (hard=%lu) to "
        "avoid per-tensor dma-buf fd exhaustion\n",
        (unsigned long)old_soft, (unsigned long)target, (unsigned long)rl.rlim_max);
}

// rk3576-coremask-fix-20260902: this used to default silently to "RK3588"
// whenever RKNPU_DEVICE was unset. On a mixed RK3576/RK3588 fleet that
// silently selects the wrong Rknpu2DeviceConfig (active_cores={0,1,2}
// instead of RK3576's {0,1} -- rknpu2-configuration.cpp) for any process
// launched without an explicit export. compute_n_segments() (above) then
// splits N into config.active_cores.size() segments and assigns
// core_id=active_cores[i] per segment; with the wrongly-selected 3-core
// config, any N large enough to produce a non-empty 3rd segment (real
// model weight matrices -- test-backend-ops' small synthetic N shapes
// mostly don't reach this) makes get_matmul_ctx()'s switch(core_id) above
// pass core_id=2 -> RKNN_NPU_CORE_2 -> mask 4, which the RK3576 NPU
// firmware (2 cores, masks 1/2/3 only) rejects as "Illegal job core_mask
// 4", leaving that job's C-buffer unwritten -- silent garbage, not a
// crash. Confirmed on hardware in the 2026-09-02 evidence window: a shell
// that exported RKNPU_DEVICE=RK3576 before test-backend-ops (fix4_tbo.log)
// logged zero such errors, while llama-bench/llama-perplexity runs in the
// same window that did not (bench_npu_20260902.err, ppl_npu_20260902.log)
// logged 45,440 of them and produced NPU PPL=970949 against CPU PPL=4.997
// -- this was never a code regression between commits, it is a
// launch-environment footgun this backend should not expose. Auto-detect
// from the kernel-provided /proc/device-tree/compatible (present on every
// mainline/vendor RK3576 and RK3588 kernel; the matmul-only SDK header
// bundled here has no chip-identity query) so the backend is correct by
// default; RKNPU_DEVICE remains a valid explicit override. If neither is
// available, fail closed (return NULL, loud stderr) instead of silently
// guessing a chip -- silently running the wrong core topology is exactly
// the failure this closes, not one to leave as a fallback.
static std::string rknpu2_detect_device_from_devicetree() {
    FILE* f = fopen("/proc/device-tree/compatible", "rb");
    if (!f) return "";
    char buf[512];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return "";
    buf[n] = '\0';
    // The file is a sequence of NUL-terminated strings, most-specific
    // board first and SoC family later (e.g. "friendlyarm,nanopi-rk3576\0
    // rockchip,rk3576\0"); scan all of them, don't assume a position.
    for (size_t i = 0; i < n; ) {
        const char* tok = buf + i;
        size_t len = strnlen(tok, n - i);
        std::string s(tok, len);
        if (s.find("rk3576") != std::string::npos) return "RK3576";
        if (s.find("rk3588") != std::string::npos) return "RK3588";
        i += len + 1;
    }
    return "";
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    rknpu2_maybe_raise_fd_limit();

    // Device selection: an explicit RKNPU_DEVICE always wins (dev/debug
    // override); otherwise auto-detect from the device tree; otherwise
    // fail closed instead of guessing (see comment above).
    const char* env_device = std::getenv("RKNPU_DEVICE");
    std::string target_device;
    if (env_device != nullptr && env_device[0] != '\0') {
        target_device = env_device;
    } else {
        target_device = rknpu2_detect_device_from_devicetree();
        if (target_device.empty()) {
            fprintf(stderr,
                "RKNPU2: RKNPU_DEVICE is not set and the NPU chip could not "
                "be auto-detected from /proc/device-tree/compatible -- "
                "refusing to guess (a wrong guess silently selects the "
                "wrong core topology and corrupts NPU output; see "
                "rk3576-coremask-fix-20260902). Set RKNPU_DEVICE=RK3576 or "
                "RKNPU_DEVICE=RK3588 explicitly.\n");
            return NULL;
        }
        fprintf(stderr, "RKNPU2: auto-detected device '%s' from "
            "/proc/device-tree/compatible (set RKNPU_DEVICE to override)\n",
            target_device.c_str());
    }
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(target_device)) {
        fprintf(stderr, "RKNPU2: RKNPU_DEVICE='%s' has no matching device "
            "config registered\n", target_device.c_str());
        return NULL;
    }

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();
    g_active_rknpu_backend_ctx = ctx;

    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    // npu_fix5b_disable_device_20260902: GGML_RKNPU_DISABLE=1 (or
    // RKNPU_DEVICE=none/off/disable) makes this backend enumerate zero
    // devices, so callers that walk ggml_backend_dev_count()/
    // ggml_backend_dev_get() -- including llama.cpp's own model-loading
    // device list -- never see an RKNPU device and no tensor can be placed
    // on it. Unlike `-ngl 0`/`-dev none`, this is not bypassed by this
    // backend registering as GGML_BACKEND_DEVICE_TYPE_ACCEL (see
    // ggml_backend_rknpu_device_get_type below): device-list filtering in
    // llama.cpp only prunes GPU-type offload devices, which is why
    // `-dev none -ngl 0` still allocated a full "RKNPU model buffer" in the
    // fix5_ppl_cpu_v1/v2 evidence -- this gate acts one layer earlier, at
    // enumeration, so no such filtering logic needs to know about ACCEL
    // devices at all.
    if (rknpu2_device_disabled()) {
        return 0;
    }
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    // Defense in depth: get_device_count()==0 already means well-behaved
    // callers never call this with index==0, but guard directly too in case
    // some caller caches an old count or calls get_device() without
    // checking get_device_count() first.
    if (rknpu2_device_disabled()) {
        return NULL;
    }
    if (index != 0) {
        return NULL;
    }

    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif