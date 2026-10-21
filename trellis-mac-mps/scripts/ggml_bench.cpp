// Minimal standalone GGML/Metal benchmark harness, closing the llama.cpp baseline gap flagged
// in this project's paper Limitations. Not test-backend-ops (its perf cases don't include this
// paper's exact reference shapes) -- a direct harness against ggml.h/ggml-backend.h/ggml-metal.h,
// at the same two reference shapes used throughout this paper's kernel-level section:
//   1. Linear/matmul at the real to_qkv shape (4096 rows x 1024 in_features -> 3072 out_features)
//   2. Self-attention flash-attn at (H=16, L=4096, D=64), no mask (matching PyTorch/MLX's default
//      non-causal full attention used in the same comparison elsewhere in this project)
//
// Build (from a cloned llama.cpp checkout, GGML_METAL=ON):
//   cmake -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
//     -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
//   cmake --build build --target ggml -j 8
//   clang++ -std=c++17 -O2 -I ggml/include ggml_bench.cpp \
//     -Wl,-force_load,build/ggml/src/libggml.a \
//     -Wl,-force_load,build/ggml/src/libggml-base.a \
//     -Wl,-force_load,build/ggml/src/ggml-metal/libggml-metal.a \
//     -Wl,-force_load,build/ggml/src/libggml-cpu.a \
//     -Wl,-force_load,build/ggml/src/ggml-blas/libggml-blas.a \
//     -framework Metal -framework Foundation -framework Accelerate -framework MetalKit \
//     -o ggml_bench
// Then: ./ggml_bench
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_random_f16(ggml_backend_t backend, ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t n = ggml_nelements(t);
    std::vector<float> f32(n);
    for (size_t i = 0; i < n; i++) f32[i] = dist(rng);
    std::vector<ggml_fp16_t> f16(n);
    ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
    ggml_backend_tensor_set(t, f16.data(), 0, n * sizeof(ggml_fp16_t));
}

static void fill_random_f32(ggml_backend_t backend, ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t n = ggml_nelements(t);
    std::vector<float> f32(n);
    for (size_t i = 0; i < n; i++) f32[i] = dist(rng);
    ggml_backend_tensor_set(t, f32.data(), 0, n * sizeof(float));
}

static void bench_graph(ggml_backend_t backend, ggml_cgraph * graph, const char * label, int n_warmup, int n_trials) {
    for (int i = 0; i < n_warmup; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);

    std::vector<double> samples;
    for (int i = 0; i < n_trials; i++) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        samples.push_back(now_ms() - t0);
    }
    double mean = 0;
    for (double s : samples) mean += s;
    mean /= samples.size();
    double var = 0;
    for (double s : samples) var += (s - mean) * (s - mean);
    double std_ = std::sqrt(var / (samples.size() - 1));
    printf("  %s: %.4f +/- %.4f ms (n=%d)\n", label, mean, std_, n_trials);
}

int main() {
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "Failed to init Metal backend\n");
        return 1;
    }
    printf("Backend: %s\n\n", ggml_backend_name(backend));

    std::mt19937 rng(0);
    const int N_WARMUP = 15;
    const int N_TRIALS = 20;

    // ---------- 1. Linear/matmul at the real to_qkv shape ----------
    {
        printf("=== 1. Linear/matmul at the real to_qkv shape (4096 x 1024 -> 3072) ===\n");
        struct ggml_init_params params = { /*.mem_size=*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(), /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true };
        ggml_context * ctx = ggml_init(params);

        // A = weight [in_features=1024, out_features=3072] (ggml: ne0=k=1024, ne1=n=3072)
        // B = input  [in_features=1024, num_rows=4096]      (ggml: ne0=k=1024, ne1=m=4096)
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 3072);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 4096);
        ggml_set_name(a, "weight");
        ggml_set_name(b, "input");
        ggml_tensor * result = ggml_mul_mat(ctx, a, b);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        fill_random_f16(backend, a, rng);
        fill_random_f16(backend, b, rng);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        bench_graph(backend, graph, "GGML/Metal MUL_MAT (fp16)", N_WARMUP, N_TRIALS);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // ---------- 2. Self-attention flash-attn at the real shape ----------
    {
        printf("\n=== 2. Self-attention flash-attn at the real shape (H=16, L=4096, D=64) ===\n");
        const int64_t hsk = 64, hsv = 64, nh = 16, kv = 4096, nb = 4096;

        struct ggml_init_params params = { /*.mem_size=*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(), /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true };
        ggml_context * ctx = ggml_init(params);

        // q: [hsk, nb, nh, 1] f32 (matches GGML's own test-backend-ops convention for flash_attn_ext)
        // k, v: [hsk/hsv, kv, nh, 1] f16
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, nb, nh, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, kv, nh, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsv, kv, nh, 1);
        ggml_set_name(q, "q");
        ggml_set_name(k, "k");
        ggml_set_name(v, "v");

        float scale = 1.0f / sqrtf((float) hsk);
        // mask = NULL -> no masking, full non-causal self-attention (matches the PyTorch/MLX
        // comparison elsewhere in this project, which also used no mask/is_causal=False).
        ggml_tensor * result = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/ NULL, scale, /*max_bias=*/ 0.0f, /*logit_softcap=*/ 0.0f);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        fill_random_f32(backend, q, rng);
        fill_random_f16(backend, k, rng);
        fill_random_f16(backend, v, rng);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        bench_graph(backend, graph, "GGML/Metal FLASH_ATTN_EXT (q=f32, k/v=f16)", N_WARMUP, N_TRIALS);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
