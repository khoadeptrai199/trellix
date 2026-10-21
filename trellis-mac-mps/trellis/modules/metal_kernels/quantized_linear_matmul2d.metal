// Metal 4 Performance Primitives (matmul2d) implementation of int8-weight-quantized nn.Linear.
// Requires the Xcode offline Metal compiler (xcrun metal / xcrun metallib) -- MetalPerformancePrimitives.h
// does not resolve under runtime JIT source compilation (MTLDevice.newLibraryWithSource_options_error_),
// confirmed empirically. See scripts/build_matmul2d_kernel.sh for the build step and
// trellis/modules/quantized_linear_matmul2d.py's module docstring for the full story (why this exists,
// what it replaces, and the tensor-layout convention below that took real trial and error to find).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Dequantizes int8 weights -> half, in place in the *original* [out_features, in_features] nn.Linear
// layout -- no transpose needed, since matmul2d's right operand with transposeRight=true consumes
// exactly that layout directly (see the layout note on matmul2d_linear below).
kernel void dequantize_weights(
    device const char* w_int8   [[buffer(0)]],
    device const float* scale   [[buffer(1)]],
    device half* w_half         [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint out_idx = gid.y;
    uint in_idx = gid.x;
    if (in_idx >= in_features || out_idx >= out_features) return;
    float v = float(w_int8[out_idx * in_features + in_idx]) * scale[out_idx];
    // Guard against the float32->half narrowing overflowing to +/-inf (half's max magnitude is
    // 65504) -- trained weights are never expected to reach this, but a silent inf here would
    // poison the entire matmul2d output through normal IEEE inf propagation, so clamp rather than
    // trust the assumption.
    w_half[out_idx * in_features + in_idx] = half(clamp(v, -65000.0f, 65000.0f));
}

// Layout convention (found empirically -- undocumented in the one public example available at the
// time this was written): MTLTensor dextents are (width, height) = (fastest-varying dim, slowest),
// the OPPOSITE of numpy/C row-major (rows, cols) shape order. A buffer holding a row-major
// [rows, cols] array should be declared dextents(cols, rows) -- "width" = row length.
//
// With that: X [num_rows, in_features] (row-major) -> dextents(in_features, num_rows).
// W is the *natural* nn.Linear weight [out_features, in_features] (row-major), declared
// dextents(in_features, out_features) -- i.e. read as if it were [in_features, out_features] --
// with transposeRight=true telling matmul2d to treat it as the transpose of that, recovering the
// true [out_features, in_features] semantics. This means no separate weight-transpose kernel is
// needed at all; dequantize_weights above can leave the natural layout untouched.
// Y [num_rows, out_features] (row-major) -> dextents(out_features, num_rows).
kernel void matmul2d_linear(
    device half* x_buf          [[buffer(0)]],
    device half* w_buf          [[buffer(1)]],
    device float* y_buf         [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& in_features  [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device half, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device half, dextents<int32_t, 2>, tensor_inline>(w_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// Same as matmul2d_linear, but W is the RAW int8 quantized weight buffer, not a pre-dequantized
// half copy -- no dequantize_weights pass needed at all. Confirmed directly from Apple's own
// MPPTensorOpsMatMul2d.h (not inferred): matmul2d natively supports a `half` left operand times an
// `int8_t` right operand producing `float`, doing the int8->float widen internally during its own
// tile load. This avoids materializing a full half-precision copy of the weight matrix in global
// memory (dequantize_weights' entire job) -- one fewer kernel dispatch AND, more importantly, one
// fewer full read+write of the weight matrix through global memory per layer.
//
// This computes the UNSCALED int8 matmul only (X @ int8_W^T, no per-row scale or bias applied) --
// matmul2d has no notion of our per-output-channel quantization scale, so it can't fold that in.
// That's fine: our scale is per-OUTPUT-channel only (from_float's row-wise symmetric quantization),
// so it factors exactly out of the reduction sum: Y[r,o] = scale[o] * sum_i(X[r,i] * int8_W[o,i]).
// Apply scale (and bias) as a cheap follow-up elementwise pass -- see scale_bias_f32 below.
kernel void matmul2d_linear_quantized(
    device half* x_buf          [[buffer(0)]],
    device int8_t* w_int8_buf   [[buffer(1)]],  // [out_features, in_features], raw quantized weights
    device float* y_buf         [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& in_features  [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device half, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(w_int8_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// Identical to matmul2d_linear_quantized above, except relaxed_precision=true in the descriptor.
// Apple's own MPPTensorOpsMatMul2d.h documents this flag directly: "set it to true to allow
// implementation to sacrifice accuracy for performance." Added as a new kernel (not editing
// matmul2d_linear_quantized in place) per this port's own versioning discipline.
//
// MEASURED, NOT SPECULATIVE (scripts/investigate_relaxed_precision.py, on Apple M5 Max): a clean
// negative result. Output is BYTE-IDENTICAL to matmul2d_linear_quantized's (relaxed_precision=false)
// across every tested shape (max abs diff between the two kernels: exactly 0.000000) and speed is
// within noise (1.003x). For this specific `half x int8_t -> float` operand combination, on this
// hardware/SDK, relaxed_precision has no effect -- there is apparently no lower-precision internal
// path left to switch to beyond what the int8 operand and float32 accumulator already are. Kept in
// place (unused by any module) rather than deleted, per this port's convention, in case a future
// SDK/hardware revision changes this for this specific dtype combination -- re-run the
// investigate script rather than assuming this result still holds if that's ever suspected.
kernel void matmul2d_linear_quantized_relaxed(
    device half* x_buf          [[buffer(0)]],
    device int8_t* w_int8_buf   [[buffer(1)]],  // [out_features, in_features], raw quantized weights
    device float* y_buf         [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& in_features  [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device half, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(w_int8_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// --- Experimental: does the OPERAND TYPE COMBINATION itself matter, not just relaxed_precision?
// matmul2d supports many combinations (see the table at the top of MPPTensorOpsMatMul2d.h) --
// these two test a fully-int8 path (both operands int8, int32 accumulator, no int8->half widen at
// all) and a plain half x half path (proxy for what native fp16 matmul might look like if routed
// through matmul2d instead of MPSGraph), to see whether either beats matmul2d_linear_quantized's
// half x int8_t -> float combination. Investigation-only, not wired into any production module
// yet -- see scripts/investigate_matmul2d_dtype_combos.py for the correctness + speed comparison.

// Fully int8: both X and W are int8, accumulator is int32 (no float/half involved in the GEMM
// itself at all). If this model's activations were also quantized to int8 (not just the weights,
// which is all this port does today), this is the combination that would actually get used.
kernel void matmul2d_int8_int8(
    device int8_t* x_int8_buf   [[buffer(0)]],
    device int8_t* w_int8_buf   [[buffer(1)]],  // [out_features, in_features], raw quantized weights
    device int* y_buf           [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& in_features  [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(x_int8_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(w_int8_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device int, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// Plain half x half -> float, both operands unquantized. Proxy for "what does matmul2d itself do
// with the exact same dtypes native nn.Linear already uses," isolating whether any gap is about
// the int8 operand specifically or about matmul2d vs MPSGraph's own GEMM in general.
kernel void matmul2d_half_half(
    device half* x_buf          [[buffer(0)]],
    device half* w_buf          [[buffer(1)]],  // [out_features, in_features]
    device float* y_buf         [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& in_features  [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device half, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device half, dextents<int32_t, 2>, tensor_inline>(w_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// bfloat x bfloat -> float. bfloat has float32's exponent range (unlike half, whose limited range
// is exactly why _ffn_compute/_self_attn_compute/_cross_attn_compute all carry a dynamic amax()
// rescale before every half cast -- this model's real activations span single digits to 8000+).
// If bfloat matches or beats half's matmul2d speed, it could eliminate that rescale overhead
// entirely, independent of any int8 question. Native Metal `bfloat` type, no packing ambiguity.
kernel void matmul2d_bfloat_bfloat(
    device bfloat* x_buf         [[buffer(0)]],
    device bfloat* w_buf         [[buffer(1)]],  // [out_features, in_features]
    device float* y_buf          [[buffer(2)]],
    constant uint& num_rows      [[buffer(3)]],
    constant uint& in_features   [[buffer(4)]],
    constant uint& out_features  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(w_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// bfloat activation x int8_t (raw quantized) weight -> float. Same weight-only-quantization shape
// as the production matmul2d_linear_quantized kernel, but with bfloat instead of half for the
// activation operand -- tests whether the "half operand is the bottleneck" finding is specific to
// `half`, or a property of matmul2d with ANY 16-bit-or-narrower left operand (bfloat included).
kernel void matmul2d_bfloat_int8(
    device bfloat* x_buf         [[buffer(0)]],
    device int8_t* w_int8_buf    [[buffer(1)]],  // [out_features, in_features], raw quantized weights
    device float* y_buf          [[buffer(2)]],
    constant uint& num_rows      [[buffer(3)]],
    constant uint& in_features   [[buffer(4)]],
    constant uint& out_features  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(w_int8_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// half x half -> half (destination half, not float): smaller output write than every other test
// kernel here (2 bytes/element vs. 4) -- tests whether the destination write width, not just the
// operand types, contributes to the gap against native fp16 Linear (which also writes half).
kernel void matmul2d_half_half_half(
    device half* x_buf          [[buffer(0)]],
    device half* w_buf          [[buffer(1)]],  // [out_features, in_features]
    device half* y_buf          [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& in_features  [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto X = tensor<device half, dextents<int32_t, 2>, tensor_inline>(x_buf, dextents<int32_t, 2>(in_features, num_rows));
    auto W = tensor<device half, dextents<int32_t, 2>, tensor_inline>(w_buf, dextents<int32_t, 2>(in_features, out_features));
    auto Y = tensor<device half, dextents<int32_t, 2>, tensor_inline>(y_buf, dextents<int32_t, 2>(out_features, num_rows));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mX = X.slice(0, tgid.y * 64);
    auto mW = W.slice(0, tgid.x * 32);
    auto mY = Y.slice(tgid.x * 32, tgid.y * 64);
    op.run(mX, mW, mY);
}

// Applies matmul2d_linear_quantized's deferred per-output-channel scale, and optionally bias, to
// its raw int8-matmul output, in place. Kept as a separate simple elementwise pass rather than
// fused into a GELU/etc kernel -- see quantized_ffn_matmul2d.py's module docstring for why
// matmul2d's output specifically should only ever be consumed by plain, non-custom-kernel ops
// downstream (a real NaN was traced to matmul2d's second call in a chain misbehaving when fed a
// buffer written by another custom kernel; this kernel is a safe *producer* into that same
// contract, not a consumer of another custom kernel's output).
kernel void scale_bias_f32(
    device float* y             [[buffer(0)]],
    device const float* scale   [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    constant uint& has_bias     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out_features || gid.y >= num_rows) return;
    uint idx = gid.y * out_features + gid.x;
    float v = y[idx] * scale[gid.x];
    if (has_bias != 0) v += bias[gid.x];
    y[idx] = v;
}

// --- Fusion kernels for QuantizedFeedForwardNetMatmul2D (fc1 -> GELU -> fc2 chained on one
// command buffer, see quantized_ffn_matmul2d.py) --------------------------------------------

// Applied to fc1's raw matmul2d output (float32, no bias yet): adds bias, applies GELU
// (tanh approximation, matching nn.GELU(approximate="tanh")), and casts down to half so the
// result is immediately usable as the next matmul2d call's X operand -- keeps the whole
// fc1 -> activation -> fc2 chain on the GPU with no CPU readback in between.
kernel void bias_gelu_to_half(
    device const float* y_in    [[buffer(0)]],
    device const float* bias    [[buffer(1)]],
    device half* x_out          [[buffer(2)]],
    constant uint& num_rows     [[buffer(3)]],
    constant uint& hidden       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= hidden || gid.y >= num_rows) return;
    uint idx = gid.y * hidden + gid.x;
    float v = y_in[idx] + bias[gid.x];
    float x3 = v * v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * x3);  // sqrt(2/pi)
    float g = 0.5f * v * (1.0f + tanh(inner));
    // Real-model activations (measured on sparse_structure_flow_model) reach |v| in the thousands
    // at some blocks/timesteps -- GELU(v) ~= v for large positive v, so g can exceed half's max
    // representable magnitude (65504) and silently become +/-inf, which then poisons fc2's entire
    // output through inf propagation. Clamp to comfortably inside half's range rather than assume
    // activations stay small; this only affects the rare outlier element, everything else in the
    // normal operating range is untouched.
    x_out[idx] = half(clamp(g, -65000.0f, 65000.0f));
}

// Bias add for fc2's raw matmul2d output, in place, float32 -- kept on-GPU so the whole
// fused FFN needs only one final waitUntilCompleted()/readback.
kernel void bias_add_f32(
    device float* y             [[buffer(0)]],
    device const float* bias    [[buffer(1)]],
    constant uint& num_rows     [[buffer(2)]],
    constant uint& out_features [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out_features || gid.y >= num_rows) return;
    y[gid.y * out_features + gid.x] += bias[gid.x];
}

// --- Batched multi-head attention matmuls: head dimension baked into the dispatch grid, not
// looped in Python -----------------------------------------------------------------------------
//
// scripts/investigate_attention_dispatch_overhead.py-adjacent finding (see APPLE_SILICON.md,
// "Beyond Linear quantization" section): looping matmul2d over 16 heads in Python for Q@K^T
// measured 2.8x SLOWER than native torch.bmm, worse than the ~1.9x gap already found for Linear
// shapes -- root cause was NOT the GEMM itself, it was that each Python-level call is a separate
// command-buffer dispatch (16 of them), while torch.bmm's native kernel handles all heads in ONE
// dispatch via a 3D threadgroup grid. These two kernels fix that specific problem: the head index
// lives in tgid.z, so ALL heads execute as a single dispatchThreads call from the PyTorch side --
// exactly the "grid natively includes the head dimension" fix, not a different way of calling the
// same per-head kernel.
//
// This does NOT attempt online-softmax/flash-attention tiling (avoiding materializing the full
// L x L score matrix) -- softmax stays a native torch op between these two kernels, isolating
// "does proper batching alone close the gap" as its own testable question before taking on the
// much harder (and much more failure-prone -- numerical stability of running max/sum, tile
// bookkeeping) online-softmax tiling work. See scripts/investigate_batched_attention_matmul2d.py
// for the correctness + speed comparison against native scaled_dot_product_attention.

// Q@K^T for all heads in one dispatch. Q, K: [H, L, D] (row-major, contiguous per head).
// Output: scores [H, L, L] (float32, unscaled -- the 1/sqrt(D) scale is applied by the native
// elementwise op that follows, same "native ops only between matmul2d calls" rule as everywhere
// else in this port).
kernel void attention_qkt_batched(
    device half* q_buf          [[buffer(0)]],
    device half* k_buf          [[buffer(1)]],
    device float* scores_buf    [[buffer(2)]],
    constant uint& L            [[buffer(3)]],
    constant uint& D            [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint h = tgid.z;
    device half* q_h = q_buf + h * L * D;
    device half* k_h = k_buf + h * L * D;
    device float* s_h = scores_buf + h * L * L;

    auto Q = tensor<device half, dextents<int32_t, 2>, tensor_inline>(q_h, dextents<int32_t, 2>(D, L));
    auto K = tensor<device half, dextents<int32_t, 2>, tensor_inline>(k_h, dextents<int32_t, 2>(D, L));
    auto S = tensor<device float, dextents<int32_t, 2>, tensor_inline>(s_h, dextents<int32_t, 2>(L, L));

    // Same convention as matmul2d_half_half: X=Q [num_rows=L, in=D], W=K [out=L, in=D],
    // transposeRight=true recovers Q @ K^T -> [L, L]. Verified against the per-head Python-loop
    // version this replaces (matmul2d_half_half via scripts/investigate_matmul2d_dtype_combos.py's
    // QK^T test), same math, only the dispatch structure changes.
    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mQ = Q.slice(0, tgid.y * 64);
    auto mK = K.slice(0, tgid.x * 32);
    auto mS = S.slice(tgid.x * 32, tgid.y * 64);
    op.run(mQ, mK, mS);
}

// probs @ V for all heads in one dispatch. probs: [H, L, L] (half, already softmax'd and cast down
// by the native op between the two kernels). V: [H, L, D]. Output: [H, L, D] (float32).
// No transpose here (unlike the Linear kernels and attention_qkt_batched above) -- V's natural
// row-major [L, D] layout already IS the [K=L, N=D] shape probs @ V needs, verified empirically
// against a CPU fp32 reference before trusting it (see the investigate script).
kernel void attention_av_batched(
    device half* probs_buf      [[buffer(0)]],
    device half* v_buf          [[buffer(1)]],
    device float* out_buf       [[buffer(2)]],
    constant uint& L            [[buffer(3)]],
    constant uint& D            [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint h = tgid.z;
    device half* p_h = probs_buf + h * L * L;
    device half* v_h = v_buf + h * L * D;
    device float* o_h = out_buf + h * L * D;

    auto P = tensor<device half, dextents<int32_t, 2>, tensor_inline>(p_h, dextents<int32_t, 2>(L, L));
    auto V = tensor<device half, dextents<int32_t, 2>, tensor_inline>(v_h, dextents<int32_t, 2>(D, L));
    auto O = tensor<device float, dextents<int32_t, 2>, tensor_inline>(o_h, dextents<int32_t, 2>(D, L));

    constexpr auto d = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;

    auto mP = P.slice(0, tgid.y * 64);
    auto mV = V.slice(tgid.x * 32, 0);
    auto mO = O.slice(tgid.x * 32, tgid.y * 64);
    op.run(mP, mV, mO);
}

// --- Real flash-attention-style fused kernel: online-softmax tiling, no full L x L device-memory
// materialization -------------------------------------------------------------------------------
//
// Follow-up to attention_qkt_batched/attention_av_batched above, which fixed the dispatch-count
// problem (batching heads into one dispatch) but measured almost no speedup (15.04ms vs the
// per-head loop's 15.28ms, both ~3x slower than native SDPA's 5.0ms) -- falsifying the hypothesis
// that dispatch count was the dominant cost. Leading remaining hypothesis, tested here: those two
// kernels still write and re-read the FULL per-head L x L score matrix through device memory
// (16.7M elements/head, 268M across 16 heads) -- exactly what real flash-attention avoids via
// online-softmax tiling. This kernel keeps the score tile, and the running softmax statistics
// (m: running max, l: running sum, acc: running weighted output), in THREADGROUP (on-chip) memory
// instead of device memory, processing one KV block (32 columns) at a time.
//
// Not using metal::cooperative_tensor's full register-resident chaining (get_left_input_
// cooperative_tensor / get_right_input_cooperative_tensor), despite it being the more
// sophisticated mechanism the header documents for this exact purpose (avoiding a device-memory
// round trip between chained matmul2d calls) -- its per-thread data layout is implementation-
// defined and opaque (accessed only via get_multidimensional_index/get_mask/iterators), which is
// real correctness risk for a first attempt. Threadgroup memory is a plain, non-opaque array --
// lower risk, still avoids the device-memory round trip that's the actual suspected bottleneck.
// One threadgroup handles one (head, 64-row query block) pair, looping internally over all KV
// blocks -- so the head AND the KV loop are both handled without any Python-level looping,
// unlike either of the two kernels above.
//
// Softmax scale (1/sqrt(D)) is applied inline when reading raw scores, not pre-baked into Q --
// keeps Q's device-memory contents untouched (shared read-only across threadgroups) rather than
// risking a race on an in-place rescale.
//
// D is assumed exactly 64 (this model's real head_dim, confirmed earlier this session) --
// hardcoded, not templated, to keep threadgroup memory sizing simple and predictable
// (score_tile + m + l + alpha + acc ~= 24.75KB, under the ~32KB per-threadgroup budget).
kernel void attention_flash_online_softmax(
    device half* q_buf          [[buffer(0)]],  // [H, L, D]
    device half* k_buf          [[buffer(1)]],  // [H, L, D]
    device half* v_buf          [[buffer(2)]],  // [H, L, D]
    device float* out_buf       [[buffer(3)]],  // [H, L, D]
    constant uint& L            [[buffer(4)]],
    constant float& scale       [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint D = 64;
    constexpr uint QBLOCK = 64;
    constexpr uint KVBLOCK = 32;

    uint h = tgid.z;
    uint q_row0 = tgid.y * QBLOCK;

    device half* q_h = q_buf + h * L * D;
    device half* k_h = k_buf + h * L * D;
    device half* v_h = v_buf + h * L * D;
    device float* out_h = out_buf + h * L * D;

    threadgroup float score_tile[QBLOCK * KVBLOCK];  // 8KB
    threadgroup float acc[QBLOCK * D];                // 16KB
    threadgroup float m[QBLOCK];                      // running max, 256B
    threadgroup float l[QBLOCK];                      // running sum, 256B
    // (no separate `alpha` array -- each lane keeps its own `a` in a register, per-row rescale
    // factor is never needed outside the lane pair that just computed it)

    if (tid < QBLOCK) {
        m[tid] = -INFINITY;
        l[tid] = 0.0f;
    }
    // Grid-stride zero-init across all 128 threads (was a 64-iteration serial loop on 64 threads,
    // with the other 64 idle -- see the parallelization note below).
    for (uint idx = tid; idx < QBLOCK * D; idx += 128) acc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto Q = tensor<device half, dextents<int32_t, 2>, tensor_inline>(q_h, dextents<int32_t, 2>(D, L));
    constexpr auto qk_desc = matmul2d_descriptor(QBLOCK, KVBLOCK, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    auto mQ = Q.slice(0, q_row0);

    constexpr auto av_desc = matmul2d_descriptor(QBLOCK, D, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<av_desc, execution_simdgroups<4>> av_op;

    for (uint kv0 = 0; kv0 < L; kv0 += KVBLOCK) {
        // Real bug, found and fixed via scripts/investigate_flash_attention_matmul2d.py's small-
        // shape correctness check (H=3,L=100 failed; H=1,L=64 and H=2,L=96 -- both multiples of
        // KVBLOCK -- passed, which is exactly the signature of a tail-block boundary bug):
        // matmul2d's own edge-checking correctly avoids READING past k_h/v_h's true L extent for
        // a partial final KV block, but it only WRITES the valid columns of score_tile -- the
        // remaining columns keep whatever was in threadgroup memory from the previous iteration
        // (or uninitialized, on the first). The bookkeeping loop below used to read all KVBLOCK
        // columns unconditionally, corrupting local_max/local_sum with stale data for any L not a
        // multiple of KVBLOCK=32. Fix: only read the `kv_valid` real columns, and explicitly zero
        // the rest (not just skip them) -- av_op.run() below still reads the FULL score_tile as
        // its left operand every iteration (it's a fixed-size threadgroup buffer, not sliced from
        // a larger tensor matmul2d can edge-check against), so the invalid columns must genuinely
        // hold 0.0 (contributing nothing to the accumulator), not just be left alone.
        uint kv_valid = min((uint)KVBLOCK, L - kv0);

        auto K = tensor<device half, dextents<int32_t, 2>, tensor_inline>(k_h, dextents<int32_t, 2>(D, L));
        auto S = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(score_tile, dextents<int32_t, 2>(KVBLOCK, QBLOCK));
        auto mK = K.slice(0, kv0);
        qk_op.run(mQ, mK, S);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online-softmax bookkeeping, parallelized across ALL 128 threads (2 lanes per row) via
        // simd_shuffle_xor, not the single-thread-per-row serial version this replaces (which left
        // 64 of 128 threads idle during this phase and did ~128 serial scalar ops per active
        // thread per KV block -- flagged as the leading suspect for the remaining gap against
        // native SDPA in APPLE_SILICON.md). Each simdgroup (32 lanes) covers 16 rows, 2 lanes/row;
        // partner lanes are (lane, lane^1) within the same simdgroup -- consecutive pairs, matching
        // row_local = lane/2 -- so simd_shuffle_xor(x, 1) exchanges exactly the right partner's
        // value. Each lane handles half of KVBLOCK's columns (16) and half of D's accumulator
        // elements (32), reducing serial work per lane by ~2x on top of removing the idle threads.
        {
            uint simdgroup_idx = tid / 32;
            uint lane = tid % 32;
            uint row_local = lane / 2;             // 0-15
            uint sub = lane % 2;                   // 0 or 1 -- which half of the row this lane owns
            uint r = simdgroup_idx * 16 + row_local;  // 0-63, global row within this query block
            uint col_start = sub * (KVBLOCK / 2);
            uint col_end = col_start + (KVBLOCK / 2);

            float local_max = -INFINITY;
            for (uint c = col_start; c < min(col_end, kv_valid); ++c) {
                float s = score_tile[r * KVBLOCK + c] * scale;
                local_max = max(local_max, s);
            }
            local_max = max(local_max, simd_shuffle_xor(local_max, 1));

            float new_max = max(m[r], local_max);
            float a = exp(m[r] - new_max);  // exp(-inf - finite) = 0 correctly on the first block

            float local_sum = 0.0f;
            for (uint c = col_start; c < col_end; ++c) {
                if (c < kv_valid) {
                    float s = score_tile[r * KVBLOCK + c] * scale;
                    float p = exp(s - new_max);
                    score_tile[r * KVBLOCK + c] = p;
                    local_sum += p;
                } else {
                    score_tile[r * KVBLOCK + c] = 0.0f;
                }
            }
            local_sum += simd_shuffle_xor(local_sum, 1);

            // Only one lane per row writes the shared per-row state -- both lanes computed the
            // identical new_max/a/local_sum after the shuffle-reduce above, so this isn't a race
            // on VALUE, but two unguarded writes to the same address is still undefined-order and
            // unnecessary; guard it explicitly rather than rely on both writes being harmless.
            if (sub == 0) {
                l[r] = l[r] * a + local_sum;
                m[r] = new_max;
            }

            // acc rescale: each lane owns half of this row's D=64 accumulator elements, no
            // reduction needed here (elementwise, not a sum) -- just split the work.
            for (uint c = sub * (D / 2); c < sub * (D / 2) + (D / 2); ++c) {
                acc[r * D + c] *= a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto V = tensor<device half, dextents<int32_t, 2>, tensor_inline>(v_h, dextents<int32_t, 2>(D, L));
        auto P = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(score_tile, dextents<int32_t, 2>(KVBLOCK, QBLOCK));
        auto ACC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(acc, dextents<int32_t, 2>(D, QBLOCK));
        auto mV = V.slice(0, kv0);
        av_op.run(P, mV, ACC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final output write, parallelized across all 128 threads the same way as the acc rescale
    // above (2 lanes/row, D/2 elements each) -- was 64-threads-active/64-idle before.
    {
        uint simdgroup_idx = tid / 32;
        uint lane = tid % 32;
        uint row_local = lane / 2;
        uint sub = lane % 2;
        uint r = simdgroup_idx * 16 + row_local;
        uint global_row = q_row0 + r;
        if (global_row < L) {
            for (uint c = sub * (D / 2); c < sub * (D / 2) + (D / 2); ++c) {
                out_h[global_row * D + c] = acc[r * D + c] / l[r];
            }
        }
    }
}

// Tile-size variant of attention_flash_online_softmax: KVBLOCK=16 instead of 32 (everything else
// identical -- same 2-lanes-per-row parallelized bookkeeping). Smaller KV block means more outer-
// loop iterations (L/16=256 vs L/32=128 for L=4096) but a much smaller score_tile (4KB vs 8KB),
// leaving more threadgroup-memory headroom -- untested whether fewer-but-bigger or more-but-
// smaller matmul2d dispatches wins on this hardware; see scripts/investigate_flash_attention_
// matmul2d.py for the direct comparison against the KVBLOCK=32 version above. Not a fundamentally
// different kernel, so no separate correctness-bug risk expected, but still run through the same
// small-shape correctness check before trusting any speed number, per this port's own discipline.
kernel void attention_flash_online_softmax_kv16(
    device half* q_buf          [[buffer(0)]],  // [H, L, D]
    device half* k_buf          [[buffer(1)]],  // [H, L, D]
    device half* v_buf          [[buffer(2)]],  // [H, L, D]
    device float* out_buf       [[buffer(3)]],  // [H, L, D]
    constant uint& L            [[buffer(4)]],
    constant float& scale       [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint D = 64;
    constexpr uint QBLOCK = 64;
    constexpr uint KVBLOCK = 16;

    uint h = tgid.z;
    uint q_row0 = tgid.y * QBLOCK;

    device half* q_h = q_buf + h * L * D;
    device half* k_h = k_buf + h * L * D;
    device half* v_h = v_buf + h * L * D;
    device float* out_h = out_buf + h * L * D;

    threadgroup float score_tile[QBLOCK * KVBLOCK];  // 4KB
    threadgroup float acc[QBLOCK * D];                // 16KB
    threadgroup float m[QBLOCK];
    threadgroup float l[QBLOCK];

    if (tid < QBLOCK) {
        m[tid] = -INFINITY;
        l[tid] = 0.0f;
    }
    for (uint idx = tid; idx < QBLOCK * D; idx += 128) acc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto Q = tensor<device half, dextents<int32_t, 2>, tensor_inline>(q_h, dextents<int32_t, 2>(D, L));
    constexpr auto qk_desc = matmul2d_descriptor(QBLOCK, KVBLOCK, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    auto mQ = Q.slice(0, q_row0);

    constexpr auto av_desc = matmul2d_descriptor(QBLOCK, D, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<av_desc, execution_simdgroups<4>> av_op;

    for (uint kv0 = 0; kv0 < L; kv0 += KVBLOCK) {
        uint kv_valid = min((uint)KVBLOCK, L - kv0);

        auto K = tensor<device half, dextents<int32_t, 2>, tensor_inline>(k_h, dextents<int32_t, 2>(D, L));
        auto S = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(score_tile, dextents<int32_t, 2>(KVBLOCK, QBLOCK));
        auto mK = K.slice(0, kv0);
        qk_op.run(mQ, mK, S);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint simdgroup_idx = tid / 32;
            uint lane = tid % 32;
            uint row_local = lane / 2;
            uint sub = lane % 2;
            uint r = simdgroup_idx * 16 + row_local;
            uint col_start = sub * (KVBLOCK / 2);
            uint col_end = col_start + (KVBLOCK / 2);

            float local_max = -INFINITY;
            for (uint c = col_start; c < min(col_end, kv_valid); ++c) {
                float s = score_tile[r * KVBLOCK + c] * scale;
                local_max = max(local_max, s);
            }
            local_max = max(local_max, simd_shuffle_xor(local_max, 1));

            float new_max = max(m[r], local_max);
            float a = exp(m[r] - new_max);

            float local_sum = 0.0f;
            for (uint c = col_start; c < col_end; ++c) {
                if (c < kv_valid) {
                    float s = score_tile[r * KVBLOCK + c] * scale;
                    float p = exp(s - new_max);
                    score_tile[r * KVBLOCK + c] = p;
                    local_sum += p;
                } else {
                    score_tile[r * KVBLOCK + c] = 0.0f;
                }
            }
            local_sum += simd_shuffle_xor(local_sum, 1);

            if (sub == 0) {
                l[r] = l[r] * a + local_sum;
                m[r] = new_max;
            }

            for (uint c = sub * (D / 2); c < sub * (D / 2) + (D / 2); ++c) {
                acc[r * D + c] *= a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto V = tensor<device half, dextents<int32_t, 2>, tensor_inline>(v_h, dextents<int32_t, 2>(D, L));
        auto P = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(score_tile, dextents<int32_t, 2>(KVBLOCK, QBLOCK));
        auto ACC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(acc, dextents<int32_t, 2>(D, QBLOCK));
        auto mV = V.slice(0, kv0);
        av_op.run(P, mV, ACC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        uint simdgroup_idx = tid / 32;
        uint lane = tid % 32;
        uint row_local = lane / 2;
        uint sub = lane % 2;
        uint r = simdgroup_idx * 16 + row_local;
        uint global_row = q_row0 + r;
        if (global_row < L) {
            for (uint c = sub * (D / 2); c < sub * (D / 2) + (D / 2); ++c) {
                out_h[global_row * D + c] = acc[r * D + c] / l[r];
            }
        }
    }
}
