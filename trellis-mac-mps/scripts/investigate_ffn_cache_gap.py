"""
Investigates the still-open gap noted in APPLE_SILICON.md ("Fused FFN kernel..." section): the
quantized FFN's true pipelined contribution in the real model is 60.1ms for 24 blocks (~2.50ms/
block), roughly 2x fp32's FFN cost, while the isolated compiled-chain microbenchmark predicted
~1.35ms/block (~32ms for 24 blocks). Leading unconfirmed hypothesis: the isolated benchmark reuses
the SAME ~8MB weight set on every call (cache-resident), while the real model calls 24 blocks'
worth of DISTINCT weights (~192MB total) once each per forward -- a fundamentally different memory
access pattern the isolated test couldn't capture.

This script tests that directly: sweep how many *distinct* weight sets the isolated benchmark
cycles through (1, 2, 4, ..., 24) and measure ms/call at each point. If the cache hypothesis is
correct, ms/call should climb as the distinct-set count grows and land near the real model's
~2.50ms/block once the working set reaches the real 24-block/~192MB size. Real shapes (num_rows,
channels, hidden) are captured directly from one instrumented real-model forward pass, not guessed.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.quantized_ffn_matmul2d as qffn

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['sparse_structure_flow_model']
model.eval()

n_quantized = qffn.quantize_model_ffns_matmul2d(model)
print(f"quantized {n_quantized} FeedForwardNet blocks\n")

# Capture real (num_rows, channels, hidden) from one instrumented real forward pass, rather than
# guessing shapes -- avoids the isolated benchmark quietly testing unrealistic dimensions.
captured = {}
orig_forward = qffn.QuantizedFeedForwardNetMatmul2D.forward
def _spy_forward(self, x):
    if 'num_rows' not in captured:
        captured['num_rows'] = x.reshape(-1, self.channels).shape[0]
        captured['channels'] = self.channels
        captured['hidden'] = self.hidden
    return orig_forward(self, x)
qffn.QuantizedFeedForwardNetMatmul2D.forward = _spy_forward

B = 1
x = torch.randn(B, 8, 16, 16, 16, device=device)
t = torch.full((B,), 500.0, device=device)
cond = torch.randn(B, 1374, 1024, device=device)
with torch.no_grad():
    model(x, t, cond)
    torch.mps.synchronize()
qffn.QuantizedFeedForwardNetMatmul2D.forward = orig_forward

num_rows, channels, hidden = captured['num_rows'], captured['channels'], captured['hidden']
weight_mb = 2 * channels * hidden / 1e6
print(f"real shapes captured from model: num_rows={num_rows}, channels={channels}, hidden={hidden}")
print(f"weight bytes/block: {weight_mb:.2f}MB -> x24 blocks = {weight_mb*24:.1f}MB\n")


def make_block():
    w1 = torch.randint(-127, 128, (hidden, channels), dtype=torch.int8, device=device)
    s1 = torch.rand(hidden, device=device) + 0.1
    b1 = torch.randn(hidden, device=device)
    w2 = torch.randint(-127, 128, (channels, hidden), dtype=torch.int8, device=device)
    s2 = torch.rand(channels, device=device) + 0.1
    b2 = torch.randn(channels, device=device)
    return (w1, s1, b1, w2, s2, b2)


with torch.no_grad():
    x2d = torch.randn(num_rows, channels, device=device)
    blocks_24 = [make_block() for _ in range(24)]

    def run_calls(blocks, n_calls):
        for i in range(6):
            b = blocks[i % len(blocks)]
            qffn._compiled_ffn_compute(x2d, b[0], b[1], b[2], b[3], b[4], b[5], channels, hidden)
        torch.mps.synchronize()
        t0 = time.time()
        for i in range(n_calls):
            b = blocks[i % len(blocks)]
            qffn._compiled_ffn_compute(x2d, b[0], b[1], b[2], b[3], b[4], b[5], channels, hidden)
        torch.mps.synchronize()
        return (time.time() - t0) / n_calls * 1000

    N_CALLS = 240
    print("distinct weight sets cycled -> avg ms/call (round-robin, same call count each time):")
    for k in [1, 2, 4, 8, 16, 24]:
        ms = run_calls(blocks_24[:k], N_CALLS)
        print(f"  {k:2d} distinct sets ({k*weight_mb:7.1f}MB working set): {ms:.3f} ms/call")

    def run_full_rounds(blocks, n_rounds):
        for _ in range(3):
            for b in blocks:
                qffn._compiled_ffn_compute(x2d, b[0], b[1], b[2], b[3], b[4], b[5], channels, hidden)
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(n_rounds):
            for b in blocks:
                qffn._compiled_ffn_compute(x2d, b[0], b[1], b[2], b[3], b[4], b[5], channels, hidden)
        torch.mps.synchronize()
        return (time.time() - t0) / n_rounds * 1000

    round_ms = run_full_rounds(blocks_24, 10)
    print(f"\nfull round, real cadence (24 distinct blocks, one call each, in order):")
    print(f"  {round_ms:.2f} ms/round = {round_ms/24:.3f} ms/block")
    print(f"\ncompare:")
    print(f"  real model measured FFN true contribution (APPLE_SILICON.md): 60.1ms/round = 2.50ms/block")
    print(f"  isolated same-weights microbenchmark (APPLE_SILICON.md):      ~32ms/round  = ~1.35ms/block")
