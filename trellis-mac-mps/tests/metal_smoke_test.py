import Metal
import numpy as np
import ctypes

device = Metal.MTLCreateSystemDefaultDevice()
print("Device:", device.name())

src = """
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]],
                     uint id [[thread_position_in_grid]]) {
    buf[id] = buf[id] + 1.0f;
}
"""

options = Metal.MTLCompileOptions.alloc().init()
library, error = device.newLibraryWithSource_options_error_(src, options, None)
if library is None:
    raise RuntimeError(f"Compile failed: {error}")
print("Compiled library OK")

fn = library.newFunctionWithName_("add_one")
pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
if pipeline is None:
    raise RuntimeError(f"Pipeline creation failed: {error}")
print("Pipeline created OK")

N = 1024
data = np.arange(N, dtype=np.float32)
buf = device.newBufferWithBytes_length_options_(data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared)

queue = device.newCommandQueue()
cmdbuf = queue.commandBuffer()
encoder = cmdbuf.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline)
encoder.setBuffer_offset_atIndex_(buf, 0, 0)
grid_size = Metal.MTLSizeMake(N, 1, 1)
tg_size = Metal.MTLSizeMake(min(pipeline.maxTotalThreadsPerThreadgroup(), N), 1, 1)
encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, tg_size)
encoder.endEncoding()
cmdbuf.commit()
cmdbuf.waitUntilCompleted()

ptr = buf.contents()
raw = ptr.as_buffer(data.nbytes)
out = np.frombuffer(raw, dtype=np.float32, count=N).copy()
expected = data + 1.0
print("Match:", np.allclose(out, expected))
print("First 5:", out[:5])
