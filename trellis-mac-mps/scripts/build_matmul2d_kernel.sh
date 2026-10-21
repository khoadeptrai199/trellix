#!/bin/bash
# Compiles trellis/modules/metal_kernels/quantized_linear_matmul2d.metal into a .metallib.
# Requires the full Xcode.app (not just Command Line Tools) with its Metal toolchain component --
# MetalPerformancePrimitives.h isn't resolvable via runtime JIT source compilation, only this
# offline xcrun metal / xcrun metallib path. See quantized_linear_matmul2d.py's module docstring.
#
# One-time setup if the Metal toolchain component isn't installed yet:
#   xcodebuild -runFirstLaunch                  # only if Xcode was installed outside the App Store
#   sudo xcodebuild -license                    # accept the Xcode license (interactive)
#   xcodebuild -downloadComponent MetalToolchain # ~688MB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/../trellis/modules/metal_kernels"
SRC="$KERNEL_DIR/quantized_linear_matmul2d.metal"
AIR="$KERNEL_DIR/quantized_linear_matmul2d.air"
METALLIB="$KERNEL_DIR/quantized_linear_matmul2d.metallib"

if ! xcrun metal --version >/dev/null 2>&1; then
    echo "error: Xcode's Metal toolchain isn't available." >&2
    echo "Run: xcodebuild -runFirstLaunch && sudo xcodebuild -license && xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
fi

echo "Compiling $SRC ..."
xcrun metal -c "$SRC" -o "$AIR"
xcrun metallib "$AIR" -o "$METALLIB"
rm -f "$AIR"
echo "Built $METALLIB"
