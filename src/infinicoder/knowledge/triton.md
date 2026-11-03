# Triton implementation and diagnosis

Start from the mathematical contract. Establish input/output shapes, dtype,
strides, storage offsets, alias rules, empty sizes, tolerances and device facts.
Each buffer has its own strides: never reuse an input stride for a contiguous
output. Derive each address from its buffer's layout before changing tiling.

Use `import triton` and `import triton.language as tl`. GPU functions require
`@triton.jit`; launch them from an ordinary Python wrapper using kernel[grid](...).
Keep torch allocation and Python validation outside JIT code. Do not invent
Triton APIs; inspect the installed package or an existing verified example.

Decompose into maps, reductions, gathers/scatters and synchronization. For a
row reduction, one program can own a row when the row fits resource limits;
large reductions may require tiled partial results and a second reduction.
Power-of-two `tl.arange` blocks require masks for tails. Masked sum loads use 0,
masked max loads use negative infinity. Accumulate FP16/BF16 reductions in FP32.
All-masked softmax rows require an explicit contract to avoid inf-inf NaNs.

For matrix products derive M/N/K tiles, operand layouts, accumulator dtype and
tail masks separately. Check divisibility and supported dot dimensions before
using tensor-core paths. Provide a correct fallback for unsupported layouts.
Avoid changing a public interface to fit a seed. Verify every seed precondition.

First get independent randomized correctness checks, then profile. Check
register/shared-memory pressure, launch count and bytes transferred before
choosing block size, warps or stages. Change one factor at a time. Retain only
correct candidates with repeated externally measured improvement. Benchmark
unseen shapes as well as the tuning shapes. A printed RESULT is not evidence.
