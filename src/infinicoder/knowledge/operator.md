# 算子工作法（教学材料，不是针对特定算子的答案）

显式记录数学语义、dtype、layout/stride、shape 域、空输入、尾部和异常策略。
先在独立 oracle 下获得正确性，再讨论性能；不能用候选输出的 PASS 或带宽作证。
检查正常、随机、边界、退化、负值、非连续布局和声明支持的跨形状泛化。
对并行代码逐项分析每个线程读写范围、唯一写者、共享存储生命周期、屏障位置。
同步必须匹配真正的数据依赖；重复写相同值也可能是 race，不能以数值近似掩盖。
数值误差后应定位失败 case 的维度、首个偏差、归约顺序与精度，而非盲目加头文件。
CUDA 的 memcheck/racecheck 与数值检查分别验收；其他后端用各自诊断工具。
GPU 计时要求正确的同步/事件；warmup、重复样本、同硬件基线和端到端开销必须公开。
只有通过当前源码、二进制、oracle 与环境身份复核的候选才能进入种子库。

## General tensor and numerical rules

- Separate logical shape, element strides, accessible storage, allocated capacity and copy size. For a nonempty tensor with nonnegative strides, its logical span from its base is `1 + sum((shape[i]-1)*stride[i])`; empty tensors address no elements. Padding after the last logical element is not required unless the contract explicitly says so. These are layout rules for arbitrary rank and computation, not an algorithm implementation.
- Exponential normalization is invariant to a common shift of all active inputs. Choose a shift that bounds exponentials before evaluating them, normalize only the participating elements, and handle an empty participating set explicitly. Arbitrary clipping changes semantics.
- If each output depends on an aggregate, finish computing that aggregate before normalizing any output. A running prefix aggregate is not the final denominator. When a participating subset is compressed, preserve the mapping back to logical indices; an original coordinate is not an index into a compressed list. These rules apply to arbitrary masked reductions and normalization algorithms.
- Reduction coverage and output coverage are separate obligations. When a logical axis exceeds the thread block, each thread may need several elements. Verify the union of addressed indices and the reduction identity, not just the first tile.
- On failure, distinguish a violated declared precondition from an extra precondition invented by the implementation. Preserve valid boundary inputs rather than rejecting them to simplify the kernel.
# Correctness before parallel optimization

When the contract permits PyTorch, first establish correctness with documented
tensor primitives and explicit reduction dimensions. Preserve broadcasting,
dtype and device semantics. Avoid Python loops over individual GPU values.
When a custom GPU kernel is mandatory, start with simple independent device
outputs, then add shared-memory trees, shuffles or vectorized accesses after
correctness. These are development strategies, not performance claims.
When parallelizing, prove that every input contributes once and that only
initialized partial sums are consumed. A lane-count remainder of zero for a
full warp does not mean zero active lanes. All threads required by a barrier
must participate. Preserve the independently verified simple implementation
when testing optimizations; never trade correctness for claimed throughput.
Binary halving reduction trees need a power-of-two participant count unless
the algorithm explicitly preserves unmatched elements. Logical width and
thread-block width are different quantities: lanes without logical input
contribute the reduction identity, while all required lanes reach barriers.
