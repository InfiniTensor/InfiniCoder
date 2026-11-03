# Framework development workflow

Map the actual call path, data ownership, interfaces and tests before editing.
Read the repository's build configuration and match the installed API versions.
Separate mathematical semantics, state transitions and resource lifetime.

Training: compare loss, gradients and updates against an independent reference.
Check normalization by samples/tokens across accumulation and distributed ranks;
unequal microbatches and ignored labels expose averaging errors. Preserve optimizer
step count, mixed-precision accumulation and collective order. Test zero-work
ranks, exceptions and shutdown. Use small deterministic models for alignment
before a multi-step or multi-GPU run; syntax is not training correctness.

Inference: trace prefill/decode, request admission, KV allocation/reclamation,
prefix sharing, cancellation and output ordering. Check capacity accounting,
stale handles, concurrent release and partial failures. Validate storage offsets,
independent strides, dtype and device at every operator boundary. Preserve
request-local RNG behavior; a global seed does not establish concurrent parity.

System changes: establish lock/ownership invariants, failure propagation and
drain/close semantics. Never hold a global lock across user callbacks or blocking
device operations without a justified invariant. Validate integer bounds before
allocation or pointer arithmetic. Test repeated lifecycle operations and races
with supported sanitizers; absence of a failure is not a race-freedom proof.

Optimization: reproduce an end-to-end bottleneck and a correct baseline. Measure
warmup separately; report latency distribution, throughput and peak memory under
fixed work. Include synchronization and avoid timing asynchronous launch only.
Keep model/data/precision/batching identical. A faster microkernel alone does not
prove training or serving improvement. Re-run numerical and lifecycle checks.
