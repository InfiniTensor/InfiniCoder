# Python input contracts (teaching, not an implementation)

Conversion is not lexical validation. Python integer conversion accepts
underscores and Unicode digits. If a contract restricts a string grammar,
validate the entire stripped string against that grammar BEFORE conversion.
Check optional signs separately from the nonempty digit sequence. An ASCII
decimal grammar permits only characters '0' through '9'; base 10 conversion
alone does not enforce that restriction. A successful conversion still needs
the declared range check. Reject bool separately when a genuine int is required.

Dependencies may themselves contain defects. Derive defaults from the declared
source of truth; do not trust a helper solely because its name sounds correct.
Preserve unrelated configuration and never mutate defaults to apply overrides.

For flat tensors, different buffers can have different layouts. Use each
buffer's own shape and strides. Validate the highest accessible logical index,
not an assumed allocation size. Empty dimensions address no storage. For
exponentials, shift the participating inputs by their maximum before computing
exponentials; handle an empty participating set explicitly.
