# Decisions

## 2026-06-04 - Use builtin float for UTM coordinate casting

**Decision**

Use `float` rather than `np.float64` when converting parsed UTM coordinate strings in the SuperVLAD dataset loader.

**Context**

NumPy 2.x removed `np.float`, causing dataset initialization to fail.

**Alternatives considered**

- `np.float64`
- builtin `float`

**Reason**

The NumPy error guidance states that builtin `float` preserves the behavior of the removed alias for this use case.

**Consequences**

Dataset initialization no longer depends on a removed NumPy alias. The resulting NumPy array remains floating point.
