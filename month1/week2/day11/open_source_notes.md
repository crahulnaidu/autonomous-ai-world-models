# tinygrad Investigation Notes

**File:** `open_source_notes.md`

## What I looked at:

* **Core Autograd Engine:** Examined `tinygrad/tensor.py` (the `Tensor` class and its `.backward()` method).
* **Operation Tracking:** Investigated `tinygrad/ops.py` and `tinygrad/dtype.py` to see how low-level operations (LazyOps, UnaryOps, BinaryOps, ReduceOps) are declared and lazily chained.
* **Gradient Computation Graphs:** Inspected how `Function` subclasses in `tinygrad/mlops.py` define forward passes alongside their corresponding symbolic backward passes.

---

## How tinygrad represents operations:

* **Lazy Evaluation via Computation Graphs:** Unlike NumPy (which executes math eagerly in memory), tinygrad builds a **Directed Acyclic Graph (DAG)** of `LazyOp` nodes. Math is only compiled and executed when `.realize()` or `.numpy()` is explicitly called.
* **Primitive Classification:** Operations are reduced to a minimal set of core primitives categorized into Unary, Binary, Ternary, Reduce, and Movement ops (like `reshape`, `permute`, `expand`).
* **Tensor Wrapper:** A `Tensor` wraps a `LazyBuffer` (or context object), holding metadata like `requires_grad`, `grad`, and a reference to the `Function` that created it.

---

## How gradients are propagated:

* **Reverse-Mode Autograd:** When `tensor.backward()` is invoked, tinygrad computes topological sort orders of the execution graph starting from the scalar loss node.
* **Chain Rule Traversal:** It traverses nodes in reverse order. Each operation's `backward()` function calculates the incoming gradient scaled by its partial derivative and passes it to its parents (`self._ctx.parents`).
* **Gradient Accumulation:** If a node has multiple downstream consumers, intermediate gradients are accumulated using `+` (which creates a lazy addition op in the graph).

---

## What is similar to my NumPy implementation:

* **Manual Chain Rule Logic:** The underlying mathematical expressions inside `backward()` (e.g., matrix transposes $W^T \cdot dZ$, element-wise derivatives for ReLU/MSE) mirror the exact formulas written in our NumPy `MLP.backward()` implementation.
* **Gradient Storage:** Gradients are stored directly on the node/tensor structure as a `.grad` attribute after backward execution completes.
* **Topological Reversal:** Both rely on processing layers/nodes in strict reverse order of execution.

---

## What is Different:

* **Eager vs. Lazy Execution:** The NumPy MLP computes forward values and matrix operations immediately during function execution. tinygrad defers calculation, building an instruction graph that compiles into C, Metal, or CUDA kernels at runtime.
* **Dynamic Graph vs. Fixed Layers:** Our NumPy model relies on hardcoded index iteration over pre-allocated weight lists (`self.weight_matrix[i]`). tinygrad dynamically constructs an arbitrary graph for any flexible expression or layer type.
* **Context (`_ctx`) Object:** tinygrad uses an explicit execution context (`_ctx`) saved on each created Tensor node to store intermediate activation buffers required during the backward pass, decoupling the forward code from backprop state management.

---

## What I learned:

* **Minimalist Primitive Reductions:** Complex deep learning operations (Convolutions, Attention, Normalization) do not need individual backward functions if they can be expressed using a tiny set of basic primitives (Add, Mul, Sum, Reshape).
* **Graph-Level Gradient Efficiency:** Performing backpropagation on a symbolic computation graph allows optimizers and compilers to fuse backward operations into single GPU/accelerator kernels, dramatically reducing memory bandwidth bottlenecks compared to array-by-array execution.