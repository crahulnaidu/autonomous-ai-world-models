Here are the solutions for the Day 11 Exercises.

---

### Exercise 1 — Chain Rule

Given:
$z = 3x + 2$
$a = z^2$
$L = 2a$

Using the Chain Rule:


$$\frac{dL}{dx} = \frac{dL}{da} \cdot \frac{da}{dz} \cdot \frac{dz}{dx}$$

1. **Local gradients:**
* $\frac{dL}{da} = 2$
* $\frac{da}{dz} = 2z = 2(3x + 2)$
* $\frac{dz}{dx} = 3$


2. **Multiply local gradients:**

$$\frac{dL}{dx} = 2 \cdot 2(3x + 2) \cdot 3 = 12(3x + 2) = 36x + 24$$



---

### Exercise 2 — ReLU

For $f(x) = \text{ReLU}(x) = \max(0, x)$, the local gradient is $\frac{df}{dx} = 1$ if $x > 0$ else $0$:

* **$x = -3$:** $\frac{df}{dx} = \mathbf{0.0}$
* **$x = 0.5$:** $\frac{df}{dx} = \mathbf{1.0}$
* **$x = 7$:** $\frac{df}{dx} = \mathbf{1.0}$

---

### Exercise 3 — Matrix Dimensions (Mandatory)

Given:

* $X \in (64, 128)$
* $W \in (128, 256)$
* $b \in (1, 256)$
* $dZ \in (64, 256)$

Linear forward formulation: $Z = XW + b$

1. **$Z$ (Forward activation output):**
* $(64, 128) \times (128, 256) + (1, 256) \implies \mathbf{(64, 256)}$


2. **$dW$ (Weight gradient):**
* Formula: $dW = X^T \cdot dZ$
* Dimensions: $(128, 64) \times (64, 256) \implies \mathbf{(128, 256)}$


3. **$db$ (Bias gradient):**
* Formula: $db = \sum_{\text{batch}} dZ$
* Summing along axis 0 across the batch of 64 $\implies \mathbf{(1, 256)}$


4. **$dX$ (Input gradient):**
* Formula: $dX = dZ \cdot W^T$
* Dimensions: $(64, 256) \times (256, 128) \implies \mathbf{(64, 128)}$



---

### Exercise 4 — Numerical Gradient

```python
import numpy as np

def numerical_gradient(f, x, epsilon=1e-6):
    """
    Computes the numerical gradient of function f at x using centered finite differences.
    f: callable function returning a scalar
    x: numpy array
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]
        
        # f(x + eps)
        x[idx] = orig_val + epsilon
        fx_plus = f(x)
        
        # f(x - eps)
        x[idx] = orig_val - epsilon
        fx_minus = f(x)
        
        # Centered finite difference formula
        grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        # Restore original value
        x[idx] = orig_val
        it.iternext()
        
    return grad


# --- Verification against analytical gradient ---
if __name__ == "__main__":
    np.random.seed(42)
    x_val = np.array([2.0, -3.0, 0.5])
    
    # Example function: L = sum(3x^2 + 2x)
    f = lambda x: np.sum(3 * x**2 + 2 * x)
    
    # Analytical gradient: dL/dx = 6x + 2
    analytical_grad = 6 * x_val + 2
    
    # Numerical gradient
    num_grad = numerical_gradient(f, x_val)
    
    # Relative Error calculation
    rel_error = np.abs(analytical_grad - num_grad) / (np.abs(analytical_grad) + np.abs(num_grad) + 1e-15)
    
    print("Analytical Grad :", analytical_grad)
    print("Numerical Grad  :", num_grad)
    print("Relative Errors :", rel_error)
    print("Status          :", "SUCCESS" if np.all(rel_error < 1e-7) else "FAILURE")

```

---

### Exercise 5 — Research Thinking

As sequence length ($S$) and batch size ($B$) increase, accelerator memory requirements scale **linearly to quadratically ($\mathcal{O}(B \cdot S)$ to $\mathcal{O}(B \cdot S^2)$)** per layer during training.

From an **LLM hardware accelerator architecture perspective**, this creates severe structural bottlenecks:

1. **SRAM Buffer Overflow to HBM Off-Chip Transfers:**
High-speed on-chip SRAM/scratchpads cannot store the massive intermediate activation footprint across layers. The hardware is forced to continuously spill activations out to external High Bandwidth Memory (HBM).
2. **The Memory Bandwidth Wall (Bus Stalls):**
When the backward pass executes, those cached activations must be fetched back from HBM into the compute engine's SRAM buffers. This turns backpropagation into an **HBM bandwidth-bound** operation where matrix multiply execution units (systolic arrays) sit idle waiting for memory bus transfers.
3. **Quadratic Scaling in Attention ($S^2$ Memory Bound):**
In Transformer layers, storing attention weight matrices ($Q K^T \in \mathbb{R}^{B \times H \times S \times S}$) for backpropagation scales quadratically with sequence length $S$.
4. **Architectural Hardware Mitigation Strategies:**
To prevent out-of-memory (OOM) crashes and compute stalls, hardware-software co-design relies on:
* **Activation Checkpointing:** Discarding intermediate SRAM activations during the forward pass and recomputing them on-demand during the backward pass (trading FLOPs for memory bandwidth).
* **Kernel Fusion (e.g., FlashAttention):** Re-structuring GPU/ASIC compute tiling to compute attention gradients in fast SRAM blocks, completely avoiding round-trip HBM writes for intermediate $S \times S$ attention maps.