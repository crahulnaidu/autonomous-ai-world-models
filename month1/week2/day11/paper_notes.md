# CS231n Optimization & Gradient Mechanics Notes

## 1. Gradient Checking

* **What it is:** A diagnostic technique that verifies hand-coded backpropagation accuracy by comparing analytical gradients ($\nabla_{\text{analytical}}$) with finite-difference numerical gradients ($\nabla_{\text{numerical}} = \frac{L(w+\epsilon) - L(w-\epsilon)}{2\epsilon}$, where $\epsilon \approx 10^{-5}$).
* **Why it's useful:** Catches silent implementation bugs where backprop runs without throwing errors but calculates incorrect updates. Essential for verifying custom CUDA kernels and hand-coded operators.
* **Metric:** Evaluated via Relative Error: $\frac{\Vert{}\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\Vert{}}{\Vert{}\nabla_{\text{analytical}}\Vert{} + \Vert{}\nabla_{\text{numerical}}\Vert{}}$. Values $< 10^{-7}$ signal correctness.

## 2. Learning-Rate Dynamics

* **Too High:** Overshoots steep loss surfaces; leads to exploding gradients, instability, or immediate `NaN` loss.
* **High:** Rapid initial drop, but oscillates around local minima without converging to optimal loss.
* **Too Low:** Extremely slow linear progress; gets trapped in shallow local minima or saddle points.
* **Optimal:** Smooth, exponential loss decay settling at a stable minimum.

## 3. Common Training Problems

* **Exploding/Vanishing Gradients:** Deep matrix multiplications scale gradients exponentially to $\infty$ or shrink them to $0$.
* **Kinks (Non-differentiability):** Functions like ReLU ($\max(0, x)$) have kinks at $x=0$, causing temporary numerical/analytical gradient mismatches when $x \pm \epsilon$ crosses $0$.
* **Dead ReLUs:** Aggressive updates push weights into regions where $x \le 0$ across all inputs, permanently zeroing their gradients.
* **Saddle Points & Ill-Conditioning:** High-dimensional zero-gradient plateaus halt training; uneven landscape curvature causes updates to bounce across ravines instead of moving toward the minimum.

## 4. Key Optimization Insights

* **Adaptive Momentum:** Adam/Nesterov filter mini-batch gradient noise and scale updates per-parameter using historical squared-gradient averages ($v_t$).
* **Numerical Precision Limits:** Setting $\epsilon$ too small in finite differences causes floating-point cancellation errors; setting it too large introduces truncation error.

---

## 5. Connection to My Research

* **Neural-Network Training:** Direct understanding of gradient flow drives proper parameter initialization (He/Kaiming) and skip-connection architectures (ResNets).
* **LLM Training:** Informs gradient clipping techniques to prevent attention layer loss spikes and dynamic loss scaling in FP16/BF16 training.
* **AI Accelerators:** Forward ($W \cdot X$) and backward ($W^T \cdot dZ$, $dZ \cdot A^T$) steps map directly to systolic array MAC units. Intermediate activation caching creates severe HBM memory bounds.
* **M.Tech Accelerator Research:** Used to verify custom low-precision/fixed-point backpropagation hardware against FP32 baselines, optimize SRAM activation buffer sizing, and determine hardware quantization limits for gradient accumulators.