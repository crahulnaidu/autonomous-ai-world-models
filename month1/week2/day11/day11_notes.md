# Day 11 Theory Notes: Backpropagation & Computational Graphs

**File:** `day11_notes.md`

---

## 1. Core Mechanics & Definitions

### What is Backpropagation?

Backpropagation (Backward Propagation of Errors) is an efficient algorithm for calculating the exact analytical gradients of a scalar loss function with respect to every trainable parameter (weights and biases) in a neural network. It applies the differential calculus **chain rule** in reverse (from output to input) across a computational graph to perform reverse-mode automatic differentiation.

### Forward Propagation vs. Backward Propagation

* **Forward Propagation:** The sequential evaluation of matrix multiplications, bias additions, and activation functions moving from input $X$ to output $\hat{Y}$, concluding with loss $L$ computation. Inputs are transformed into intermediate activations and cached in memory.
* **Backward Propagation:** The reverse pass starting from loss $L$ moving backward toward input $X$. It uses cached forward values to compute partial derivatives of $L$ with respect to every internal activation and weight parameter.

### Computational Graphs

A computational graph represents complex mathematical expressions as a Directed Acyclic Graph (DAG):

* **Nodes:** Operations or functions (e.g., $+$, $\times$, $\text{ReLU}$, $\text{MatMul}$).
* **Edges:** Tensors / variables flowing between operations.
By breaking a neural network down into primitive computational graph nodes, gradient derivation is simplified into evaluating isolated local node derivatives.

### Chain Rule

If a scalar $z$ depends on $y$, which depends on $x$ ($z = f(y)$ where $y = g(x)$), the derivative of $z$ with respect to $x$ is the product of local derivatives along the path:


$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

### Local Gradient vs. Upstream Gradient

* **Upstream Gradient ($\frac{\partial L}{\partial y}$):** The incoming error gradient backpropagated from deeper layers further down the computational graph.
* **Local Gradient ($\frac{\partial y}{\partial x}$):** The derivative of a node's output with respect to its local input, evaluated using values saved during forward propagation.
* **Downstream Gradient:** Calculated via element-wise / matrix multiplication:

$$\text{Downstream Gradient} = \text{Upstream Gradient} \times \text{Local Gradient}$$



### Gradient Flow

Gradient flow refers to the magnitude and behavior of error signals as they move backward through successive layers of a deep network. Smooth, uninhibited gradient flow allows all layers to update effectively. Bottlenecks (vanishing/exploding gradients) halt learning in early layers.

---

## 2. Key Derivatives

### Linear Unit: $z = w \cdot x + b$

* **Derivative w.r.t $w$:** $\frac{\partial z}{\partial w} = x$
* **Derivative w.r.t $x$:** $\frac{\partial z}{\partial x} = w$
* **Derivative w.r.t $b$:** $\frac{\partial z}{\partial b} = 1$

### ReLU Activation: $\sigma(x) = \max(0, x)$

$$\frac{d}{dx} \text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$


*(Note: At $x = 0$, the subgradient is typically set to $0$ or $1$ in practice).*

### Sigmoid Activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$

$$\frac{d}{dx} \sigma(x) = \sigma(x)\left(1 - \sigma(x)\right)$$

### Mean Squared Error (MSE): $L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)$$

---

## 3. Dense-Layer Matrix Gradients

For a dense linear layer with input matrix $X \in \mathbb{R}^{N \times d_{\text{in}}}$, weights $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, and bias $b \in \mathbb{R}^{d_{\text{out}} \times 1}$:

$$\text{Forward Pass: } Z = X W^T + b^T \quad \text{or in column vector notation: } Z^T = W X^T + b$$

Given incoming upstream gradient $dZ = \frac{\partial L}{\partial Z} \in \mathbb{R}^{N \times d_{\text{out}}}$:

1. **Weight Gradient ($dW$):**

$$dW = dZ^T X \quad \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$


2. **Bias Gradient ($db$):**

$$db = \sum_{i=1}^{N} dZ_{i, :} \quad \in \mathbb{R}^{d_{\text{out}} \times 1} \quad \text{(Sum of upstream gradients across batch size } N\text{)}$$


3. **Input Gradient ($dX$):**

$$dX = dZ W \quad \in \mathbb{R}^{N \times d_{\text{in}}}$$



---

## 4. Memory Dynamics: Intermediate Activations & Training vs. Inference

### Why Intermediate Activations Are Cached

To compute local gradients during backpropagation (e.g., $dW = dZ^T \cdot X$), the backward pass requires exact values of intermediate layer inputs ($X$, $Z$, activations $A$) computed during the forward pass. These tensors must be retained in memory until their corresponding layer's backward step executes.

### Why Training Needs Significantly More Memory Than Inference

* **Inference Mode:** Only requires memory for the input, current layer activations, and model weights. Once a layer finishes computation, its input tensor can be instantly overwritten / freed.
* **Training Mode:** Must hold **all** forward-pass intermediate activations across all layers in memory simultaneously, alongside model weights, gradients ($dW, db$), and optimizer states (e.g., Adam's 1st and 2nd momentum vectors $m_t, v_t$).

---

## 5. Hardware Accelerator Connection (AI Accelerators & Custom Chips)

1. **Matrix Multiplication Symmetry:** Both forward pass ($W \cdot X$) and backpropagation ($dZ \cdot W$, $dZ^T \cdot X$) consist primarily of General Matrix Multiplications (GEMMs). Systolic arrays in GPUs and TPUs execute both passes using the exact same Hardware Processing Element (PE) MAC grids by transposing matrix inputs.
2. **The Memory Bandwidth Wall (HBM Bound):** Backpropagation is constrained heavily by memory bandwidth rather than compute limits. Caching activations across deep networks leads to massive High Bandwidth Memory (HBM) footprints. Hardware optimizations (e.g., FlashAttention, activation recomputation/checkpointing, and operator fusion) directly combat this memory bottleneck by re-computing local activations on-chip (SRAM) instead of reading them back from off-chip HBM.
3. **Custom Accelerator Buffer Design:** When designing domain-specific training accelerators (e.g., M.Tech research chips), memory hierarchy allocation (SRAM sizes) must balance activation buffer space against throughput to prevent memory stalls during backpropagation cycles.