import numpy as np

# ==========================================
# 1. MLP Implementation
# ==========================================
class MLP:
    def __init__(self, input_dim: int, hidden_layers: int, hidden_layer_dim: int, output_dim: int, lr: float):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim
        self.lr = lr

        self.weight_matrix = []
        self.output_activations = []
        self.input = []
        self.gradients = []

        total_matrices = 1 + self.hidden_layers

        # He / Kaiming Normal Initialization for ReLU
        for i in range(total_matrices):
            if i == 0:
                scale = np.sqrt(2.0 / self.input_dim)
                self.weight_matrix.append(np.random.randn(self.hidden_layer_dim, self.input_dim + 1) * scale)
            elif i < total_matrices - 1:
                scale = np.sqrt(2.0 / self.hidden_layer_dim)
                self.weight_matrix.append(np.random.randn(self.hidden_layer_dim, self.hidden_layer_dim + 1) * scale)
            else:
                scale = np.sqrt(2.0 / self.hidden_layer_dim)
                self.weight_matrix.append(np.random.randn(self.output_dim, self.hidden_layer_dim + 1) * scale)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)  # ReLU

    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def forward_propogate(self, X: np.ndarray) -> np.ndarray:
        self.input = []
        self.output_activations = []

        # Add bias feature to X -> Shape: (N, input_dim + 1)
        X_b = np.c_[X, np.ones((X.shape[0], 1))]
        A = X_b.T  # Shape: (input_dim + 1, N)
        self.output_activations.append(A)

        for i in range(len(self.weight_matrix)):
            Z = self.weight_matrix[i] @ A
            self.input.append(Z)

            if i < len(self.weight_matrix) - 1:
                A = self.activation(Z)
                A = np.r_[A, np.ones((1, A.shape[1]))]  # Add bias row
                self.output_activations.append(A)

        return Z.T  # Shape: (N, output_dim)

    def compute_loss(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        # Mean Squared Error (MSE): L = (1/N) * sum((Y_pred - Y)^2)
        return float(np.mean((Y_pred - Y) ** 2))

    def backward(self, X: np.ndarray, Y: np.ndarray) -> None:
        N = X.shape[0]
        self.gradients = [None] * len(self.weight_matrix)

        # Output layer forward output Z_out
        Z_out = self.input[-1]
        
        # Derivative of MSE Loss w.r.t Z_out
        dZ = 2 * (Z_out - Y.T) / N
        self.gradients[-1] = dZ @ self.output_activations[-1].T

        # Backpropagation through hidden layers
        for i in range(len(self.weight_matrix) - 2, -1, -1):
            W_nobias = self.weight_matrix[i + 1][:, :-1]
            dA = W_nobias.T @ dZ
            dZ = dA * self.activation_derivative(self.input[i])
            self.gradients[i] = dZ @ self.output_activations[i].T


# ==========================================
# 2. Gradient Verification Functions
# ==========================================
def compute_numerical_gradients(mlp: MLP, X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-6) -> list:
    """Computes numerical gradients using centered finite difference."""
    num_gradients = []

    for l, W in enumerate(mlp.weight_matrix):
        grad_W = np.zeros_like(W)
        
        # Iterate over every element in weight matrix W
        it = np.nditer(W, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            original_val = W[idx]

            # 1. Loss for W + epsilon
            W[idx] = original_val + epsilon
            Y_pred_plus = mlp.forward_propogate(X)
            loss_plus = mlp.compute_loss(Y_pred_plus, Y)

            # 2. Loss for W - epsilon
            W[idx] = original_val - epsilon
            Y_pred_minus = mlp.forward_propogate(X)
            loss_minus = mlp.compute_loss(Y_pred_minus, Y)

            # 3. Finite Difference Gradient
            grad_W[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original weight
            W[idx] = original_val
            it.iternext()

        num_gradients.append(grad_W)

    return num_gradients


def relative_error(grad_analytical: np.ndarray, grad_numerical: np.ndarray) -> float:
    """Computes relative error: ||g_a - g_n|| / (||g_a|| + ||g_n||)."""
    numerator = np.linalg.norm(grad_analytical - grad_numerical)
    denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ==========================================
# 3. Run Gradient Check
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    # Small network and dataset to ensure quick calculation and avoid ReLU kink issues
    X = np.random.randn(20, 3)
    Y = np.random.randn(20, 1)

    mlp = MLP(input_dim=3, hidden_layers=1, hidden_layer_dim=4, output_dim=1, lr=0.01)

    # 1. Compute Analytical Gradients via Backpropagation
    Y_pred = mlp.forward_propogate(X)
    mlp.backward(X, Y)
    analytical_grads = mlp.gradients

    # 2. Compute Numerical Gradients via Finite Difference
    epsilon = 1e-6
    numerical_grads = compute_numerical_gradients(mlp, X, Y, epsilon=epsilon)

    # 3. Print Sample Parameter Comparisons
    print("=" * 65)
    print("           GRADIENT VERIFICATION RESULTS")
    print("=" * 65)

    for layer_idx in range(len(mlp.weight_matrix)):
        W_analytical = analytical_grads[layer_idx]
        W_numerical = numerical_grads[layer_idx]
        
        print(f"\n--- Layer {layer_idx + 1} (Weight Matrix Shape: {mlp.weight_matrix[layer_idx].shape}) ---")
        
        # Pick 3 random or representative indices to inspect
        indices_to_sample = [(0, 0), (0, 1), (1, 0)] if W_analytical.shape[0] > 1 else [(0, 0)]
        for r, c in indices_to_sample:
            if r < W_analytical.shape[0] and c < W_analytical.shape[1]:
                g_a = W_analytical[r, c]
                g_n = W_numerical[r, c]
                rel_err_param = abs(g_a - g_n) / (abs(g_a) + abs(g_n) + 1e-15)

                print(f"Parameter: W{layer_idx + 1}[{r},{c}]")
                print(f"  Analytical gradient : {g_a:12.8f}")
                print(f"  Numerical gradient  : {g_n:12.8f}")
                print(f"  Relative error      : {rel_err_param:12.8e}\n")

    # 4. Overall Global Relative Error
    a_flat = np.concatenate([g.ravel() for g in analytical_grads])
    n_flat = np.concatenate([g.ravel() for g in numerical_grads])
    overall_err = relative_error(a_flat, n_flat)

    print("=" * 65)
    print(f"Overall Global Relative Error: {overall_err:12.8e}")
    
    # Interpretation Rule of Thumb
    if overall_err < 1e-7:
        print("Status: SUCCESS - Backpropagation gradients are verified correct!")
    elif overall_err < 1e-4:
        print("Status: WARNING - Relative error is moderate. (Check for ReLU non-differentiable points at 0).")
    else:
        print("Status: FAILURE - Backpropagation implementation has bugs!")
    print("=" * 65)