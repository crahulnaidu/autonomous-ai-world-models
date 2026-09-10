import numpy as np

class MLP:
  def __init__(self, input_dim: int, hidden_layers: int, hidden_layer_dim: int, output_dim: int, lr: float):
    self.input_dim = input_dim
    self.hidden_layers = hidden_layers
    self.hidden_layer_dim = hidden_layer_dim
    self.output_dim = output_dim
    self.weight_matrix = []
    self.output_activations = []
    self.input = []
    self.lr = lr
    self.gradients = []

    total_matrices = 1 + self.hidden_layers

    for i in range(total_matrices):
      if i == 0:
        self.weight_matrix.append(np.random.randn(self.hidden_layer_dim, self.input_dim + 1) * np.sqrt(2.0 / self.input_dim))
      elif i < total_matrices - 1:
        self.weight_matrix.append(np.random.randn(self.hidden_layer_dim, self.hidden_layer_dim + 1) * np.sqrt(2.0 / self.hidden_layer_dim))
      else:
        self.weight_matrix.append(np.random.randn(self.output_dim, self.hidden_layer_dim + 1) * np.sqrt(2.0 / self.hidden_layer_dim))

  def activation(self, x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

  def activation_derivative(self, x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

  def forward_propogate(self, X: np.ndarray) -> np.ndarray:
    # Clear cached forward values for the current pass
    self.input = []
    self.output_activations = []

    # Add bias to input X -> Shape: (N, input_dim + 1)
    X_b = np.c_[X, np.ones((X.shape[0], 1))]
    A = X_b.T  # Transpose to shape (input_dim + 1, N)
    self.output_activations.append(A)

    for i in range(len(self.weight_matrix)):
      Z = self.weight_matrix[i] @ A
      self.input.append(Z)

      if i < len(self.weight_matrix) - 1:
        A = self.activation(Z)
        A = np.r_[A, np.ones((1, A.shape[1]))]  # Append bias row
        self.output_activations.append(A)

    # Output layer (no activation applied here)
    return Z.T

  def backward(self, X: np.ndarray, Y: np.ndarray) -> None:
    N = X.shape[0]
    self.gradients = [None] * len(self.weight_matrix)

    # Re-run forward pass or assume current output Z_out
    Z_out = self.input[-1]  # Shape: (output_dim, N)
    
    # 1. Output Layer Gradient (MSE Derivative)
    # Loss = (1/N) * ||Y_pred - Y||^2  => dL/dZ_out = 2 * (Y_pred - Y)^T / N
    dZ = 2 * (Z_out - Y.T) / N

    # Gradient for output layer weights: dW = dZ @ A_prev^T
    self.gradients[-1] = dZ @ self.output_activations[-1].T

    # 2. Backpropagate through hidden layers
    for i in range(len(self.weight_matrix) - 2, -1, -1):
      # Backpropagate error through weights (excluding bias row from previous weights)
      W_nobias = self.weight_matrix[i + 1][:, :-1]
      dA = W_nobias.T @ dZ

      # Apply derivative of ReLU on linear input Z_i
      dZ = dA * self.activation_derivative(self.input[i])

      # Compute weight gradient for layer i
      self.gradients[i] = dZ @ self.output_activations[i].T

  def update(self) -> None:
    # Update weight matrices using stored gradients
    for i in range(len(self.weight_matrix)):
      self.weight_matrix[i] -= self.lr * self.gradients[i]





nn=MLP(4,2,8,8,1)

X=np.random.randn(100,4)
Y=X@np.random.randn(4,1)+np.random.rand(100,1)

nn.forward_propogate(X)
nn.backward(X,Y)
nn.update()      