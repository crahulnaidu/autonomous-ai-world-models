"""
optimizer_benchmark.py

A modular benchmarking suite to compare standard Gradient Descent against
Momentum Optimization on standard 2D non-convex benchmark functions.
"""

from typing import Callable, Dict, List, Tuple
import time
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# 1. Benchmark Objective Functions & Derivatives
# =====================================================================

def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Computes the Rosenbrock function value for 2D inputs.
    f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Global minimum is at (a, a^2) with value 0.
    """
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """Computes the gradient of the 2D Rosenbrock function."""
    df_dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    df_dy = 2 * b * (x[1] - x[0] ** 2)
    return np.array([df_dx, df_dy])


def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """
    Computes the 2D Rastrigin function value.
    f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
    Global minimum is at (0, 0) with value 0.
    """
    return A * 2 + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def rastrigin_grad(x: np.ndarray, A: float = 10.0) -> np.ndarray:
    """Computes the gradient of the 2D Rastrigin function."""
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


# =====================================================================
# 2. Optimization Algorithms
# =====================================================================

def gradient_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    init_x: np.ndarray,
    lr: float = 0.001,
    num_iterations: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Performs standard Gradient Descent optimization.

    Args:
        grad_fn: Function returning the gradient vector at position x.
        init_x: Initial parameter vector.
        lr: Learning rate (step size).
        num_iterations: Maximum number of iterations to execute.
        tol: Convergence tolerance for the gradient norm.

    Returns:
        Tuple containing final parameter vector and history trajectory.
    """
    x = init_x.copy().astype(np.float64)
    history = [x.copy()]

    for _ in range(num_iterations):
        grad = grad_fn(x)
        if np.linalg.norm(grad) < tol:
            break
        x -= lr * grad
        history.append(x.copy())

    return x, history


def momentum_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    init_x: np.ndarray,
    lr: float = 0.001,
    beta: float = 0.9,
    num_iterations: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Performs Gradient Descent with Momentum optimization.

    Args:
        grad_fn: Function returning the gradient vector at position x.
        init_x: Initial parameter vector.
        lr: Learning rate.
        beta: Momentum decay hyperparameter (typically 0.9).
        num_iterations: Maximum number of iterations.
        tol: Convergence tolerance for the gradient norm.

    Returns:
        Tuple containing final parameter vector and history trajectory.
    """
    x = init_x.copy().astype(np.float64)
    v = np.zeros_like(x)
    history = [x.copy()]

    for _ in range(num_iterations):
        grad = grad_fn(x)
        if np.linalg.norm(grad) < tol:
            break
        v = beta * v + lr * grad
        x -= v
        history.append(x.copy())

    return x, history


# =====================================================================
# 3. Benchmarking Utilities
# =====================================================================

def run_benchmark(
    func: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    init_x: np.ndarray,
    optimizers: Dict[str, Callable],
) -> Dict[str, dict]:
    """
    Executes multiple optimizers on a objective function, recording runtime,
    loss trajectory, and solution quality.

    Args:
        func: Objective scalar function.
        grad_fn: Function evaluating gradient.
        init_x: Initial starting point.
        optimizers: Mapping of optimizer names to callable runner functions.

    Returns:
        Dictionary mapping optimizer names to benchmark metrics and trajectories.
    """
    results = {}

    for name, opt_fn in optimizers.items():
        start_time = time.perf_counter()
        final_x, trajectory = opt_fn(grad_fn, init_x)
        elapsed_time = (time.perf_counter() - start_time) * 1000.0  # in ms

        loss_history = [func(pt) for pt in trajectory]

        results[name] = {
            "final_x": final_x,
            "final_loss": loss_history[-1],
            "trajectory": np.array(trajectory),
            "loss_history": loss_history,
            "elapsed_ms": elapsed_time,
            "iterations": len(trajectory) - 1,
        }

    return results


# =====================================================================
# 4. Plot Generation
# =====================================================================

def plot_benchmark_results(
    func: Callable[[np.ndarray], float],
    results: Dict[str, dict],
    bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -1.0, 3.0),
    title: str = "Optimizer Optimization Paths",
) -> None:
    """
    Plots the contour loss landscape along with optimizer paths and loss convergence curves.

    Args:
        func: Objective scalar function.
        results: Results dictionary generated by `run_benchmark`.
        bounds: (x_min, x_max, y_min, y_max) plot boundaries.
        title: Title of the visualization plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Contour Plot & Trajectories ---
    x_min, x_max, y_min, y_max = bounds
    X = np.linspace(x_min, x_max, 250)
    Y = np.linspace(y_min, y_max, 250)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z_grid = np.array([func(np.array([x, y])) for x, y in zip(X_grid.ravel(), Y_grid.ravel())]).reshape(X_grid.shape)

    # Log scale contours for sharp landscapes
    ax1.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(-1, 3, 20), cmap="jet", alpha=0.6)

    for name, res in results.items():
        traj = res["trajectory"]
        ax1.plot(traj[:, 0], traj[:, 1], "o-", label=f"{name} ({res['iterations']} iter)", markersize=3)
        ax1.plot(traj[0, 0], traj[0, 1], "ko", markersize=6)  # Start
        ax1.plot(traj[-1, 0], traj[-1, 1], "rx", markersize=8, markeredgewidth=2)  # End

    ax1.set_title(f"{title} - Trajectories")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Plot 2: Convergence Curves ---
    for name, res in results.items():
        ax2.plot(res["loss_history"], label=f"{name} (Final Loss: {res['final_loss']:.4e})")

    ax2.set_yscale("log")
    ax2.set_title("Loss Convergence (Log Scale)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# =====================================================================
# 5. Main Execution Entry Point
# =====================================================================

if __name__ == "__main__":
    init_point = np.array([-1.2, 1.0])
    max_iters = 2000
    learning_rate = 0.001

    # Define optimization routines
    opts = {
        "Standard GD": lambda g, x: gradient_descent(g, x, lr=learning_rate, num_iterations=max_iters),
        "Momentum (beta=0.9)": lambda g, x: momentum_descent(g, x, lr=learning_rate, beta=0.9, num_iterations=max_iters),
    }

    # Run benchmark on Rosenbrock
    print("Running Rosenbrock Benchmark...")
    rosen_results = run_benchmark(rosenbrock, rosenbrock_grad, init_point, opts)

    for opt_name, metrics in rosen_results.items():
        print(f"[{opt_name}] Execution Time: {metrics['elapsed_ms']:.2f} ms | Final Loss: {metrics['final_loss']:.6f}")

    # Plot results
    plot_benchmark_results(rosenbrock, rosen_results, bounds=(-1.5, 1.5, -0.5, 1.5), title="Rosenbrock Function")