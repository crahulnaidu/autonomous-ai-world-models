## Theory

1.An objective function is a mathematical function that tells how good or bad a modelis.

2.Optimization involve's tuning the parameters of a model to minimize or maximize an objective function.Learning on the other hand involve' s recognizing and identifying patterns in data to gain meaningful insights.Optimization is a part of learning.

3.local minimum is the point at which a function has the minimum value around the epsilon neighborhood,where epsilon>0.

Global minimum is the point x* such that f(x*)<=f(x) for all values of x.

4.Gradient give's the direction of steepest ascent , so moving oppsite to it reduce's the function value by the maximum amount leading to the global minimum.

5.Because of the many parameter's involved in a neural network and non linear activation functions the cost function has many local minimum.

## Math.

1.6x

2.[2x,2y]

3.gradient at (2,3)=(4,6)

## Coding.

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def grad_f(x):
    return 2*x

# Fix random seed for consistent testing shapes
np.random.seed(42)
x_init = np.random.uniform(-15, 15)

lr_arr = [0.01, 0.1, 0.5, 1.0]
color_arr = ['red', 'green', 'blue', 'black']
iterations = 50
diff = 1e-5

plt.figure(figsize=(10, 6))

# FIXED: 'ind' gets the index (0,1,2,3), 'lr' gets the actual float value
for ind, lr in enumerate(lr_arr):
    x_hist = []
    y_hist = []

    x = x_init
    x_hist.append(x)
    y_hist.append(f(x))

    print(f"\n--- Testing Learning Rate: {lr} ---")
    for epoch in range(iterations):
        grad = grad_f(x)
        x_new = x - lr * grad

        if abs(f(x_new) - f(x)) < diff:
            print(f"Early stoppage criteria met at epoch {epoch}!")
            print(f"Final x value is {x_new:.5f}")
            break

        # FIXED: Update x first, then store the new state to history arrays
        x = x_new
        x_hist.append(x)
        y_hist.append(f(x))

    x_hist = np.array(x_hist)
    y_hist = np.array(y_hist)

    # FIXED: Indexing color_arr with the integer 'ind', and using 'lr' in the label
    plt.plot(x_hist, y_hist, label=f"learning rate={lr}", color=color_arr[ind], marker='o', alpha=0.6)

# Plot decoration parameters
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title(f"Gradient Descent Convergence with Early Stopping (Start $x_0$ = {x_init:.2f})")
plt.xlabel("Value of x")
plt.ylabel("Loss $f(x)$")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

## Quantum

1.Entanglement is a non-classical quantum correlation such that the measurement of one qubit affects the other.

2.Bell states are entangled states and form the basis for all 2 qubit quantum states.

3.No ,because in order to send information about the qubit we have to use classical communication which is bound by special relativity.


 
